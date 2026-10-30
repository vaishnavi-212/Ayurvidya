"""Hybrid retrieval = dense (meaning) + BM25 (exact Sanskrit forms), fused with Reciprocal Rank Fusion,
then an optional cross-encoder reranker, then (for chat) de-duplication + MMR for variety.

    RRF(d) = sum over retrievers r of 1 / (k + rank_r(d)),  k = 60   (Cormack, Clarke & Buettcher, 2009)

RRF uses ranks, not raw scores, so cosine similarities and BM25 scores never have to be put on one scale.
Optional extra ranked lists join the same fusion: HyDE queries, and the knowledge graph (kg.py).

Overlapping verses: one commentary passage often covers a verse pair (31-32) and the same idea is repeated
in parallel texts, so near-identical chunks come back twice. `dedupe_mmr` drops near-duplicates
(character 4-gram Jaccard > 0.85) and picks the rest with Maximal Marginal Relevance.
"""
from __future__ import annotations

from dataclasses import dataclass

from .index import KnowledgeBase, bm25_tokens
from .schema import Chunk, Hit
from .text import fold_for_search, to_devanagari


def rrf(rank_lists: list[list[int]], k: int = 60, weights: list[float] | None = None) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for n, lst in enumerate(rank_lists):
        w = weights[n] if weights else 1.0
        for r, idx in enumerate(lst, start=1):
            scores[idx] = scores.get(idx, 0.0) + w / (k + r)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


class Reranker:  # pragma: no cover - needs a downloaded model
    """Cross-encoder reranker, e.g. BAAI/bge-reranker-v2-m3 (568M, multilingual, XLM-RoBERTa-large base)."""

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError("pip install sentence-transformers  (needed for the reranker)") from e
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, chunks: list[Chunk]) -> list[float]:
        return [float(s) for s in self.model.predict([(query, c.text) for c in chunks])]


def shingles(text: str) -> set[str]:
    t = fold_for_search(text).replace(" ", "")
    return {t[i:i + 4] for i in range(len(t) - 3)} or {t}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def dedupe_mmr(hits: list[Hit], k: int, lam: float = 0.7, dup: float = 0.85, against=()) -> list[Hit]:
    """Drop hits that repeat each other (or repeat a text in `against`, e.g. the pinned verse), then
    choose k with MMR: score = lam * relevance - (1 - lam) * max similarity to what is already chosen."""
    seen = [shingles(t) for t in against]
    kept: list[tuple[Hit, set[str]]] = []
    for h in hits:
        sh = shingles(h.chunk.text)
        if any(jaccard(sh, o) > dup for o in seen) or any(jaccard(sh, o) > dup for _, o in kept):
            continue
        kept.append((h, sh))
    n = max(len(kept), 1)
    rel = {id(h): 1 - i / n for i, (h, _) in enumerate(kept)}
    chosen: list[tuple[Hit, set[str]]] = []
    while kept and len(chosen) < k:
        best = max(kept, key=lambda x: lam * rel[id(x[0])] - (1 - lam) * max(
            (jaccard(x[1], c) for _, c in chosen), default=0.0))
        chosen.append(best)
        kept.remove(best)
    return [Hit(h.chunk, h.score, r) for r, (h, _) in enumerate(chosen, 1)]


@dataclass
class RetrievalConfig:
    dense_k: int = 20
    sparse_k: int = 20
    rrf_k: int = 60
    final_k: int = 6
    graph_weight: float = 2.0     # the graph votes only when confident, and then counts double


class HybridRetriever:
    def __init__(self, kb: KnowledgeBase, cfg: RetrievalConfig | None = None, reranker: Reranker | None = None,
                 expander=None, graph=None, dedupe: bool = True):
        self.kb = kb
        self.cfg = cfg or RetrievalConfig()
        self.reranker = reranker
        self.expander = expander      # query.QueryExpander or None
        self.graph = graph            # kg.KnowledgeGraph or None
        self.dedupe = dedupe

    def search(self, query: str, k: int | None = None, source_types=None, exclude_ids=None,
               exclude_shlokas=None, mode: str = "hybrid", diverse: bool = False, rerank: bool = True,
               extra_queries=(), use_graph: bool = True, expand: bool = True) -> list[Hit]:
        """mode: 'hybrid' | 'dense' | 'sparse' (the last two exist for the ablation experiment).
        diverse=True removes near-duplicates and picks with MMR (used by the chatbot)."""
        k = k or self.cfg.final_k
        q = to_devanagari(query)
        qx = self.expander.expand(q) if (self.expander and expand) else q
        allowed = self.kb.mask(source_types, exclude_ids, exclude_shlokas)
        lists: list[list[int]] = []
        weights: list[float] = []
        if mode in ("hybrid", "dense"):
            qv = self.kb.embedder.embed_query(qx)
            lists.append([i for i, _ in self.kb.dense.search(qv, self.cfg.dense_k, allowed, source_types)])
            weights.append(1.0)
        if mode in ("hybrid", "sparse"):
            lists.append([i for i, _ in self.kb.bm25.search(bm25_tokens(qx), self.cfg.sparse_k, allowed)])
            weights.append(1.0)
            for extra in extra_queries:                       # HyDE: a hypothetical answer as a 2nd query
                lists.append([i for i, _ in self.kb.bm25.search(bm25_tokens(extra), self.cfg.sparse_k, allowed)])
                weights.append(1.0)
        if self.graph is not None and use_graph and mode == "hybrid":
            g = self.graph.rank_chunks(query, self.kb, allowed)
            if g:
                lists.append(g)
                weights.append(self.cfg.graph_weight)
        keep = [n for n, lst in enumerate(lists) if lst]
        lists, weights = [lists[n] for n in keep], [weights[n] for n in keep]
        if not lists:
            return []
        fused = rrf(lists, self.cfg.rrf_k, weights) if len(lists) > 1 else [(i, 1.0 / (self.cfg.rrf_k + r))
                                                                   for r, i in enumerate(lists[0], 1)]
        use_rerank = self.reranker is not None and rerank
        cand = fused[: k * 3] if (use_rerank or diverse) else fused[:k]
        if use_rerank and cand:
            chunks = [self.kb.chunks[i] for i, _ in cand]
            scores = self.reranker.rerank(q, chunks)
            cand = sorted(zip([i for i, _ in cand], scores), key=lambda x: -x[1])
        hits = [Hit(self.kb.chunks[i], s, r) for r, (i, s) in enumerate(cand, 1)]
        if diverse and self.dedupe:
            return dedupe_mmr(hits, k)
        return hits[:k]

    def context_for_step(self, shloka_id: str, query: str, source_types, k: int, step: int | None = None,
                         n_examples: int = 2, pinned_types=("mula", "commentary"),
                         exclude_shlokas_for_gold: bool = True) -> list[Hit]:
        """Context for one step of one shloka: structured lookup FIRST, semantic search SECOND.

        1. Pinned: every chunk explicitly linked to this shloka whose type the step uses (its own mula
           text, Dalhana's comment on it). These are always relevant, so we fetch them by id, not by luck.
        2. Searched: the remaining slots are filled by hybrid search (kosha, dhatu, cross-texts, examples).
        3. Leakage guard: 'gold' examples of THIS shloka are excluded, so the system never copies the answer
           it will later be evaluated against."""
        pinned = [c for c in self.kb.chunks_for_shloka(shloka_id)
                  if c.source_type in pinned_types and c.source_type in source_types]
        hits = [Hit(c, 1.0, r) for r, c in enumerate(pinned, 1)]
        seen = {c.id for c in pinned}
        remaining = max(k - len(hits), 0)
        if remaining:
            excl = [shloka_id] if exclude_shlokas_for_gold else []
            gold_free = [t for t in source_types if t != "gold"]
            searched: list[Hit] = []
            if gold_free:
                searched += self.search(query, remaining, source_types=gold_free, exclude_ids=seen)
            if "gold" in source_types and n_examples:   # few-shot examples of the SAME step, OTHER verses
                other_steps = [c.id for c in self.kb.chunks
                               if c.source_type == "gold" and step is not None and c.meta.get("step") != step]
                searched += self.search(query, n_examples, source_types=["gold"],
                                        exclude_ids=other_steps, exclude_shlokas=excl)
            if self.dedupe:   # a searched passage that repeats the pinned verse/commentary adds nothing
                pinned_sh = [shingles(c.text) for c in pinned]
                searched = [h for h in searched if h.chunk.source_type == "gold"
                            or not any(jaccard(shingles(h.chunk.text), p) > 0.85 for p in pinned_sh)]
            for h in searched:
                if h.chunk.id not in seen:
                    seen.add(h.chunk.id)
                    hits.append(Hit(h.chunk, h.score, len(hits) + 1))
        return hits