"""Indexes: an exact dense vector store, a BM25 keyword index, and the KnowledgeBase that ties them
to the chunk list and persists everything under data/kb/.

Why an exact NumPy store by default: Chapter 1 + its commentary + lexicon entries is a few thousand
chunks. A brute-force dot product over 5,000 x 1,024 float32 vectors takes about a millisecond, has
perfect recall (approximate indexes such as HNSW can miss neighbours) and needs no server.
Switch `vector_store: chroma` in settings when the corpus grows to whole Samhitas.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

from .embed import CachedEmbedder, Embedder, load_embedder, make_embedder
from .ingest import load_chunks, save_chunks
from .schema import Chunk
from .text import fold_for_search


# ------------------------------------------------------------------ dense
class NumpyVectorStore:
    def __init__(self, ids: list[str], matrix):
        if matrix.shape[0] != len(ids):
            raise ValueError("ids and matrix length differ")
        self.ids = ids
        self.m = matrix.astype(np.float32)      # dense ndarray or scipy sparse CSR (TF-IDF)

    def search(self, q: np.ndarray, k: int, allowed: np.ndarray | None = None,
               source_types=None) -> list[tuple[int, float]]:
        scores = np.asarray(self.m @ q.astype(np.float32)).ravel()
        if allowed is not None:
            scores = np.where(allowed, scores, -np.inf)
        k = min(k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]

    @property
    def dim(self) -> int:
        return int(self.m.shape[1])


class ChromaVectorStore:  # pragma: no cover - optional backend (pip install chromadb)
    """Persistent Chroma collection (cosine space) with metadata, so a search can be filtered by source
    type inside the database. Rebuilt automatically when the chunk set changes (sha in the metadata)."""

    def __init__(self, ids: list[str], matrix: np.ndarray | None, path: Path, chunks=None, sha: str = "",
                 name: str = "ayurvidya"):
        import chromadb
        self.ids = ids
        self.index_of = {cid: i for i, cid in enumerate(ids)}
        self.client = chromadb.PersistentClient(path=str(path))
        col = self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine", "sha": sha or "-"})
        if matrix is not None and ((col.metadata or {}).get("sha") != (sha or "-") or col.count() != len(ids)):
            self.client.delete_collection(name)
            col = self.client.create_collection(name, metadata={"hnsw:space": "cosine", "sha": sha or "-"})
            metas = [{"source_type": c.source_type, "work": c.work or "-", "shlokas": ",".join(c.shloka_ids) or "-",
                      "step": int(c.meta.get("step", 0) or 0)} for c in chunks] if chunks else None
            for s in range(0, len(ids), 500):
                col.upsert(ids=ids[s:s + 500], embeddings=np.asarray(matrix[s:s + 500]).tolist(),
                           metadatas=metas[s:s + 500] if metas else None)
        self.col = col

    def search(self, q: np.ndarray, k: int, allowed: np.ndarray | None = None,
               source_types=None) -> list[tuple[int, float]]:
        where = {"source_type": {"$in": list(source_types)}} if source_types else None
        n = min(self.col.count(), max(k * 3, 30))
        if n == 0:
            return []
        res = self.col.query(query_embeddings=[np.asarray(q).tolist()], n_results=n, where=where)
        out = []
        for cid, dist in zip(res["ids"][0], res["distances"][0]):
            i = self.index_of.get(cid)
            if i is not None and (allowed is None or allowed[i]):
                out.append((i, 1.0 - float(dist)))
            if len(out) == k:
                break
        return out

    @property
    def dim(self) -> int:
        return int(len(self.col.peek(1)["embeddings"][0])) if self.col.count() else 0


# ------------------------------------------------------------------ sparse
def bm25_tokens(text: str) -> list[str]:
    """Folded words + character 4-grams of each word.

    Sanskrit sandhi glues words together (वायुरामाशये = वायुः + आमाशये), so whole-word matching alone
    misses. 4-grams let 'vayu' inside 'vayuramasaye' still match a query for 'vayu'."""
    toks: list[str] = []
    for w in fold_for_search(text).split():
        toks.append(w)
        if len(w) > 4:
            toks.extend(f"#{w[i:i + 4]}" for i in range(len(w) - 3))
    return toks


class BM25:
    """Okapi BM25 (Robertson & Zaragoza). k1 and b are the standard defaults."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg = float(self.len.mean()) if len(docs) else 0.0
        df: Counter = Counter()
        for d in self.tf:
            df.update(d.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.postings: dict[str, list[int]] = {}
        for i, d in enumerate(self.tf):
            for t in d:
                self.postings.setdefault(t, []).append(i)

    def search(self, query_tokens: list[str], k: int, allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for t in set(query_tokens):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.postings[t]:
                if allowed is not None and not allowed[i]:
                    continue
                f = self.tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / (self.avg or 1))
                scores[i] = scores.get(i, 0.0) + idf * f * (self.k1 + 1) / denom
        return sorted(scores.items(), key=lambda x: -x[1])[:k]


# ------------------------------------------------------------------ knowledge base
def _save_matrix(matrix, kb_dir: Path) -> None:
    for old in ("embeddings.npy", "embeddings.npz"):
        (kb_dir / old).unlink(missing_ok=True)
    if sparse.issparse(matrix):
        sparse.save_npz(kb_dir / "embeddings.npz", matrix.tocsr())
    else:
        np.save(kb_dir / "embeddings.npy", np.asarray(matrix, dtype=np.float32))


def _load_matrix(kb_dir: Path):
    if (kb_dir / "embeddings.npz").exists():
        return sparse.load_npz(kb_dir / "embeddings.npz").tocsr()
    return np.load(kb_dir / "embeddings.npy")


class KnowledgeBase:
    """Chunks + dense store + BM25 + lookup tables, built once and loaded by the app."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder, dense):
        self.chunks = chunks
        self.embedder = embedder
        self.dense = dense
        self.bm25 = BM25([bm25_tokens(c.text) for c in chunks])
        self.by_id = {c.id: i for i, c in enumerate(chunks)}
        self.by_shloka: dict[str, list[int]] = {}
        for i, c in enumerate(chunks):
            for sid in c.shloka_ids:
                self.by_shloka.setdefault(sid, []).append(i)
        self.types = np.array([c.source_type for c in chunks])

    # build / persist -------------------------------------------------
    @classmethod
    def build(cls, chunks: list[Chunk], embedder_spec: str, kb_dir: Path, vector_store: str = "numpy"):
        kb_dir = Path(kb_dir)
        kb_dir.mkdir(parents=True, exist_ok=True)
        emb = make_embedder(embedder_spec)
        if embedder_spec != "tfidf":      # dense: embed each chunk once ever, vectors kept in data/vectors/
            slug = embedder_spec.replace(":", "_").replace("/", "_")
            emb = CachedEmbedder(emb, kb_dir.parent / "vectors" / f"{slug}.npz")
        texts = [c.text for c in chunks]
        emb.fit(texts)
        matrix = emb.embed_documents(texts)
        save_chunks(chunks, kb_dir / "chunks.jsonl")
        _save_matrix(matrix, kb_dir)
        if embedder_spec == "tfidf":
            emb.save(kb_dir / "tfidf.pkl")
        manifest = {"embedder": embedder_spec, "vector_store": vector_store, "n_chunks": len(chunks),
                    "dim": int(matrix.shape[1]),
                    "chunks_sha1": hashlib.sha1("".join(texts).encode("utf-8")).hexdigest()}
        (kb_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return cls(chunks, emb, cls._dense(vector_store, [c.id for c in chunks], matrix, kb_dir, chunks,
                                           manifest["chunks_sha1"]))

    @classmethod
    def load(cls, kb_dir: Path):
        kb_dir = Path(kb_dir)
        man_path = kb_dir / "manifest.json"
        if not man_path.exists():
            raise FileNotFoundError(f"No knowledge base at {kb_dir}. Run: python scripts/build_kb.py")
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
        chunks = load_chunks(kb_dir / "chunks.jsonl")
        matrix = _load_matrix(kb_dir)
        emb = load_embedder(manifest["embedder"], kb_dir)
        return cls(chunks, emb, cls._dense(manifest["vector_store"], [c.id for c in chunks], matrix, kb_dir, chunks,
                                           manifest.get("chunks_sha1", "")))

    @staticmethod
    def _dense(kind: str, ids, matrix, kb_dir, chunks=None, sha: str = ""):
        if kind == "numpy":
            return NumpyVectorStore(ids, matrix)
        if kind == "chroma":
            if sparse.issparse(matrix):
                raise ValueError("Chroma needs dense embeddings: use gemini:<model> or st:<model> with vector_store: chroma")
            return ChromaVectorStore(ids, matrix, Path(kb_dir) / "chroma", chunks, sha)
        raise ValueError(f"Unknown vector_store '{kind}'")

    # helpers ---------------------------------------------------------
    def mask(self, source_types=None, exclude_ids=None, exclude_shlokas=None) -> np.ndarray | None:
        if not source_types and not exclude_ids and not exclude_shlokas:
            return None
        m = np.ones(len(self.chunks), dtype=bool)
        if source_types:
            m &= np.isin(self.types, list(source_types))
        for cid in exclude_ids or ():
            if cid in self.by_id:
                m[self.by_id[cid]] = False
        for sid in exclude_shlokas or ():
            for i in self.by_shloka.get(sid, []):
                m[i] = False
        return m

    def chunks_for_shloka(self, shloka_id: str, source_types=None) -> list[Chunk]:
        return [self.chunks[i] for i in self.by_shloka.get(shloka_id, [])
                if not source_types or self.chunks[i].source_type in source_types]