"""Compare retrievers on the test questions: which embedder + store + query expansion finds the right verse?

    python scripts/compare_embedders.py                                   # tfidf + gemini (needs GEMINI_API_KEY)
    python scripts/compare_embedders.py tfidf gemini:gemini-embedding-001 st:BAAI/bge-m3
    python scripts/compare_embedders.py gemini:gemini-embedding-001 --chroma   # same, stored in ChromaDB

Each embedder gets its own KB in outputs/kb_compare/ (your data/kb is not touched). Dense vectors are
cached in outputs/kb_compare/vectors/, so the second run costs no API calls. Every embedder is run in BM25, dense and
hybrid mode, with and without query expansion.
Results: outputs/retrieval_comparison.csv and outputs/retrieval_comparison.md (the paper table).
"""
from __future__ import annotations

import csv
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from build_kb import topics_from_csv, topics_from_gold  # noqa: E402

from ayurvidya.config import load_settings  # noqa: E402
from ayurvidya.evaluation.retrieval_metrics import evaluate, load_qrels  # noqa: E402
from ayurvidya.index import KnowledgeBase  # noqa: E402
from ayurvidya.ingest import build_chunks, load_gold_xlsx, load_mula  # noqa: E402
from ayurvidya.query import QueryExpander  # noqa: E402
from ayurvidya.retrieve import HybridRetriever, RetrievalConfig  # noqa: E402

DEFAULT = ["tfidf", "gemini:gemini-embedding-001"]


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    store = "chroma" if "--chroma" in sys.argv else "numpy"
    specs = args or DEFAULT
    s = load_settings()
    gold_files = sorted((s.data_dir / "gold").glob("*.xlsx"))
    topics = {**topics_from_csv(s.data_dir / "topics_su_ni_1.csv"), **topics_from_gold(gold_files)}
    shlokas, _ = load_mula(s.data_dir / "raw" / "niimh_su_ni_1_mula.json", topics=topics)
    gold = {}
    for g in gold_files:
        gold.update(load_gold_xlsx(g))
    sources = sorted(p for p in (s.data_dir / "sources").glob("*") if p.suffix in (".txt", ".jsonl"))
    chunks = build_chunks(shlokas, gold, sources)
    qrels = load_qrels(s.data_dir / "eval" / "retrieval_qrels.jsonl")
    expander = QueryExpander(s.data_dir / "synonyms.csv")
    print(f"{len(chunks)} chunks, {len(qrels)} test questions, store = {store}\n")

    rows = []
    for spec in specs:
        print(f"== {spec}")
        kb_dir = ROOT / "outputs" / "kb_compare" / (spec.replace(":", "_").replace("/", "_") + "_" + store)
        shutil.rmtree(kb_dir, ignore_errors=True)
        t0 = time.time()
        try:
            kb = KnowledgeBase.build(chunks, spec, kb_dir, store if spec != "tfidf" else "numpy")
        except Exception as e:   # no key / model not installed / chromadb missing
            print(f"   FAILED: {e}\n")
            rows.append({"embedder": spec, "store": store, "mode": f"failed: {str(e)[:60]}"})
            continue
        secs = round(time.time() - t0, 1)
        for exp in (False, True):
            r = HybridRetriever(kb, RetrievalConfig(s.dense_k, s.sparse_k, s.rrf_k, s.final_k),
                                expander=expander if exp else None)
            for mode, m in evaluate(r, qrels).items():
                rows.append({"embedder": spec, "store": store if spec != "tfidf" else "numpy", "expansion": exp,
                             "mode": mode, **m, "build_s": secs})
                print(f"   expansion={exp!s:5} {mode:6s}  recall@5={m['recall@5']}  MRR={m['mrr']}")
        print()

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    keys = ["embedder", "store", "expansion", "mode", "recall@1", "recall@3", "recall@5", "recall@10", "mrr",
            "ndcg@10", "build_s"]
    with open(out / "retrieval_comparison.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    md = ["| embedder | store | expansion | mode | R@1 | R@5 | R@10 | MRR | nDCG@10 |", "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['embedder']} | {r['store']} | {r.get('expansion', '')} | {r['mode']} | {r.get('recall@1', '')} | "
           f"{r.get('recall@5', '')} | {r.get('recall@10', '')} | {r.get('mrr', '')} | {r.get('ndcg@10', '')} |" for r in rows]
    (out / "retrieval_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print("\nsaved: outputs/retrieval_comparison.csv and .md")


if __name__ == "__main__":
    main()