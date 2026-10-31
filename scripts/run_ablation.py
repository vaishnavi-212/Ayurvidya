"""Ablation study for the paper: add one part of the pipeline at a time and measure retrieval quality
on single-verse and multi-verse questions, plus chat latency from the timing log.

    python scripts/run_ablation.py                  # retrieval only: no AI calls, runs in seconds
    python scripts/run_ablation.py --chat           # also the chatbot on 20 questions per config (AI calls; count printed)

Configurations (each adds one thing to the one before):
    1 BM25 only · 2 dense only · 3 hybrid (RRF) · 4 + query expansion · 5 + knowledge graph · 6 + dedupe/MMR
    (+ rerank, if a reranker is set in settings)
Writes outputs/ablation.csv, outputs/latency.csv, outputs/paper_tables.md and outputs/figures/*.png.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ayurvidya.evaluation.retrieval_metrics import load_qrels, mrr, ndcg_at_k, recall_at_k  # noqa: E402
from ayurvidya.llm import LLMError  # noqa: E402
from ayurvidya.service import build_app  # noqa: E402
from ayurvidya.verify import citation_precision  # noqa: E402

OUT = ROOT / "outputs"


def configs(app):
    g, exp = app.graph, app.retriever.expander
    rows = [("1 BM25 only", dict(mode="sparse", expand=False, use_graph=False, diverse=False)),
            ("2 dense only", dict(mode="dense", expand=False, use_graph=False, diverse=False)),
            ("3 hybrid RRF", dict(mode="hybrid", expand=False, use_graph=False, diverse=False)),
            ("4 + query expansion", dict(mode="hybrid", expand=True, use_graph=False, diverse=False))]
    if g is not None:
        rows.append(("5 + knowledge graph", dict(mode="hybrid", expand=True, use_graph=True, diverse=False)))
    rows.append(("6 + dedupe/MMR", dict(mode="hybrid", expand=True, use_graph=g is not None, diverse=True)))
    if app.retriever.reranker is not None:
        rows.append(("7 + rerank", dict(mode="hybrid", expand=True, use_graph=g is not None, diverse=True, rerank=True)))
    return rows if exp is not None else [r for r in rows if not r[1]["expand"]]


def score(retriever, qrels, opts, k=10):
    out = []
    for q in qrels:
        ranked = [h.chunk.id for h in retriever.search(q["query"], k, source_types=q.get("types") or None,
                                                       rerank=opts.get("rerank", False),
                                                       **{x: v for x, v in opts.items() if x != "rerank"})]
        rel = set(q["relevant"])
        out.append({"recall@5": recall_at_k(ranked, rel, 5), "recall@10": recall_at_k(ranked, rel, 10),
                    "mrr": mrr(ranked, rel), "ndcg@10": ndcg_at_k(ranked, rel, 10)})
    return {m: round(statistics.mean(r[m] for r in out), 4) for m in out[0]} if out else {}


def figures(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(pip install matplotlib for the figures)")
        return
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    names = list(dict.fromkeys(r["config"] for r in rows))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for j, (qset, color) in enumerate((("single-verse", "#8c2a1c"), ("multi-verse", "#b9801c"))):
        vals = [next((r["recall@5"] for r in rows if r["config"] == n and r["questions"] == qset), 0) for n in names]
        ax.bar([i + j * 0.38 for i in range(len(names))], vals, width=0.38, label=qset, color=color)
    ax.set_xticks([i + 0.19 for i in range(len(names))], [n.split(" ", 1)[1] for n in names], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Recall@5")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "recall_by_config.png", dpi=200)
    print("figure: outputs/figures/recall_by_config.png")


def latency_table() -> list[str]:
    log = OUT / "chat_timing.csv"
    if not log.exists():
        return ["(no outputs/chat_timing.csv yet: run scripts/chat_benchmark.py or use the app)"]
    with open(log, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("total_ms")]
    by = {}
    for r in rows:
        by.setdefault(r["model"], []).append(r)
    md = ["| chat model | answers | cached | median first word (ms) | median total (ms) | median prompt (chars) |",
          "|---|---|---|---|---|---|"]
    for m, rs in by.items():
        num = lambda k: [float(r[k]) for r in rs if r.get(k) not in ("", None)]
        md.append(f"| {m} | {len(rs)} | {sum(r['cached'] == '1' for r in rs)} | "
                  f"{statistics.median(num('first_token_ms') or [0]):.0f} | {statistics.median(num('total_ms') or [0]):.0f} | "
                  f"{statistics.median(num('prompt_chars') or [0]):.0f} |")
    with open(OUT / "latency.csv", "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(md))
    return md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    app = build_app()
    sets = {"single-verse": load_qrels(ROOT / "data" / "eval" / "retrieval_qrels.jsonl"),
            "multi-verse": load_qrels(ROOT / "data" / "eval" / "multiverse_qrels.jsonl")}
    rows = []
    for name, opts in configs(app):
        for qset, qrels in sets.items():
            m = score(app.retriever, qrels, opts)
            rows.append({"config": name, "questions": qset, "n": len(qrels), **m})
            print(f"{name:22s} {qset:13s} R@5={m['recall@5']:.3f}  MRR={m['mrr']:.3f}  nDCG@10={m['ndcg@10']:.3f}")
    OUT.mkdir(exist_ok=True)
    with open(OUT / "ablation.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    md = [f"# Ayurvidya results", "", f"Embedder: {app.settings.embedder} · store: {app.settings.vector_store} · "
          f"graph facts: {len(app.graph.facts) if app.graph else 0}", "", "## Retrieval ablation", "",
          "| configuration | questions | n | Recall@5 | Recall@10 | MRR | nDCG@10 |", "|---|---|---|---|---|---|---|"]
    md += [f"| {r['config']} | {r['questions']} | {r['n']} | {r['recall@5']} | {r['recall@10']} | {r['mrr']} | {r['ndcg@10']} |"
           for r in rows]

    if args.chat:
        with open(ROOT / "data" / "eval" / "chat_questions.csv", encoding="utf-8-sig", newline="") as f:
            qs = list(csv.DictReader(f))
        chat_cfgs = [("hybrid", {"use_kg": False, "relevance_gate": False}), ("+ graph", {"use_kg": True, "relevance_gate": False}),
                     ("+ graph + gate", {"use_kg": True, "relevance_gate": True})]
        print(f"\nchat: {len(chat_cfgs)} configs x {len(qs)} questions = up to {len(chat_cfgs) * len(qs)} AI calls "
              f"on {app.chatbot.llm.name}")
        if args.yes or input("Go? [y/N] ").strip().lower() == "y":
            md += ["", "## Chatbot", "", "| configuration | grounded | citation precision | off-topic refused | median total (ms) |",
                   "|---|---|---|---|---|"]
            graph = app.graph
            for name, c in chat_cfgs:
                app.chatbot.graph = graph if c["use_kg"] else None
                app.retriever.graph = graph if c["use_kg"] else None
                app.chatbot.cfg.relevance_gate = c["relevance_gate"]
                app.chatbot._cache.clear()
                res = []
                for q in qs:
                    try:
                        res.append((q, app.chatbot.ask(q["question"], q["shloka_id"] or None)))
                    except LLMError as e:
                        print(f"  AI error: {str(e)[:80]}")
                if not res:
                    continue
                precs = [p for p in (citation_precision(a.checks) for _, a in res) if p is not None]
                off = [a for q, a in res if q["kind"] == "offtopic"]
                md.append(f"| {name} | {sum(a.grounded for _, a in res) / len(res):.2f} | "
                          f"{statistics.mean(precs) if precs else float('nan'):.2f} | "
                          f"{sum('could not find' in a.answer for a in off)}/{len(off)} | "
                          f"{statistics.median(a.timings.get('total_ms', 0) for _, a in res):.0f} |")
                print(md[-1])
    md += ["", "## Chat latency (from outputs/chat_timing.csv)", ""] + latency_table()
    md += ["", "Add the per-step chrF table from scripts/quality_report.py, CVI (scripts/compute_cvi.py) and SUS here."]
    (OUT / "paper_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    figures(rows)
    print("\nsaved: outputs/ablation.csv, outputs/paper_tables.md")


if __name__ == "__main__":
    main()