"""Compare chat models on the same questions: speed, groundedness, citation precision (paper table).

    python scripts/compare_chat_models.py openai:llama-3.1-8b-instant openai:llama-3.3-70b-versatile
    python scripts/compare_chat_models.py ollama:ayurvidya-llama ollama:llama3.1:8b
    python scripts/compare_chat_models.py gemini:gemini-3.5-flash-lite openai:llama-3.1-8b-instant --n 10

Prefixes: openai:<model> = Groq (GROQ_API_KEY) · ollama:<model> = local Ollama · gemini:<model> = Gemini (OPENAI_API_KEY).
Questions: data/eval/chat_questions.csv (20). Each model x question = 1 AI call; the count is printed first.
Writes outputs/chat_model_comparison.csv (every answer) and outputs/chat_model_comparison.md (summary).
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

from ayurvidya.chatbot import NOT_FOUND  # noqa: E402
from ayurvidya.llm import LLMError, make_llm  # noqa: E402
from ayurvidya.service import build_app  # noqa: E402
from ayurvidya.verify import citation_precision  # noqa: E402

SERVERS = {"openai": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
           "ollama": ("http://localhost:11434/v1", "NO_KEY"),
           "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "OPENAI_API_KEY")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--n", type=int, default=20, help="how many questions")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    with open(ROOT / "data" / "eval" / "chat_questions.csv", encoding="utf-8-sig", newline="") as f:
        qs = list(csv.DictReader(f))[: args.n]
    print(f"{len(args.models)} models x {len(qs)} questions = up to {len(args.models) * len(qs)} AI calls")
    if not args.yes and input("Go? [y/N] ").strip().lower() != "y":
        return
    app = build_app()
    rows, summary = [], []
    for spec in args.models:
        kind, _, model = spec.partition(":")
        url, key = SERVERS[kind]
        app.chatbot.llm = make_llm(f"openai:{model}", url, app.settings.chat_temperature, app.settings.chat_max_tokens,
                                   key, retries=1)
        app.chatbot._cache.clear()
        print(f"\n== {spec}")
        ok = []
        for q in qs:
            try:
                a = app.chatbot.ask(q["question"], q["shloka_id"] or None)
            except LLMError as e:
                print(f"  {q['qid']}: AI error {str(e)[:80]}")
                continue
            prec = citation_precision(a.checks)
            row = {"model": spec, "qid": q["qid"], "kind": q["kind"], "grounded": a.grounded,
                   "not_found": NOT_FOUND in a.answer, "citation_precision": prec, "n_citations": len(a.citations),
                   "first_token_ms": a.timings.get("first_token_ms"), "total_ms": a.timings.get("total_ms"),
                   "answer": a.answer}
            rows.append(row)
            ok.append(row)
            print(f"  {q['qid']}  {row['total_ms']:7.0f} ms  grounded={a.grounded!s:5}  {a.answer[:60]!r}")
        if ok:
            med = lambda k: statistics.median([r[k] for r in ok if r[k] is not None] or [0])
            precs = [r["citation_precision"] for r in ok if r["citation_precision"] is not None]
            offtopic = [r for r in ok if r["kind"] == "offtopic"]
            summary.append({"model": spec, "answered": len(ok), "grounded_rate": round(sum(r["grounded"] for r in ok) / len(ok), 3),
                            "citation_precision": round(statistics.mean(precs), 3) if precs else None,
                            "refused_offtopic": sum(r["not_found"] for r in offtopic), "n_offtopic": len(offtopic),
                            "median_first_token_ms": med("first_token_ms"), "median_total_ms": med("total_ms")})
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    if rows:
        with open(out / "chat_model_comparison.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    md = ["| model | answered | grounded | citation precision | off-topic refused | first word (ms) | total (ms) |",
          "|---|---|---|---|---|---|---|"]
    md += [f"| {s['model']} | {s['answered']} | {s['grounded_rate']} | {s['citation_precision']} | "
           f"{s['refused_offtopic']}/{s['n_offtopic']} | {s['median_first_token_ms']:.0f} | {s['median_total_ms']:.0f} |" for s in summary]
    (out / "chat_model_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md) + "\n\nsaved: outputs/chat_model_comparison.csv and .md")


if __name__ == "__main__":
    main()