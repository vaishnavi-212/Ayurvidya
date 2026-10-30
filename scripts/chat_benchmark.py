"""Chatbot speed benchmark: asks the questions in data/eval/chat_questions.csv and prints median / p90 of
retrieval time, time to first word and total time. Every answer is also appended to outputs/chat_timing.csv.

    python scripts/chat_benchmark.py              # first 10 questions (verse 13 and 22)
    python scripts/chat_benchmark.py --all        # all 20 (needs the chat AI; costs about 20 calls)
    python scripts/chat_benchmark.py --mock       # no AI at all: measures only retrieval + prompt building
Run it once before and once after a change; the two tables go in the paper.
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

from ayurvidya.llm import LLMError, MockLLM  # noqa: E402
from ayurvidya.service import build_app  # noqa: E402


def p90(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.9 * (len(xs) - 1))))] if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    app = build_app(chat_llm=MockLLM() if args.mock else None)
    with open(ROOT / "data" / "eval" / "chat_questions.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = rows if args.all else rows[:10]
    print(f"chat model: {app.chatbot.llm.name}   questions: {len(rows)}\n")
    stats = {"retrieval_ms": [], "first_token_ms": [], "total_ms": [], "prompt_chars": []}
    for r in rows:
        try:
            a = app.chatbot.ask(r["question"], r["shloka_id"] or None)
        except LLMError as e:
            print(f"  {r['qid']}  AI error: {e}")
            continue
        for k in stats:
            if a.timings.get(k) is not None:
                stats[k].append(a.timings[k])
        print(f"  {r['qid']}  {a.timings.get('total_ms', 0):7.0f} ms  gated={a.timings.get('gate', False)!s:5}  "
              f"{a.answer[:70]!r}")
    print("\n               median      p90")
    for k, v in stats.items():
        if v:
            print(f"  {k:15s} {statistics.median(v):8.0f} {p90(v):8.0f}")
    print(f"\nlog: {app.settings.timing_log}")


if __name__ == "__main__":
    main()