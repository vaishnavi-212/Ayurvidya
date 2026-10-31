"""Draft knowledge-graph facts with the STEP model (Gemini), one verse per call, for a scholar to check.

    python scripts/extract_kg.py --verses 22-39          # 18 AI calls; prints the count first
    python scripts/extract_kg.py --verses 40-91 --pause 5
Writes data/kg/facts_draft.csv (status = draft). Resumable: verses already in the draft are skipped.
Every fact must quote its verse exactly; facts whose quote is not found are dropped and counted.
Then: python scripts/review_kg.py export  ->  scholar  ->  python scripts/review_kg.py import <file>
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ayurvidya.llm import LLMError  # noqa: E402
from ayurvidya.service import build_app  # noqa: E402

DRAFT = ROOT / "data" / "kg" / "facts_draft.csv"
FIELDS = ["subject", "relation", "object", "object_deva", "verse", "quote", "status"]
RELATIONS = {"located_in", "performs", "causes", "covered_by_pitta", "covered_by_kapha"}
SUBJECTS = {"Prana", "Udana", "Samana", "Vyana", "Apana", "Vayu"}


def parse(answer: str) -> list[dict]:
    m = re.search(r"\[.*\]", answer, re.S)
    try:
        return json.loads(m.group(0)) if m else []
    except ValueError:
        return []


def in_range(number: str, lo: int, hi: int) -> bool:
    return any(lo <= int(p) <= hi for p in number.split("-") if p.isdigit())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verses", default="12-39", help="range, e.g. 12-39")
    ap.add_argument("--pause", type=float, default=4.0, help="seconds between AI calls (free-tier limits)")
    ap.add_argument("--yes", action="store_true", help="do not ask before calling the AI")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.verses.split("-"))
    app = build_app()
    done = set()
    if DRAFT.exists():
        with open(DRAFT, encoding="utf-8", newline="") as f:
            done = {r["verse"] for r in csv.DictReader(f)}
    todo = [s for s in app.store.shlokas("su.ni.1") if in_range(s["number"], lo, hi) and s["number"] not in done]
    print(f"step model: {app.orchestrator.llm.name}   verses to do: {len(todo)}  (= {len(todo)} AI calls)")
    if not todo or (not args.yes and input("Go? [y/N] ").strip().lower() != "y"):
        return
    system = (ROOT / "prompts" / "kg_extract.md").read_text(encoding="utf-8")
    new = not DRAFT.exists()
    DRAFT.parent.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    with open(DRAFT, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for s in todo:
            steps = app.store.steps(s["id"])
            comm = [c.text for c in app.kb.chunks_for_shloka(s["id"], ["commentary"])][:2]
            user = (f"VERSE {s['number']}:\n{s['text']}\n\n" + (f"COMMENTARY:\n{chr(10).join(comm)}\n\n" if comm else "")
                    + "".join(f"STEP {n}:\n{steps[n]['output'][:1200]}\n\n" for n in (4, 5) if n in steps))
            try:
                facts = parse(app.orchestrator.llm.complete(system, user))
            except LLMError as e:
                print(f"  verse {s['number']}: AI error ({e}); stopping - run again later to continue")
                break
            for x in facts:
                ok = (x.get("subject") in SUBJECTS and x.get("relation") in RELATIONS and x.get("object")
                      and x.get("quote") and x["quote"] in s["text"])
                if ok:
                    w.writerow({"subject": x["subject"], "relation": x["relation"], "object": x["object"],
                                "object_deva": x.get("object_deva", ""), "verse": s["number"], "quote": x["quote"],
                                "status": "draft"})
                    kept += 1
                else:
                    dropped += 1
            f.flush()
            print(f"  verse {s['number']}: {len(facts)} facts")
            time.sleep(args.pause)
    print(f"\nkept {kept}, dropped {dropped} (bad relation or quote not in the verse) -> {DRAFT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()