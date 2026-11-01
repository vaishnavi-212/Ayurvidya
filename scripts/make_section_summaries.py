"""Section summaries (RAPTOR-style hierarchy): one short AI summary per section of the chapter, stored as
searchable chunks of type 'summary' that point to every verse of their section.

Why: a question like "What does the avarana section teach?" or "Summarise the five Vayus" is about a
whole section, not one verse. Verse-level chunks only answer it piece by piece; a section chunk answers
it at once and its verse list sends the reader to the right verses.

    python scripts/make_section_summaries.py          # 10 AI calls (step model), asks first
    python scripts/build_kb.py                         # makes them searchable
Writes data/sources/section_summaries.jsonl (commit it; ask a scholar to read it - it is AI text).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ayurvidya.llm import LLMError  # noqa: E402

OUT = ROOT / "data" / "sources" / "section_summaries.jsonl"
SECTIONS = [  # same sections as the app's table of contents
    (1, 6, "Opening and the greatness of Vayu"), (7, 10, "Nature, qualities and functions of Vayu"),
    (11, 21, "The five Vayus: seats, actions, disorders"), (22, 31, "Vayu lodged in organs and tissues"),
    (32, 39, "Vata joined with or covered by other doshas"), (40, 50, "Vatarakta: causes, signs and prognosis"),
    (51, 59, "Convulsions, spasms and rigidity"), (60, 67, "Hemiplegia, Apatantraka, stiff neck"),
    (68, 73, "Facial palsy"), (74, 91, "Disorders of limbs, senses and abdomen"),
]


def first_number(number: str) -> int:
    return int(number.split("-")[0])


def build(app, out: Path = OUT, pause: float = 4.0) -> int:
    system = (ROOT / "prompts" / "section_summary.md").read_text(encoding="utf-8")
    verses = app.store.shlokas("su.ni.1")
    lines = []
    for lo, hi, title in SECTIONS:
        part = [v for v in verses if lo <= first_number(v["number"]) <= hi]
        user = f"SECTION {lo}-{hi}: {title}\n\n"
        for v in part:
            steps = app.store.steps(v["id"])
            notes = " ".join(steps[n]["output"][:500] for n in (4, 5) if n in steps)
            user += f"VERSE {v['number']}:\n{v['text']}\nNOTES: {notes}\n\n"
        text = app.orchestrator.llm.complete(system, user).strip()
        lines.append({"id": f"summary.su.ni.1.{lo}-{hi}", "source_type": "summary",
                      "work": f"Section summary {lo}-{hi} (AI, to be checked)",
                      "shloka_ids": [v["id"] for v in part], "text": f"{title}. {text}", "section": f"{lo}-{hi}"})
        print(f"  section {lo}-{hi}: {len(text)} characters")
        time.sleep(pause)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(l, ensure_ascii=False) + "\n" for l in lines), encoding="utf-8")
    return len(lines)


def main() -> None:
    from ayurvidya.service import build_app
    app = build_app()
    print(f"step model: {app.orchestrator.llm.name}   10 sections = 10 AI calls")
    if "--yes" not in sys.argv and input("Go? [y/N] ").strip().lower() != "y":
        return
    try:
        n = build(app)
    except LLMError as e:
        sys.exit(f"AI error: {e}")
    print(f"\n{n} summaries -> {OUT.relative_to(ROOT)}\nNow run: python scripts/build_kb.py")


if __name__ == "__main__":
    main()