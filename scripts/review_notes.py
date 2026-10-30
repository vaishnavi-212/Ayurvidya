"""Teacher review of student notes -> the good ones join the shared sources.

    python scripts/review_notes.py export            # submitted notes -> outputs/notes_review.xlsx
    (teacher fills column 'approve' with yes/no, may correct 'text')
    python scripts/review_notes.py import outputs/notes_review.xlsx
        approved rows -> data/sources/student_approved.jsonl (commit it) and status 'approved' in the notes DB
    python scripts/build_kb.py                       # so the approved notes become searchable for everyone

Uses DATABASE_URL if set (the online notes), else data/study/notes.sqlite3.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from openpyxl import Workbook, load_workbook  # noqa: E402

from ayurvidya.config import load_dotenv  # noqa: E402
from ayurvidya.notes import NotesDB  # noqa: E402

COLS = ["id", "student_id", "verse_id", "kind", "word", "text", "source", "approve", "teacher_note"]
OUT = ROOT / "data" / "sources" / "student_approved.jsonl"


def db() -> NotesDB:
    load_dotenv()
    return NotesDB(os.environ.get("DATABASE_URL") or str(ROOT / "data" / "study" / "notes.sqlite3"))


def export() -> None:
    rows = db().q("SELECT id, student_id, verse_id, kind, word, text, source FROM user_notes "
                  "WHERE active=1 AND status='submitted' ORDER BY verse_id, id")
    wb = Workbook()
    ws = wb.active
    ws.append(COLS)
    for r in rows:
        ws.append([r[c] for c in COLS[:7]] + ["", ""])
    ws.column_dimensions["F"].width = 80
    path = ROOT / "outputs" / "notes_review.xlsx"
    path.parent.mkdir(exist_ok=True)
    wb.save(path)
    print(f"{len(rows)} submitted notes -> {path.relative_to(ROOT)}  (fill 'approve' with yes/no)")


def import_(path: str) -> None:
    d = db()
    ws = load_workbook(path, read_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    h = [str(c or "") for c in rows[0]]
    have = {json.loads(l)["id"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()} if OUT.exists() else set()
    n = 0
    with open(OUT, "a", encoding="utf-8", newline="\n") as f:
        for r in rows[1:]:
            rec = dict(zip(h, r))
            if str(rec.get("approve") or "").strip().lower() not in ("yes", "y", "1", "true"):
                continue
            cid = f"student_approved.{rec['id']}"
            if cid not in have:
                f.write(json.dumps({"id": cid, "source_type": "student_approved", "work": "Student notes (teacher-approved)",
                                    "shloka_ids": [rec["verse_id"]], "text": str(rec["text"]), "kind": rec["kind"],
                                    "word": rec.get("word") or ""}, ensure_ascii=False) + "\n")
                n += 1
            d.q("UPDATE user_notes SET status='approved' WHERE id=?", (int(rec["id"]),))
    print(f"{n} notes added to {OUT.relative_to(ROOT)}. Now run: python scripts/build_kb.py")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "export":
        export()
    elif len(sys.argv) == 3 and sys.argv[1] == "import":
        import_(sys.argv[2])
    else:
        sys.exit(__doc__)