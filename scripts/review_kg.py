"""Scholar check of knowledge-graph facts (same idea as the step review sheet).

    python scripts/review_kg.py export        # seed + draft facts -> outputs/kg_review.xlsx
    (scholar: 'approve' = yes / no; may correct subject, relation, object, object_deva)
    python scripts/review_kg.py import outputs/kg_review.xlsx
        -> data/kg/facts.csv: approved rows get status 'approved'; rows marked no are removed; others kept as they were
Commit data/kg/facts.csv (and facts_draft.csv). The app loads seed + approved facts.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from openpyxl import Workbook, load_workbook  # noqa: E402
from openpyxl.styles import PatternFill  # noqa: E402

KG = ROOT / "data" / "kg"
FIELDS = ["subject", "relation", "object", "object_deva", "verse", "quote", "status"]


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def key(r: dict) -> tuple:
    return (r["subject"], r["relation"], r["verse"], r["quote"])


def export() -> None:
    rows = read(KG / "facts.csv")
    have = {key(r) for r in rows}
    rows += [r for r in read(KG / "facts_draft.csv") if key(r) not in have]
    wb = Workbook()
    ws = wb.active
    ws.append(FIELDS + ["approve", "note"])
    yellow = PatternFill("solid", fgColor="FFF2CC")
    for r in rows:
        ws.append([r[c] for c in FIELDS] + ["yes" if r["status"] == "approved" else "", ""])
        ws.cell(ws.max_row, len(FIELDS) + 1).fill = yellow
    for col, w in zip("ABCDEFGHI", (10, 17, 40, 24, 7, 40, 10, 9, 30)):
        ws.column_dimensions[col].width = w
    out = ROOT / "outputs" / "kg_review.xlsx"
    out.parent.mkdir(exist_ok=True)
    wb.save(out)
    print(f"{len(rows)} facts -> {out.relative_to(ROOT)}  (fill 'approve' with yes / no)")


def import_(path: str) -> None:
    ws = load_workbook(path, read_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    h = [str(c or "") for c in rows[0]]
    out, n_yes, n_no = [], 0, 0
    for r in rows[1:]:
        d = {k: ("" if v is None else str(v).strip()) for k, v in zip(h, r)}
        verdict = d.get("approve", "").lower()
        if verdict in ("no", "n"):
            n_no += 1
            continue
        if verdict in ("yes", "y"):
            d["status"] = "approved"
            n_yes += 1
        if d["status"] in ("seed", "approved"):          # drafts nobody approved stay in facts_draft.csv only
            out.append({k: d.get(k, "") for k in FIELDS})
    with open(KG / "facts.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    print(f"approved {n_yes}, removed {n_no}; data/kg/facts.csv now has {len(out)} facts")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "export":
        export()
    elif len(sys.argv) == 3 and sys.argv[1] == "import":
        import_(sys.argv[2])
    else:
        sys.exit(__doc__)