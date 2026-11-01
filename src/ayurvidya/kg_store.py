"""Knowledge-graph facts in a database, so a scholar can upload a CSV / Excel sheet in the app and the
graph changes at once - no Git, no rebuild.

Same database as student notes: Postgres when DATABASE_URL is set (Supabase), else a local SQLite file.
Every upload is a new version; nothing is ever deleted, only marked inactive, so each fact keeps its
history (who added it, when, what it replaced). For the paper, `export_csv` writes a frozen snapshot to
data/kg/facts.csv: commit that file with the results so the experiment can be repeated exactly.

Each row must quote its verse exactly (the quote is checked against the mula text), so a fact can never
point at words the verse does not contain.
"""
from __future__ import annotations

import csv
import io
import time
from pathlib import Path

from .kg import Fact, KnowledgeGraph
from .notes import NotesDB

FIELDS = ["subject", "relation", "object", "object_deva", "verse", "quote", "status"]
SUBJECTS = {"Prana", "Udana", "Samana", "Vyana", "Apana", "Vayu"}
RELATIONS = {"located_in", "performs", "causes", "covered_by_pitta", "covered_by_kapha"}
STATUSES = {"seed", "draft", "approved", "rejected"}

DDL = """CREATE TABLE IF NOT EXISTS kg_facts (
  id INTEGER PRIMARY KEY AUTOINCREMENT, subject TEXT NOT NULL, relation TEXT NOT NULL, object TEXT NOT NULL,
  object_deva TEXT NOT NULL DEFAULT '', verse TEXT NOT NULL, quote TEXT NOT NULL, status TEXT NOT NULL,
  added_by TEXT NOT NULL, version INTEGER NOT NULL, active INTEGER NOT NULL DEFAULT 1, replaced_in INTEGER,
  created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS ix_kg_active ON kg_facts(active)"""


class KGStore:
    def __init__(self, db: NotesDB, verse_texts: dict[str, str]):
        self.db = db
        self.verse_texts = verse_texts              # "13" -> mula text (also "31-32" and each part)
        ddl = DDL if not db.pg else DDL.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
        for stmt in ddl.split(";"):
            if stmt.strip():
                db.q(stmt)

    @classmethod
    def for_app(cls, db: NotesDB, store) -> "KGStore":
        texts = {}
        for s in store.shlokas("su.ni.1"):
            texts[s["number"]] = s["text"]
            for part in s["number"].split("-"):
                texts.setdefault(part, s["text"])
        return cls(db, texts)

    # ---------------------------------------------------------- reading
    def version(self) -> int:
        r = self.db.q("SELECT MAX(version) AS v FROM kg_facts")
        return int(r[0]["v"] or 0) if r else 0

    def rows(self, include_inactive: bool = False) -> list[dict]:
        where = "" if include_inactive else "WHERE active=1"
        return self.db.q(f"SELECT id, subject, relation, object, object_deva, verse, quote, status, added_by, version, "
                         f"active FROM kg_facts {where} ORDER BY verse, subject, relation, id")

    def facts(self, statuses=("seed", "approved")) -> list[Fact]:
        return [Fact(r["subject"], r["relation"], r["object"], r["object_deva"], f"su.ni.1.{r['verse']}", r["quote"],
                     r["status"]) for r in self.rows() if r["status"] in statuses]

    def graph(self, synonyms, kb) -> KnowledgeGraph:
        from .query import QueryExpander
        return KnowledgeGraph(self.facts(), QueryExpander(synonyms), kb)

    # ---------------------------------------------------------- writing
    def parse(self, data: bytes, filename: str) -> tuple[list[dict], list[dict]]:
        """CSV or XLSX with the facts.csv columns. Returns (good rows, [{row, problem}])."""
        try:
            if filename.lower().endswith(".xlsx"):
                from openpyxl import load_workbook
                wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
                rows = [["" if c is None else str(c) for c in r] for r in wb.worksheets[0].iter_rows(values_only=True)]
                wb.close()
            else:
                rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
        except Exception:
            return [], [{"row": 0, "problem": "Could not read the file. Use .csv (UTF-8) or .xlsx."}]
        if not rows:
            return [], [{"row": 0, "problem": "The file is empty."}]
        head = [h.strip().lower() for h in rows[0]]
        missing = [f for f in FIELDS[:6] if f not in head]
        if missing:
            return [], [{"row": 1, "problem": f"missing columns: {', '.join(missing)}"}]
        good, bad = [], []
        for n, r in enumerate(rows[1:], 2):
            d = {h: (r[i].strip() if i < len(r) and r[i] is not None else "") for i, h in enumerate(head)}
            if not any(d.values()):
                continue
            d["verse"] = d["verse"].removesuffix(".0")
            d["status"] = (d.get("status") or "approved").lower()
            if d.get("approve", "").lower() in ("no", "n"):
                d["status"] = "rejected"
            text = self.verse_texts.get(d["verse"])
            if d["subject"] not in SUBJECTS:
                bad.append({"row": n, "problem": f"subject '{d['subject']}' must be one of {', '.join(sorted(SUBJECTS))}"})
            elif d["relation"] not in RELATIONS:
                bad.append({"row": n, "problem": f"relation '{d['relation']}' must be one of {', '.join(sorted(RELATIONS))}"})
            elif text is None:
                bad.append({"row": n, "problem": f"unknown verse '{d['verse']}'"})
            elif not d["object"] or not d["quote"]:
                bad.append({"row": n, "problem": "object and quote are required"})
            elif d["quote"] not in text:
                bad.append({"row": n, "problem": "the quote is not found word-for-word in that verse"})
            elif d["status"] not in STATUSES:
                bad.append({"row": n, "problem": f"status must be one of {', '.join(sorted(STATUSES))}"})
            else:
                good.append({k: d.get(k, "") for k in FIELDS})
        return good, bad

    def add(self, rows: list[dict], who: str) -> int:
        """New version: each row replaces the active fact with the same (subject, relation, verse, object)."""
        v = self.version() + 1
        now = time.time()
        for g in rows:
            self.db.q("UPDATE kg_facts SET active=0, replaced_in=? WHERE active=1 AND subject=? AND relation=? "
                      "AND verse=? AND object=?", (v, g["subject"], g["relation"], g["verse"], g["object"]))
            self.db.q("INSERT INTO kg_facts (subject, relation, object, object_deva, verse, quote, status, added_by, "
                      "version, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (g["subject"], g["relation"], g["object"], g["object_deva"], g["verse"], g["quote"], g["status"],
                       who, v, now))
        return v

    def upload(self, data: bytes, filename: str, who: str) -> dict:
        good, bad = self.parse(data, filename)
        v = self.add(good, who) if good else self.version()
        return {"ok_rows": len(good), "errors": bad, "version": v}

    def undo(self) -> int:
        v = self.version()
        if v > 1:                                    # version 1 = the seed import, never undone
            self.db.q("DELETE FROM kg_facts WHERE version=?", (v,))
            self.db.q("UPDATE kg_facts SET active=1, replaced_in=NULL WHERE replaced_in=?", (v,))
        return v

    def seed_from_csv(self, path: Path) -> int:
        """First run only: copy data/kg/facts.csv into the empty table as version 1."""
        if self.version() or not Path(path).exists():
            return 0
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = [{k: (r.get(k) or "").strip() for k in FIELDS} for r in csv.DictReader(f)]
        self.add(rows, "seed-file")
        return len(rows)

    def export_csv(self, path: Path | None = None) -> str:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for r in self.rows():
            if r["status"] != "rejected":
                w.writerow({k: r[k] for k in FIELDS})
        if path:
            Path(path).write_text(buf.getvalue(), encoding="utf-8")
        return buf.getvalue()