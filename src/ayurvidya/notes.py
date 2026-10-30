"""Student notes: a student uploads an Excel sheet of their own translations / word meanings / notes,
and the tutor uses it when THAT student asks a doubt. Notes never touch the checked answers and never
reach another student. A teacher can promote good notes into the shared sources (scripts/review_notes.py).

Switched on by AYV_CLASS_CODE in the environment. Storage: SQLite file (local) or Postgres when
DATABASE_URL is set (needed on Render's free plan, which wipes local files on every restart).

Versioning: every upload is a new version. A new row replaces the student's active row with the same
(verse, kind, word); "undo" removes the last version and brings the replaced rows back.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import os
import re
import sqlite3
import time
from pathlib import Path

import numpy as np

from .index import BM25, bm25_tokens
from .retrieve import rrf

KINDS = ("translation", "word-meaning", "anvaya", "note", "question")
COLUMNS = ["verse", "kind", "step", "word", "text", "source"]
MAX_ROWS, MAX_TEXT = 2000, 2000

DDL_SQLITE = """CREATE TABLE IF NOT EXISTS user_notes (
  id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT NOT NULL, verse_id TEXT NOT NULL, kind TEXT NOT NULL,
  step INTEGER, word TEXT NOT NULL DEFAULT '', text TEXT NOT NULL, source TEXT NOT NULL DEFAULT '',
  version INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'personal', active INTEGER NOT NULL DEFAULT 1,
  replaced_in INTEGER, created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS ix_notes_student ON user_notes(student_id, active);"""
DDL_PG = DDL_SQLITE.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")


class NotesDB:
    def __init__(self, url_or_path: str):
        self.pg = url_or_path.startswith(("postgres://", "postgresql://"))
        if self.pg:                                  # pragma: no cover - needs a Postgres server
            import psycopg
            self.conn = psycopg.connect(url_or_path, autocommit=True)
            for stmt in DDL_PG.split(";"):
                if stmt.strip():
                    self.conn.execute(stmt)
        else:
            Path(url_or_path).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(url_or_path, check_same_thread=False, isolation_level=None)
            self.conn.executescript(DDL_SQLITE)

    def close(self) -> None:
        self.conn.close()

    def q(self, sql: str, args=()) -> list[dict]:
        if self.pg:                                  # pragma: no cover
            cur = self.conn.execute(sql.replace("?", "%s"), args)
            if cur.description is None:
                return []
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
        cur = self.conn.execute(sql, args)
        cols = [d[0] for d in cur.description or []]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:30] or "student"


class NotesService:
    def __init__(self, db: NotesDB, class_code: str, secret: str, kb, shlokas: list[dict], expander=None):
        self.db, self.class_code, self.secret = db, class_code, (secret or class_code).encode("utf-8")
        self.kb, self.expander = kb, expander
        self.verse_ids: dict[str, str] = {}             # "13" -> "su.ni.1.13", "31" -> "su.ni.1.31-32"
        for s in shlokas:
            self.verse_ids[s["number"]] = s["id"]
            for part in s["number"].split("-"):
                self.verse_ids.setdefault(part, s["id"])
            self.verse_ids[s["id"]] = s["id"]
        self._index: dict[str, tuple] = {}

    @classmethod
    def from_settings(cls, s, kb, store):
        code = os.environ.get("AYV_CLASS_CODE", "")
        if not code:
            return None
        url = os.environ.get("DATABASE_URL") or str(Path(s.data_dir) / "study" / "notes.sqlite3")
        from .query import QueryExpander
        return cls(NotesDB(url), code, os.environ.get("AYV_NOTES_SECRET", ""), kb, store.shlokas("su.ni.1"),
                   QueryExpander(Path(s.data_dir) / "synonyms.csv"))

    # ---------------------------------------------------------- identity (pilot level: class code + name)
    def _sign(self, sid: str) -> str:
        return hmac.new(self.secret, sid.encode("utf-8"), hashlib.sha256).hexdigest()[:20]

    def login(self, class_code: str, name: str) -> dict:
        if not name.strip() or not hmac.compare_digest(class_code.strip(), self.class_code):
            raise PermissionError("Wrong class code. Ask your teacher for it.")
        sid = f"{_slug(name)}-{hashlib.sha1((self.class_code + name.strip().lower()).encode()).hexdigest()[:6]}"
        return {"student_id": sid, "name": name.strip(), "token": f"{sid}.{self._sign(sid)}"}

    def verify(self, token: str) -> str | None:
        sid, _, sig = token.rpartition(".")
        return sid if sid and hmac.compare_digest(sig, self._sign(sid)) else None

    # ---------------------------------------------------------- template + parsing
    @staticmethod
    def template() -> bytes:
        from openpyxl import Workbook
        from openpyxl.styles import Font
        wb = Workbook()
        ws = wb.active
        ws.title = "notes"
        ws.append(COLUMNS)
        for c in ws[1]:
            c.font = Font(bold=True)
        ws.append([13, "translation", 4, "", "The Vayu that moves in the mouth is called Prana; it holds the body.", "class notes"])
        ws.append([13, "word-meaning", 2, "देहधृक्", "one who holds / sustains the body", "Dalhana"])
        ws.append([22, "note", "", "", "Compare with Charaka Chikitsa 28 on Vata in the stomach.", ""])
        for col, w in zip("ABCDEF", (8, 14, 6, 16, 70, 20)):
            ws.column_dimensions[col].width = w
        info = wb.create_sheet("how to fill")
        for line in ["One row per note. Only the sheet called 'notes' is read.",
                     "verse: the verse number, e.g. 13 or 31-32",
                     f"kind: one of {', '.join(KINDS)}",
                     "step: 1-7 if the note belongs to one study step (optional)",
                     "word: the Sanskrit word, for word-meaning rows (optional)",
                     "text: your note (up to 2000 characters)",
                     "source: where it comes from: teacher, book, page (optional)",
                     "Uploading again replaces your earlier row for the same verse, kind and word."]:
            info.append([line])
        info.column_dimensions["A"].width = 90
        buf = io.BytesIO()
        wb.save(buf)
        return buf.getvalue()

    def parse(self, data: bytes) -> tuple[list[dict], list[dict]]:
        from openpyxl import load_workbook
        try:
            wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        except Exception:
            return [], [{"row": 0, "problem": "This is not an .xlsx Excel file."}]
        try:
            ws = wb["notes"] if "notes" in wb.sheetnames else wb.worksheets[0]
            rows = list(ws.iter_rows(values_only=True, max_row=MAX_ROWS + 5))
        finally:
            wb.close()
        head = next((i for i, r in enumerate(rows[:10]) if r and {"verse", "text"} <= {str(c or "").strip().lower() for c in r}), None)
        if head is None:
            return [], [{"row": 1, "problem": "No header row with 'verse' and 'text'. Use the template."}]
        cols = {str(c or "").strip().lower(): i for i, c in enumerate(rows[head])}
        good, bad = [], []
        for n, r in enumerate(rows[head + 1:], head + 2):
            if not r or all(c in (None, "") for c in r):
                continue
            if len(good) >= MAX_ROWS:
                bad.append({"row": n, "problem": f"more than {MAX_ROWS} rows; the rest were skipped"})
                break
            get = lambda k: ("" if cols.get(k) is None or cols[k] >= len(r) or r[cols[k]] is None else str(r[cols[k]]).strip())
            verse = re.sub(r"\.0$", "", get("verse")).translate(str.maketrans("०१२३४५६७८९", "0123456789"))
            kind = (get("kind") or "note").lower()
            text, step = get("text"), re.sub(r"\.0$", "", get("step"))
            if verse not in self.verse_ids:
                bad.append({"row": n, "problem": f"unknown verse '{verse}'"})
            elif kind not in KINDS:
                bad.append({"row": n, "problem": f"kind '{kind}' is not one of {', '.join(KINDS)}"})
            elif not text:
                bad.append({"row": n, "problem": "text is empty"})
            elif len(text) > MAX_TEXT:
                bad.append({"row": n, "problem": f"text is longer than {MAX_TEXT} characters"})
            elif step and step not in {"1", "2", "3", "4", "5", "6", "7"}:
                bad.append({"row": n, "problem": f"step '{step}' must be 1-7"})
            else:
                good.append({"verse_id": self.verse_ids[verse], "kind": kind, "step": int(step) if step else None,
                             "word": get("word")[:100], "text": text, "source": get("source")[:200]})
        return good, bad

    # ---------------------------------------------------------- writes
    def version(self, sid: str) -> int:
        r = self.db.q("SELECT MAX(version) AS v FROM user_notes WHERE student_id=?", (sid,))
        return int(r[0]["v"] or 0) if r else 0

    def upload(self, sid: str, data: bytes) -> dict:
        good, bad = self.parse(data)
        if not good:
            return {"ok_rows": 0, "errors": bad, "version": self.version(sid)}
        v = self.version(sid) + 1
        now = time.time()
        for g in good:
            self.db.q("UPDATE user_notes SET active=0, replaced_in=? WHERE student_id=? AND active=1 AND verse_id=? "
                      "AND kind=? AND word=?", (v, sid, g["verse_id"], g["kind"], g["word"]))
            self.db.q("INSERT INTO user_notes (student_id, verse_id, kind, step, word, text, source, version, created_at) "
                      "VALUES (?,?,?,?,?,?,?,?,?)", (sid, g["verse_id"], g["kind"], g["step"], g["word"], g["text"],
                                                     g["source"], v, now))
        self._index.pop(sid, None)
        return {"ok_rows": len(good), "errors": bad, "version": v}

    def undo(self, sid: str) -> int:
        v = self.version(sid)
        if v:
            self.db.q("DELETE FROM user_notes WHERE student_id=? AND version=?", (sid, v))
            self.db.q("UPDATE user_notes SET active=1, replaced_in=NULL WHERE student_id=? AND replaced_in=?", (sid, v))
            self._index.pop(sid, None)
        return v

    def submit(self, sid: str) -> int:
        rows = self.db.q("SELECT COUNT(*) AS n FROM user_notes WHERE student_id=? AND active=1 AND status='personal'", (sid,))
        self.db.q("UPDATE user_notes SET status='submitted' WHERE student_id=? AND active=1 AND status='personal'", (sid,))
        return int(rows[0]["n"]) if rows else 0

    # ---------------------------------------------------------- reads
    def _shape(self, r: dict) -> dict:
        return {**r, "verse": r["verse_id"].split(".")[-1]}

    def list(self, sid: str) -> list[dict]:
        return [self._shape(r) for r in self.db.q(
            "SELECT id, verse_id, kind, step, word, text, source, version, status FROM user_notes "
            "WHERE student_id=? AND active=1 ORDER BY verse_id, kind, id", (sid,))]

    def for_verse(self, sid: str, shloka_id: str) -> list[dict]:
        return [n for n in self.list(sid) if n["verse_id"] == shloka_id]

    def search(self, sid: str, query: str, k: int = 3) -> list[dict]:
        """Hybrid search over this student's notes only: BM25 + the KB's embedder, fused with RRF."""
        v = self.version(sid)
        if sid not in self._index or self._index[sid][0] != v:
            notes = self.list(sid)
            texts = [f"{n['word']} {n['text']}" for n in notes]
            bm = BM25([bm25_tokens(t) for t in texts]) if notes else None
            mat = None
            if notes and self.kb is not None:
                try:
                    mat = self.kb.embedder.embed_documents(texts)
                except Exception:                          # embedding API down: BM25 alone is fine
                    mat = None
            self._index[sid] = (v, notes, bm, mat)
        _, notes, bm, mat = self._index[sid]
        if not notes:
            return []
        q = self.expander.expand(query) if self.expander else query
        lists = [[i for i, _ in bm.search(bm25_tokens(q), 10)]]
        if mat is not None:
            qv = self.kb.embedder.embed_query(q)
            scores = np.asarray(mat @ qv).ravel()           # dense array or sparse TF-IDF matrix
            lists.append([int(i) for i in np.argsort(-scores)[:10] if scores[i] > 0])
        fused = rrf([lst for lst in lists if lst])[:k]
        return [notes[i] for i, _ in fused]