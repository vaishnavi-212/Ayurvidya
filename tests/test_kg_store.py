"""Scholar fact uploads: seeding, row checks (quote must be in the verse), versions + undo, live graph,
export snapshot, and section summaries (with the mock LLM)."""
from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ayurvidya.config import Settings  # noqa: E402
from ayurvidya.index import KnowledgeBase  # noqa: E402
from ayurvidya.ingest import build_chunks, load_mula  # noqa: E402
from ayurvidya.kg_store import KGStore  # noqa: E402
from ayurvidya.notes import NotesDB  # noqa: E402
from ayurvidya.service import build_app  # noqa: E402
from ayurvidya.store import Store  # noqa: E402

MULA = ROOT / "data" / "raw" / "niimh_su_ni_1_mula.json"
HEAD = "subject,relation,object,object_deva,verse,quote,status\n"


class KGStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.sh, _ = load_mula(MULA)
        cls.kb = KnowledgeBase.build(build_chunks(cls.sh), "tfidf", cls.tmp / "kb")
        cls.store = Store(cls.tmp / "s.sqlite3")
        cls.store.upsert_chapter("su.ni.1", "sushruta", "nidana", 1, "x")
        cls.store.upsert_shlokas("su.ni.1", cls.sh)

    @classmethod
    def tearDownClass(cls):
        cls.store.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.db = NotesDB(str(self.tmp / f"k{id(self)}.sqlite3"))
        self.kg = KGStore.for_app(self.db, self.store)
        self.kg.seed_from_csv(ROOT / "data" / "kg" / "facts.csv")

    def tearDown(self):
        self.db.close()

    def test_seed_once(self):
        self.assertEqual(self.kg.version(), 1)
        self.assertEqual(len(self.kg.rows()), 38)
        self.assertEqual(self.kg.seed_from_csv(ROOT / "data" / "kg" / "facts.csv"), 0)   # not twice

    def test_rows_are_checked_against_the_verse(self):
        csv = (HEAD + "Prana,causes,fever,ज्वर,13,ज्वरं करोति,approved\n"          # quote not in verse 13
               "Agni,causes,x,x,13,प्राणो,approved\n"                                 # bad subject
               "Prana,located_in,mouth,वक्त्र,95,x,approved\n"                        # unknown verse
               "Udana,performs,speech,भाषित,15,भाषितगीतादि,approved\n")               # good
        r = self.kg.upload(csv.encode("utf-8"), "f.csv", "Dr A")
        self.assertEqual(r["ok_rows"], 1)
        self.assertEqual([e["row"] for e in r["errors"]], [2, 3, 4])
        self.assertIn("word-for-word", r["errors"][0]["problem"])

    def test_replace_reject_undo_and_export(self):
        before = len(self.kg.facts())
        csv = HEAD + "Prana,located_in,mouth,वक्त्र,13,यो वायुर्वक्त्रसञ्चारी स प्राणो नाम,rejected\n"
        self.kg.upload(csv.encode("utf-8"), "f.csv", "Dr A")
        self.assertEqual(len(self.kg.facts()), before - 1)                 # rejected facts leave the graph
        self.assertNotIn("Prana,located_in,mouth,", self.kg.export_csv())
        self.kg.undo()
        self.assertEqual(len(self.kg.facts()), before)
        self.assertEqual(self.kg.undo(), 1)                                # the seed is never undone
        self.assertEqual(len(self.kg.rows()), 38)

    def test_xlsx_upload_changes_the_graph(self):
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.append(["subject", "relation", "object", "object_deva", "verse", "quote", "status"])
        ws.append(["Vayu", "causes", "in the skin: discolouration, throbbing", "वैवर्ण्य स्फुरण", 25,
                   "वैवर्ण्यं स्फुरणं रौक्ष्यं", "approved"])
        buf = io.BytesIO()
        wb.save(buf)
        self.assertEqual(self.kg.upload(buf.getvalue(), "f.xlsx", "Dr A")["ok_rows"], 1)
        g = self.kg.graph(ROOT / "data" / "synonyms.csv", self.kb)
        self.assertTrue(any("discolouration" in l for l, _ in g.facts_for("Which Vayu causes discolouration of the skin?")))


class SummaryTests(unittest.TestCase):
    def test_section_summaries_with_mock(self):
        import make_section_summaries as mss
        tmp = Path(tempfile.mkdtemp())
        try:
            sh, _ = load_mula(MULA)
            KnowledgeBase.build(build_chunks(sh), "tfidf", tmp / "kb")
            app = build_app(Settings(kb_dir=tmp / "kb", db_path=tmp / "a.sqlite3", llm="mock", data_dir=ROOT / "data"))
            app.store.upsert_chapter("su.ni.1", "sushruta", "nidana", 1, "x")
            app.store.upsert_shlokas("su.ni.1", sh)
            n = mss.build(app, tmp / "s.jsonl", pause=0)
            rows = [json.loads(l) for l in (tmp / "s.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(n, 10)
            self.assertEqual(rows[2]["source_type"], "summary")
            self.assertIn("su.ni.1.13", rows[2]["shloka_ids"])
            app.store.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)