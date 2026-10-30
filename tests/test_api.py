"""HTTP API tests (need fastapi + httpx: pip install -r requirements-server.txt httpx). Uses a temporary KB,
the mock LLM and a test class code, so no key and no network are needed."""
from __future__ import annotations

import io
import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from fastapi.testclient import TestClient
    HAVE = True
except ImportError:      # pragma: no cover
    HAVE = False


@unittest.skipUnless(HAVE, "fastapi / httpx not installed")
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ayurvidya.index import KnowledgeBase
        from ayurvidya.ingest import build_chunks, load_mula
        from ayurvidya.schema import StepResult
        from ayurvidya.store import Store
        cls.tmp = Path(tempfile.mkdtemp())
        sh, _ = load_mula(ROOT / "data" / "raw" / "niimh_su_ni_1_mula.json")
        KnowledgeBase.build(build_chunks(sh), "tfidf", cls.tmp / "kb")
        st = Store(cls.tmp / "a.sqlite3")
        st.upsert_chapter("su.ni.1", "sushruta", "nidana", 1, "x")
        st.upsert_shlokas("su.ni.1", sh)
        st.save_step(StepResult("su.ni.1.13", 4, "प्राणः = Prana [R1]", True, retrieved_ids=["mula.su.ni.1.13"],
                                model="mock", prompt_version="v"))
        st.close()
        cls.env = {"AYV_KB_DIR": str(cls.tmp / "kb"), "AYV_DB_PATH": str(cls.tmp / "a.sqlite3"), "AYV_LLM": "mock",
                   "AYV_CHAT_LLM": "", "AYV_CLASS_CODE": "TEST1", "AYV_NOTES_SECRET": "s",
                   "DATABASE_URL": str(cls.tmp / "notes.sqlite3"), "AYV_TIMING_LOG": ""}
        cls.old = {k: os.environ.get(k) for k in cls.env}
        os.environ.update(cls.env)
        loaded = "ayurvidya.api" in sys.modules
        import ayurvidya.api as api_mod
        cls.api = importlib.reload(api_mod) if loaded else api_mod
        cls.c = TestClient(cls.api.api)

    @classmethod
    def tearDownClass(cls):
        cls.api.app_state.store.close()
        if cls.api.app_state.notes is not None:
            cls.api.app_state.notes.db.close()
        for k, v in cls.old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_health_verse_source(self):
        self.assertTrue(self.c.get("/health").json()["ok"])
        v = self.c.get("/shlokas/su.ni.1.13").json()
        self.assertEqual(v["steps"]["4"]["refs"]["R1"]["id"], "mula.su.ni.1.13")
        self.assertIn("प्राणो", self.c.get("/source/mula.su.ni.1.13").json()["text"])

    def test_chat_stream(self):
        r = self.c.post("/chat/stream", json={"question": "What is Prana?", "shloka_id": "su.ni.1.13"})
        events = [json.loads(l[5:]) for l in r.text.split("\n\n") if l.startswith("data:")]
        self.assertEqual(events[-1]["type"], "done")
        self.assertGreater(sum(e["type"] == "token" for e in events), 1)
        self.assertIn("tags", events[-1])
        self.assertEqual(self.c.post("/chat", json={"question": "x", "shloka_id": "su.ni.1.999"}).status_code, 404)

    def test_notes_flow(self):
        from openpyxl import Workbook
        if self.api.app_state.notes is None:
            self.skipTest("student notes not added yet (notes.py)")
        self.assertEqual(self.c.get("/notes").status_code, 401)
        self.assertEqual(self.c.post("/notes/login", json={"class_code": "bad", "name": "A"}).status_code, 403)
        tok = self.c.post("/notes/login", json={"class_code": "TEST1", "name": "Asha"}).json()["token"]
        wb = Workbook()
        ws = wb.active
        ws.title = "notes"
        ws.append(["verse", "kind", "step", "word", "text", "source"])
        ws.append([13, "translation", 4, "", "Prana moves in the mouth.", ""])
        buf = io.BytesIO()
        wb.save(buf)
        r = self.c.post("/notes/upload", headers={"X-Student": tok},
                        files={"file": ("n.xlsx", buf.getvalue(), "application/octet-stream")})
        self.assertEqual(r.json()["ok_rows"], 1)
        self.assertEqual(len(self.c.get("/notes", headers={"X-Student": tok}).json()), 1)
        self.assertEqual(len(self.c.get("/notes/verse/su.ni.1.13", headers={"X-Student": tok}).json()), 1)
        self.assertEqual(self.c.get("/notes/template").status_code, 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
    