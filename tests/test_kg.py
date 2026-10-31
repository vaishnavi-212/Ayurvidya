"""Knowledge graph: loading, linking, 2-hop walk, the RRF vote, chat facts and the Vayu map."""
from __future__ import annotations

import csv
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ayurvidya.chatbot import ChatConfig, Chatbot  # noqa: E402
from ayurvidya.index import KnowledgeBase  # noqa: E402
from ayurvidya.ingest import build_chunks, load_mula  # noqa: E402
from ayurvidya.kg import KnowledgeGraph  # noqa: E402
from ayurvidya.llm import MockLLM  # noqa: E402
from ayurvidya.query import QueryExpander  # noqa: E402
from ayurvidya.retrieve import HybridRetriever  # noqa: E402
from ayurvidya.store import Store  # noqa: E402

MULA = ROOT / "data" / "raw" / "niimh_su_ni_1_mula.json"
SYN = ROOT / "data" / "synonyms.csv"


class KGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.sh, _ = load_mula(MULA)
        cls.kb = KnowledgeBase.build(build_chunks(cls.sh), "tfidf", cls.tmp / "kb")
        cls.g = KnowledgeGraph.load(ROOT / "data" / "kg" / "facts.csv", SYN, cls.kb)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_seed_quotes_are_in_their_verses(self):
        text = {f"su.ni.1.{s.number}": s.text for s in self.sh}
        for f in self.g.facts:
            self.assertIn(f.quote, text[f.verse], f)

    def test_two_hop(self):
        lines = [l for l, _ in self.g.facts_for("Which Vayu causes hiccup? Where does it move?", k=6)]
        self.assertTrue(any("hiccup" in l for l in lines))
        self.assertTrue(any("Prana" in l and "seat" in l for l in lines))     # reached through Prana

    def test_any_script(self):
        lines = [l for l, _ in self.g.facts_for("समानः कुत्र", k=3)]
        self.assertTrue(all("Samana" in l for l in lines))

    def test_graph_vote_in_retrieval(self):
        r = HybridRetriever(self.kb, expander=QueryExpander(SYN), graph=self.g)
        ids = [h.chunk.id for h in r.search("Where is Samana located and which diseases does it cause?", 5, ["mula"])]
        self.assertIn("mula.su.ni.1.16", ids)
        self.assertIn("mula.su.ni.1.17", ids)
        self.assertEqual(self.g.rank_chunks("How do I bake a cake?", self.kb), [])  # no vote when unsure

    def test_status_filter_and_summary(self):
        p = self.tmp / "f.csv"
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["subject", "relation", "object", "object_deva", "verse", "quote", "status"])
            w.writerow(["Prana", "located_in", "mouth", "वक्त्र", "13", "वक्त्रसञ्चारी", "approved"])
            w.writerow(["Prana", "causes", "fever", "ज्वर", "13", "x", "draft"])
        g = KnowledgeGraph.load(p, SYN, self.kb)
        self.assertEqual(len(g.facts), 1)                                     # drafts are not used
        s = g.summary()
        self.assertEqual(s["vayus"][0]["sthana"][0]["verse"], "su.ni.1.13")
        self.assertEqual(s["checked"], 1)

    def test_chat_gets_graph_facts(self):
        store = Store(self.tmp / "c.sqlite3")
        store.upsert_chapter("su.ni.1", "sushruta", "nidana", 1, "x")
        store.upsert_shlokas("su.ni.1", self.sh)
        bot = Chatbot(HybridRetriever(self.kb, graph=self.g), MockLLM(), store, ROOT / "prompts", cfg=ChatConfig(),
                      graph=self.g)
        bot.ask("Which Vayu lives in the colon and what does it do?")
        self.assertIn("[G1] Apana", bot.llm.calls[0][1])
        store.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)