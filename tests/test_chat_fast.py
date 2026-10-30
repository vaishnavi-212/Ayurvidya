"""Fast chatbot tests: dedupe/MMR, trimming, step choice, query expansion, citation check, relevance gate,
cache, streaming, timing log and the separate chat LLM. All offline (MockLLM)."""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ayurvidya.chatbot import NOT_FOUND, ChatConfig, Chatbot, pick_steps, trim  # noqa: E402
from ayurvidya.config import Settings  # noqa: E402
from ayurvidya.index import KnowledgeBase  # noqa: E402
from ayurvidya.ingest import build_chunks, load_mula  # noqa: E402
from ayurvidya.llm import MockLLM, OpenAICompatLLM, make_llm  # noqa: E402
from ayurvidya.query import QueryExpander, detect_script  # noqa: E402
from ayurvidya.retrieve import HybridRetriever, dedupe_mmr  # noqa: E402
from ayurvidya.schema import Chunk, Hit, StepResult  # noqa: E402
from ayurvidya.service import build_app  # noqa: E402
from ayurvidya.store import Store  # noqa: E402
from ayurvidya.verify import check_citations, citation_precision, parse_tags  # noqa: E402

MULA = ROOT / "data" / "raw" / "niimh_su_ni_1_mula.json"
SYN = ROOT / "data" / "synonyms.csv"


class UnitTests(unittest.TestCase):
    def test_dedupe_mmr_drops_near_duplicates(self):
        a = Chunk("a", "प्राणो नाम वक्त्रसञ्चारी वायुः अन्नं प्रवेशयति देहधृक्", "commentary")
        b = Chunk("b", "प्राणो नाम वक्त्रसञ्चारी वायुः अन्नं प्रवेशयति देहधृक् ।", "crosstext")
        c = Chunk("c", "उदानो नाम यस्तूर्ध्वमुपैति पवनोत्तमः", "mula")
        out = dedupe_mmr([Hit(a, 1, 1), Hit(b, .9, 2), Hit(c, .8, 3)], 3)
        self.assertEqual([h.chunk.id for h in out], ["a", "c"])
        self.assertEqual([h.chunk.id for h in dedupe_mmr([Hit(a, 1, 1), Hit(c, .8, 2)], 2, against=[b.text])], ["c"])

    def test_trim_keeps_matching_sentence(self):
        text = "One unrelated sentence here. " * 30 + "The Prana moves in the mouth. " + "Filler text again. " * 30
        out = trim(text, {"prana", "mouth"}, 120)
        self.assertIn("Prana moves in the mouth", out)
        self.assertLessEqual(len(out), 130)

    def test_pick_steps(self):
        steps = {n: {"output": f"text {n}"} for n in range(1, 8)}
        self.assertIn(4, pick_steps("anything", steps, 3))
        self.assertIn(6, pick_steps("Which root does this word come from?", steps, 3))
        self.assertIn(2, pick_steps("Explain the grammar and case", steps, 3))
        self.assertEqual(len(pick_steps("x", steps, 3)), 3)

    def test_query_expansion(self):
        qe = QueryExpander(SYN)
        info = qe.analyse("Which Vayu moves in the mouth? verse 13")
        self.assertEqual(info.verse, "13")
        self.assertIn("mouth", info.groups)
        self.assertIn("वक्त्र", qe.expand("Which Vayu moves in the mouth?"))
        self.assertIn("vata", qe.analyse("वायुः कुपितः").groups)
        self.assertEqual(detect_script("प्राणः"), "devanagari")
        self.assertEqual(detect_script("prāṇa"), "iast")

    def test_citation_check(self):
        self.assertEqual(parse_tags("x [R1, 2] y [S4]"), ["R1", "R2", "S4"])
        texts = {"S4": "प्राणः = the Prana vayu which moves in the mouth and pushes food inside",
                 "R1": "प्राणो नाम वक्त्रसञ्चारी"}
        checks = check_citations("Prana moves in the mouth and pushes food inside [S4]. "
                                 "It causes fever in the bones [S4]. It is called प्राणो वक्त्रसञ्चारी [R1].", texts)
        self.assertEqual([c["status"] for c in checks], ["supported", "weak", "supported"])
        self.assertAlmostEqual(citation_precision(checks), 0.667, places=2)

    def test_chat_llm_retries(self):
        chat = OpenAICompatLLM("llama-3.1-8b-instant", "https://api.groq.com/openai/v1", retries=1, api_key_env="X_NONE")
        self.assertEqual((chat.tries, chat.delay), (2, 1.0))
        steps = make_llm("openai:gemini-3.5-flash-lite", "x")
        self.assertEqual(steps.tries, 5)


class ChatbotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.shlokas, _ = load_mula(MULA)
        cls.kb = KnowledgeBase.build(build_chunks(cls.shlokas), "tfidf", cls.tmp / "kb")
        cls.store = Store(cls.tmp / "c.sqlite3")
        cls.store.upsert_chapter("su.ni.1", "sushruta", "nidana", 1, "Vatavyadhi Nidana")
        cls.store.upsert_shlokas("su.ni.1", cls.shlokas)
        for n in range(1, 8):
            cls.store.save_step(StepResult("su.ni.1.13", n, f"Step {n}: प्राणः = Prana, the vayu moving in the mouth",
                                           True, model="mock", prompt_version="v"))

    @classmethod
    def tearDownClass(cls):
        cls.store.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def bot(self, **kw):
        r = HybridRetriever(self.kb, expander=QueryExpander(SYN))
        return Chatbot(r, MockLLM(), self.store, ROOT / "prompts", cfg=ChatConfig(**kw))

    def test_answer_is_capped_cited_and_timed(self):
        bot = self.bot(max_context_chars=1500, timing_log=str(self.tmp / "t.csv"))
        a = bot.ask("Which Vayu moves in the mouth?", "su.ni.1.13")
        user = bot.llm.calls[0][1]
        self.assertLessEqual(len(user), 1500 + 200)
        self.assertEqual(user.count("[S"), 3)                 # only 3 of 7 steps
        self.assertTrue(a.grounded)
        self.assertIn("first_token_ms", a.timings)
        self.assertTrue((self.tmp / "t.csv").exists())

    def test_cache_and_stream(self):
        bot = self.bot()
        toks = [v for k, v in bot.stream("What is Prana?", "su.ni.1.13") if k == "token"]
        self.assertGreater(len(toks), 1)                     # streamed word by word
        a2 = bot.ask("what is  prana?", "su.ni.1.13")
        self.assertTrue(a2.cached)
        self.assertEqual(len(bot.llm.calls), 1)

    def test_relevance_gate(self):
        bot = self.bot()
        a = bot.ask("What is the capital of France and the price of petrol?")
        self.assertEqual(a.answer, NOT_FOUND)
        self.assertEqual(len(bot.llm.calls), 0)              # no LLM call at all

    def test_expansion_finds_verse_13_for_english(self):
        r = HybridRetriever(self.kb, expander=QueryExpander(SYN))
        ids = [h.chunk.id for h in r.search("Which prana moves in the mouth?", 5, source_types=["mula"])]
        self.assertIn("mula.su.ni.1.13", ids)


class WiringTests(unittest.TestCase):
    def test_separate_chat_llm(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            sh, _ = load_mula(MULA)
            KnowledgeBase.build(build_chunks(sh), "tfidf", tmp / "kb")
            s = Settings(kb_dir=tmp / "kb", db_path=tmp / "a.sqlite3", llm="mock", data_dir=ROOT / "data",
                         chat_llm="openai:llama-3.1-8b-instant", chat_llm_base_url="https://api.groq.com/openai/v1")
            app = build_app(s)
            self.assertEqual(app.chatbot.llm.name, "openai:llama-3.1-8b-instant")
            self.assertEqual(app.orchestrator.llm.name, "mock")
            app.store.close()
            s.chat_llm = ""
            app = build_app(s)
            self.assertIs(app.chatbot.llm, app.orchestrator.llm)   # empty chat_llm -> same as steps
            app.store.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)