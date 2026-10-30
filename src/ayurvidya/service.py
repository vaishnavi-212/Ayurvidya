"""Wires settings -> knowledge base -> retriever -> LLMs -> store -> orchestrator/chatbot.
Both the scripts and the API create the app through `build_app()`, so they always agree.

Two LLMs: the STEP LLM (Gemini, offline generation of the 7 steps) and the CHAT LLM (Llama via Groq or
Ollama, live). Optional parts switch on when their files exist: the knowledge graph (data/kg/facts.csv,
module kg.py) and student notes (module notes.py)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chatbot import ChatConfig, Chatbot
from .config import Settings, load_settings
from .index import KnowledgeBase
from .llm import make_llm
from .pipeline import Orchestrator
from .query import QueryExpander
from .retrieve import HybridRetriever, Reranker, RetrievalConfig
from .store import Store


@dataclass
class App:
    settings: Settings
    kb: KnowledgeBase
    retriever: HybridRetriever
    store: Store
    orchestrator: Orchestrator
    chatbot: Chatbot
    graph: object = None
    notes: object = None


def _graph(s: Settings, kb: KnowledgeBase):
    facts = Path(s.data_dir) / "kg" / "facts.csv"
    if not (s.use_kg and facts.exists()):
        return None
    try:
        from .kg import KnowledgeGraph
    except ImportError:
        return None
    return KnowledgeGraph.load(facts, Path(s.data_dir) / "synonyms.csv", kb)


def _notes(s: Settings, kb: KnowledgeBase, store: Store):
    try:
        from .notes import NotesService
    except ImportError:
        return None
    return NotesService.from_settings(s, kb, store)


def build_app(settings: Settings | None = None, llm=None, chat_llm=None) -> App:
    s = settings or load_settings()
    kb = KnowledgeBase.load(s.kb_dir)
    reranker = Reranker(s.reranker.split(":", 1)[1]) if s.reranker.startswith("bge:") else None
    expander = QueryExpander(Path(s.data_dir) / "synonyms.csv") if s.query_expansion else None
    graph = _graph(s, kb)
    retriever = HybridRetriever(kb, RetrievalConfig(s.dense_k, s.sparse_k, s.rrf_k, s.final_k), reranker,
                                expander=expander, graph=graph, dedupe=s.dedupe)
    llm = llm or make_llm(s.llm, s.llm_base_url, s.temperature, s.max_tokens, s.llm_api_key_env)
    if chat_llm is None:
        if s.chat_llm:
            chat_llm = make_llm(s.chat_llm, s.chat_llm_base_url, s.chat_temperature, s.chat_max_tokens,
                                s.chat_api_key_env, retries=s.chat_retries)
        else:
            chat_llm = llm
    store = Store(s.db_path)
    notes = _notes(s, kb, store)
    orch = Orchestrator(retriever, llm, store, s.prompts_dir, s.max_retries)
    bot = Chatbot(retriever, chat_llm, store, s.prompts_dir, cfg=ChatConfig.from_settings(s), notes=notes, graph=graph)
    return App(s, kb, retriever, store, orch, bot, graph, notes)