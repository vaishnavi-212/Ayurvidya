"""Settings, loaded from config/settings.yaml with environment-variable overrides.

Secrets (API keys) are read ONLY from the environment or the git-ignored .env file,
never from the YAML file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    # paths
    data_dir: Path = ROOT / "data"
    kb_dir: Path = ROOT / "data" / "kb"
    prompts_dir: Path = ROOT / "prompts"
    db_path: Path = ROOT / "data" / "ayurvidya.sqlite3"
    # embeddings: "tfidf" (offline, no download) | "st:<model>" (sentence-transformers)
    embedder: str = "tfidf"
    # vector store: "numpy" (exact, built-in) | "chroma"
    vector_store: str = "numpy"
    # reranker: "" (off) | "bge:<model>" e.g. "bge:BAAI/bge-reranker-v2-m3"
    reranker: str = ""
    # retrieval
    dense_k: int = 20
    sparse_k: int = 20
    rrf_k: int = 60
    final_k: int = 6
    # LLM: "anthropic:<model>" | "openai:<model>" (any OpenAI-compatible server) | "mock"
    llm: str = "mock"
    llm_base_url: str = ""          # for openai-compatible servers, e.g. http://localhost:11434/v1
    temperature: float = 0.0
    max_tokens: int = 2000
    max_retries: int = 2           # re-asks when a step's output fails validation
    llm_api_key_env: str = "OPENAI_API_KEY"   # env var holding the key for the step LLM
    # chatbot (its own model + limits; empty chat_llm = use the step LLM)
    chat_llm: str = ""             # e.g. openai:llama-3.1-8b-instant (Groq) | openai:llama3.2:3b (Ollama)
    chat_llm_base_url: str = ""    # https://api.groq.com/openai/v1 | http://localhost:11434/v1
    chat_api_key_env: str = "GROQ_API_KEY"
    chat_max_tokens: int = 400
    chat_temperature: float = 0.2
    chat_retries: int = 1          # one quick retry, never the long back-off
    chat_steps: int = 3            # saved steps put in the chat prompt (most relevant first)
    chat_refs: int = 4             # references put in the chat prompt
    chat_ref_chars: int = 700      # each reference trimmed to this many characters
    chat_max_context_chars: int = 6000
    chat_cache_size: int = 256
    # pipeline switches (each can be turned off for the ablation)
    query_expansion: bool = True   # synonyms + English -> Sanskrit glossary (data/synonyms.csv)
    hyde: bool = False             # one-line hypothetical answer as an extra query (costs 1 LLM call)
    dedupe: bool = True            # drop overlapping / near-duplicate chunks, pick with MMR
    citation_check: bool = True    # mark answer sentences whose cited source does not support them
    relevance_gate: bool = True    # say "not found" without calling the LLM when nothing relevant is retrieved
    gate_min_overlap: float = 0.15
    use_kg: bool = True            # use data/kg/facts.csv when it exists (graph RAG)
    embed_dim: int = 768           # output size for gemini embeddings
    timing_log: str = "outputs/chat_timing.csv"
    extra: dict[str, Any] = field(default_factory=dict)


def load_dotenv(path: str | os.PathLike | None = None) -> None:
    """Read KEY=VALUE lines from .env into the environment (existing variables win).
    Lines starting with # and blank values are ignored."""
    env = Path(path) if path else ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def load_settings(path: str | os.PathLike | None = None) -> Settings:
    load_dotenv()
    s = Settings()
    cfg = Path(path) if path else ROOT / "config" / "settings.yaml"
    if cfg.exists():
        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        for k, v in data.items():
            if not hasattr(s, k):
                raise KeyError(f"Unknown setting '{k}' in {cfg}")
            cur = getattr(s, k)
            setattr(s, k, (ROOT / v) if isinstance(cur, Path) and not Path(v).is_absolute() else
                    (Path(v) if isinstance(cur, Path) else v))
    # environment overrides: AYV_LLM, AYV_EMBEDDER, ...
    for k in list(vars(s)):
        env = os.environ.get("AYV_" + k.upper())
        if env is not None:
            cur = getattr(s, k)
            if isinstance(cur, bool):
                setattr(s, k, env.lower() in ("1", "true", "yes"))
            elif isinstance(cur, int):
                setattr(s, k, int(env))
            elif isinstance(cur, float):
                setattr(s, k, float(env))
            elif isinstance(cur, Path):
                setattr(s, k, Path(env))
            else:
                setattr(s, k, env)
    return s