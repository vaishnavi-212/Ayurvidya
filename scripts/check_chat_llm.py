"""Check that the CHAT model (Llama on Groq, or Ollama) answers, and how fast.

    python scripts/check_chat_llm.py
Reads AYV_CHAT_LLM / AYV_CHAT_LLM_BASE_URL / AYV_CHAT_API_KEY_ENV from .env (see .env.example).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ayurvidya.config import load_settings  # noqa: E402
from ayurvidya.llm import LLMError, make_llm  # noqa: E402


def main() -> None:
    s = load_settings()
    spec, url = s.chat_llm or s.llm, s.chat_llm_base_url if s.chat_llm else s.llm_base_url
    key_env = s.chat_api_key_env if s.chat_llm else s.llm_api_key_env
    print(f"chat model : {spec}\nserver     : {url or '(default)'}\nkey        : {key_env} "
          f"{'is set' if os.environ.get(key_env) else 'is NOT set (fine for Ollama)'}")
    if spec == "mock":
        sys.exit("\nStill on the mock model. Set AYV_CHAT_LLM in .env (see .env.example).")
    llm = make_llm(spec, url, s.chat_temperature, 120, key_env, retries=1)
    t0 = time.time()
    first = None
    out = []
    try:
        for tok in llm.stream("You are a Sanskrit teacher. Answer in one short English sentence.",
                              "What does the word वायुः mean?"):
            first = first or time.time() - t0
            out.append(tok)
    except LLMError as e:
        print(f"\nFAILED: {e}\nCheck the key in .env, the model name, and for Ollama that the Ollama app is running.")
        sys.exit(1)
    print(f"\nreply      : {''.join(out).strip()}\nfirst word : {first:.2f} s\ntotal      : {time.time() - t0:.2f} s"
          "\n\nOK - the chat model is connected.")


if __name__ == "__main__":
    main()