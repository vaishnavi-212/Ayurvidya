"""LLM clients behind two methods: complete(system, user) -> str and stream(system, user) -> tokens.

  anthropic:<model>   Claude via the Messages API (needs ANTHROPIC_API_KEY)
  openai:<model>      any OpenAI-compatible endpoint: Gemini, Groq (Llama), Ollama, vLLM, OpenAI ...
                      key read from the env var named by `api_key_env`; base URL from settings
  mock                deterministic offline stub used by the test-suite

Only `requests` is used, so no vendor SDK is required. Step generation retries with exponential
back-off (it runs offline); the chatbot uses `retries=1` so a student never waits half a minute.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Callable, Iterator, Protocol

import requests


class LLM(Protocol):
    name: str
    def complete(self, system: str, user: str) -> str: ...
    def stream(self, system: str, user: str) -> Iterator[str]: ...


class LLMError(RuntimeError):
    pass


RETRY_CODES = (429, 500, 502, 503, 504, 529)


def _post_with_retry(url: str, headers: dict, payload: dict, tries: int = 5, timeout: int = 120,
                     delay: float = 2.0, stream: bool = False) -> requests.Response:
    for attempt in range(1, tries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout, stream=stream)
        except requests.RequestException as e:
            if attempt == tries:
                raise LLMError(f"network error: {e}") from e
        else:
            if r.status_code == 200:
                return r
            if r.status_code not in RETRY_CODES or attempt == tries:
                raise LLMError(f"HTTP {r.status_code}: {r.text[:500]}")
        time.sleep(delay)
        delay *= 2
    raise LLMError("unreachable")


class AnthropicLLM:
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 2000, retries: int = 5):
        self.key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.key:
            raise LLMError("Set ANTHROPIC_API_KEY in your environment (.env)")
        self.name, self.model, self.temperature, self.max_tokens = f"anthropic:{model}", model, temperature, max_tokens
        self.tries = retries if retries > 1 else retries + 1

    def complete(self, system: str, user: str) -> str:
        data = _post_with_retry(self.URL, {
            "x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json",
        }, {
            "model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
        }, tries=self.tries).json()
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()

    def stream(self, system: str, user: str) -> Iterator[str]:
        yield self.complete(system, user)


class OpenAICompatLLM:
    def __init__(self, model: str, base_url: str = "", temperature: float = 0.0, max_tokens: int = 2000,
                 api_key_env: str = "OPENAI_API_KEY", retries: int = 5, timeout: int = 120):
        self.base = (base_url or "https://api.openai.com/v1").rstrip("/")
        # only the named key is sent: a Gemini key must never go to Groq or Ollama by accident
        self.key = os.environ.get(api_key_env) or "not-needed-for-local"
        self.name, self.model, self.temperature, self.max_tokens = f"openai:{model}", model, temperature, max_tokens
        # steps: 5 tries with 2-4-8-16 s back-off · chat (retries <= 1): first try + one retry after 1 s
        self.tries = retries if retries > 1 else retries + 1
        self.delay, self.timeout = (2.0 if retries > 1 else 1.0), timeout

    def _payload(self, system: str, user: str, stream: bool) -> dict:
        return {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                "stream": stream,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}

    def complete(self, system: str, user: str) -> str:
        data = _post_with_retry(f"{self.base}/chat/completions", self._headers(), self._payload(system, user, False),
                                tries=self.tries, timeout=self.timeout, delay=self.delay).json()
        return (data["choices"][0]["message"]["content"] or "").strip()

    def stream(self, system: str, user: str) -> Iterator[str]:
        """Server-sent events from /chat/completions with stream=true (Groq, Ollama, Gemini, OpenAI)."""
        r = _post_with_retry(f"{self.base}/chat/completions", self._headers(), self._payload(system, user, True),
                             tries=self.tries, timeout=self.timeout, delay=self.delay, stream=True)
        with r:
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                except (ValueError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta


class MockLLM:
    """Offline stand-in. Produces format-valid output for each step so the whole pipeline, cache,
    API and evaluation code can be tested without a key. `responder` overrides the default."""
    name = "mock"

    def __init__(self, responder: Callable[[str, str], str] | None = None):
        self.responder = responder
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.responder:
            return self.responder(system, user)
        m = re.search(r"TASK: step (\d)", user)
        step = int(m.group(1)) if m else 0
        sh = re.search(r"SHLOKA[^\n]*\n(.+?)\n\n", user, re.S)
        words = re.findall(r"[ऀ-ॿ]+", sh.group(1) if sh else "")
        words = [w for w in words if not re.fullmatch(r"[।॥०-९]+", w)][:8] or ["पदम्"]
        if step == 2:
            return "\n".join(f"{i}. {w} — सुबन्त — पुं., प्रथमा, एकवचनम्" for i, w in enumerate(words, 1))
        if step == 3:
            return " ".join(words) + "।"
        if step == 4:
            return " | ".join(f"{w} = meaning of {w}" for w in words) + "\nThe verse states its literal sense [R1]."
        if step == 5:
            return "The author introduces the topic of this verse [R1]. It fits the chapter on Vata disorders."
        if step == 6:
            return (f"पद: {words[0]}\nSynonyms: —\nGrammar: सुबन्त\nधातु: — | धात्वर्थ: —\nउपसर्ग: —\n"
                    f"Contribution: Its form adds emphasis.")
        if step == 7:
            return "The wording hints at a deeper intent of the Acharya [R1]. It links to parallel texts."
        if step == 0:   # chatbot / judge calls
            return "Based on the references [R1], the verse explains this point."
        return "1st पाद- " + " ".join(words[:2])

    def stream(self, system: str, user: str) -> Iterator[str]:
        for w in re.findall(r"\S+\s*", self.complete(system, user)):
            yield w


def make_llm(spec: str, base_url: str = "", temperature: float = 0.0, max_tokens: int = 2000,
             api_key_env: str = "OPENAI_API_KEY", retries: int = 5) -> LLM:
    if spec == "mock":
        return MockLLM()
    provider, _, model = spec.partition(":")
    if provider == "anthropic":
        return AnthropicLLM(model, temperature, max_tokens, retries)
    if provider == "openai":
        return OpenAICompatLLM(model, base_url, temperature, max_tokens, api_key_env, retries,
                               timeout=120 if retries > 1 else 45)
    raise ValueError(f"Unknown llm spec '{spec}' (anthropic:<model> | openai:<model> | mock)")