"""LLM provider abstraction.

Everything above this module talks to `LLMClient.complete()` and catches the
`LLMError` subclasses. Swapping Groq for Gemini, or for the FakeLLM in tests,
changes nothing elsewhere.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from app.config import Settings

Message = dict[str, str]  # {"role": "system"|"user", "content": "..."}


# --- errors: the pipeline maps all of these to a "try again" response -------

class LLMError(Exception):
    """Base class for every provider failure."""


class LLMTimeout(LLMError):
    pass


class LLMRateLimited(LLMError):
    """Quota exhausted (HTTP 429). Not worth retrying within one request."""


class LLMUnavailable(LLMError):
    """5xx, network error, or a response we cannot read."""


class LLMBudgetExceeded(LLMError):
    """A code path tried to make more LLM calls than allowed. This is a bug."""


# --- interface ---------------------------------------------------------------

class LLMClient(Protocol):
    name: str

    def complete(self, messages: list[Message], timeout_s: float) -> str:
        """Return the raw text of the model's reply (expected to be JSON)."""
        ...


@dataclass
class LLMBudget:
    """Per-question call counter. Enforces the 'at most N LLM calls' rule in code."""

    client: LLMClient
    max_calls: int
    calls: int = 0

    def complete(self, messages: list[Message], timeout_s: float) -> str:
        if self.calls >= self.max_calls:
            raise LLMBudgetExceeded(f"LLM call budget of {self.max_calls} exhausted")
        self.calls += 1
        return self.client.complete(messages, timeout_s)


# --- Groq (OpenAI-compatible endpoint, free tier) ------------------------------

class GroqClient:
    name = "groq"

    def __init__(self, api_key: str, model: str, base_url: str, reasoning_effort: str = "",
                 transport: httpx.BaseTransport | None = None):
        if not api_key:
            raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        self._model = model
        self._reasoning_effort = reasoning_effort
        # `transport` lets tests replace the network with a mock.
        self._http = httpx.Client(base_url=base_url, headers={"Authorization": f"Bearer {api_key}"},
                                  transport=transport)

    def complete(self, messages: list[Message], timeout_s: float) -> str:
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        if self._reasoning_effort:
            # Reasoning models think before answering; keeping it minimal keeps latency low.
            payload["reasoning_effort"] = self._reasoning_effort
        try:
            resp = self._http.post("/chat/completions", json=payload, timeout=timeout_s)
        except httpx.TimeoutException as e:
            raise LLMTimeout(str(e)) from e
        except httpx.HTTPError as e:
            raise LLMUnavailable(str(e)) from e

        if resp.status_code == 429:
            raise LLMRateLimited("provider quota exhausted")
        if resp.status_code >= 400:
            raise LLMUnavailable(f"provider returned HTTP {resp.status_code}")
        try:
            return resp.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError) as e:
            raise LLMUnavailable("unexpected provider response shape") from e


# --- Fake (tests, offline eval, demo without a key) ------------------------------

Responder = Callable[[list[Message]], str]

_EXCERPT = re.compile(r'<excerpt id="(C\d+)"[^>]*>\n(.*?)\n</excerpt>', re.DOTALL)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
NOT_COVERED_JSON = '{"status": "not_covered", "answer": "", "citations": [], "quote": ""}'


def excerpts_in(messages: list[Message]) -> dict[str, str]:
    """label -> excerpt text, parsed back out of a prompt (used by fakes and tests)."""
    return dict(_EXCERPT.findall("\n".join(m["content"] for m in messages)))


def extractive_responder(messages: list[Message]) -> str:
    """A 'model' that answers with the first sentence of the top excerpt.

    Deterministic and free. It makes the offline pipeline exercise every step
    (citations, quote check), but it does not understand the question, so it
    is only meaningful for retrieval metrics, never for answer quality.
    """
    excerpts = excerpts_in(messages)
    if not excerpts:
        return NOT_COVERED_JSON
    label, text = next(iter(excerpts.items()))
    sentence = _SENTENCE_END.split(text.strip())[0]
    return json.dumps({"status": "answered", "answer": sentence, "citations": [label], "quote": sentence})


@dataclass
class FakeLLM:
    """Scriptable fake. Give it a responder function, or a list of replies/exceptions.

    Every call is recorded in `received`, so tests can inspect the exact prompt.
    """

    responder: Responder = extractive_responder
    script: list[str | Exception] = field(default_factory=list)
    received: list[list[Message]] = field(default_factory=list)
    name: str = "fake"

    def complete(self, messages: list[Message], timeout_s: float) -> str:
        self.received.append(list(messages))
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return self.responder(messages)


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "groq":
        return GroqClient(settings.groq_api_key, settings.groq_model, settings.groq_base_url,
                          settings.groq_reasoning_effort)
    return FakeLLM()
