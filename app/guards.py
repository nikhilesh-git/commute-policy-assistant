"""Deterministic safety checks, in Python, on both sides of the LLM.

Input:  question cleanup and length, questions naming another company.
Ingest: replace instruction-like sentences in policy text (prompt injection).
Output: JSON schema, citations must be excerpts we sent, the quote must appear
        in a cited excerpt, the answer must not name another company.
Anything that fails an output check becomes "not covered": a missing answer
is safer than a wrong one.
"""

import html
import json
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

# --- prompt injection in policy text -------------------------------------------

INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(all\s+|any\s+)?(previous|prior|above)\s+(instructions|rules)",
        r"\byou\s+are\s+now\b",
        r"^\s*system\s*(note|prompt)?\s*:",
        r"\bdisregard\b.*\b(instructions|rules|policy)\b",
    )
]

REMOVED_MARKER = "[removed: instruction-like text]"
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def sanitise_policy_text(text: str) -> tuple[str, bool]:
    """Replace instruction-like sentences with a marker. Return (text, flagged).

    Works per sentence, so a legitimate rule on the same line survives.
    This is one layer only: a cleverly worded instruction will not match
    (see the Kestrel sample policy), which is why the prompt framing and the
    output validation exist too.
    """
    flagged = False
    out_lines = []
    for line in text.splitlines():
        kept = []
        for sentence in _SENTENCE_END.split(line):
            if any(p.search(sentence) for p in INJECTION_PATTERNS):
                kept.append(REMOVED_MARKER)
                flagged = True
            else:
                kept.append(sentence)
        out_lines.append(" ".join(kept))
    return "\n".join(out_lines), flagged


# --- input -------------------------------------------------------------------------

class QuestionRejected(Exception):
    """The question is unusable; carries the user-facing message. (HTTP 422)"""


def clean_question(question: str, max_chars: int) -> str:
    """Strip control characters, enforce length; raise QuestionRejected if unusable."""
    question = "".join(ch for ch in question if ch.isprintable() or ch == " ").strip()
    if not question:
        raise QuestionRejected("Please type a question.")
    if len(question) > max_chars:
        raise QuestionRejected(f"Please keep your question under {max_chars} characters.")
    return question


def mentioned_company(text: str, aliases: list[str]) -> str | None:
    """Return the first alias found in text as a whole word (case-insensitive)."""
    for alias in aliases:
        if re.search(rf"\b{re.escape(alias)}\b", text, re.IGNORECASE):
            return alias
    return None


# --- output -------------------------------------------------------------------------

class MalformedOutput(Exception):
    """The model's reply is not the JSON we asked for. Worth one retry."""


class LLMReply(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: Literal["answered", "not_covered"]
    answer: str = ""
    citations: list[str] = []
    quote: str = ""


@dataclass(frozen=True)
class ValidatedAnswer:
    status: Literal["answered", "not_covered"]
    answer: str
    cited_labels: list[str]  # subset of what was sent, deduplicated, in the model's order
    quote: str
    reason: str  # why it ended up with this status (logged; used in eval notes)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_llm_reply(raw: str) -> LLMReply:
    text = _FENCE.sub("", raw.strip())
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        raise MalformedOutput(f"not JSON: {e}") from e
    if not isinstance(data, dict):
        raise MalformedOutput("JSON is not an object")
    try:
        return LLMReply.model_validate(data)
    except ValidationError as e:
        raise MalformedOutput(f"wrong shape: {e.error_count()} error(s)") from e


_TYPOGRAPHY = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",  # hyphen, non-breaking hyphen, dashes
})


def normalise(text: str) -> str:
    """For quote matching: undo our escaping, unify quotes/dashes, ignore case and spacing.

    Models often emit typographic characters (gpt-oss writes "ad‑hoc" with a
    non-breaking hyphen and "8:00 PM" with a narrow no-break space); \\s already
    covers the special spaces, the table above covers the rest.
    """
    text = html.unescape(text)
    text = text.translate(_TYPOGRAPHY)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text.strip(" \"'.,;:")


MIN_QUOTE_CHARS = 12


def _not_covered(reason: str) -> ValidatedAnswer:
    return ValidatedAnswer("not_covered", "", [], "", reason)


def validate_llm_output(raw: str, excerpts: dict[str, str], other_company_aliases: list[str],
                        max_citations: int) -> ValidatedAnswer:
    """Check the model's reply against what we actually sent it.

    excerpts: label -> excerpt text exactly as sent (e.g. {"C1": "..."}).
    Raises MalformedOutput for unparseable replies; every other problem
    becomes not_covered with a reason.
    """
    reply = parse_llm_reply(raw)
    if reply.status == "not_covered":
        return _not_covered("model said not covered")

    answer = reply.answer.strip()
    if not answer:
        return _not_covered("answered with empty text")

    cited = [c for c in dict.fromkeys(label.strip() for label in reply.citations) if c in excerpts]
    cited = cited[:max_citations]
    if not cited:
        return _not_covered(f"no valid citation (model cited {reply.citations!r})")

    quote = normalise(reply.quote)
    if len(quote) < MIN_QUOTE_CHARS:
        return _not_covered("quote missing or too short")
    if not any(quote in normalise(excerpts[label]) for label in cited):
        return _not_covered("quote not found in cited excerpts")

    leaked = mentioned_company(answer, other_company_aliases)
    if leaked:
        return _not_covered(f"answer mentions another company ({leaked})")

    return ValidatedAnswer("answered", answer, cited, reply.quote.strip(), "ok")
