"""The question-answering pipeline. Orchestration only; each step lives in its own module.

ask():
  1. guards.clean_question                      bad input -> QuestionRejected (422)
  2. question names another company?            -> refusal, 0 LLM calls
  3. retrieval.retrieve (company from the JWT)  store down/slow -> TRY_AGAIN
                                                no evidence     -> NOT_COVERED, 0 LLM calls
  4. prompts.build_messages (<= 3 excerpts, labelled C1..C3)
  5. LLM call through LLMBudget (max 2 calls), inside the request deadline
       timeout / provider error / malformed JSON  -> one retry if time is left
       rate limited (429)                          -> no retry
       nothing usable                              -> TRY_AGAIN
  6. guards.validate_llm_output                  unsafe/unsupported -> NOT_COVERED
  7. citations built from chunk METADATA (title, version, section), never from model text
  8. one query_log row (outcome, LLM calls, timings)

Every business rule above is Python. The model only ever chooses words and
which of the excerpts we gave it to cite.
"""

import logging
import sqlite3

from app import guards, prompts
from app.auth import CurrentUser
from app.config import Settings
from app.guards import MalformedOutput, ValidatedAnswer
from app.llm import LLMBudget, LLMClient, LLMRateLimited, LLMTimeout, LLMUnavailable
from app.metrics import RequestTimings, log_query
from app.retrieval import RetrievalUnavailable, RetrievedChunk, retrieve
from app.schemas import AskResponse, Citation
from app.vectorstore import VectorStore

log = logging.getLogger(__name__)

NOT_COVERED_MESSAGE = (
    "Your company's transport policy does not cover this. "
    "Please contact your transport desk for help."
)
TRY_AGAIN_MESSAGE = "We couldn't get an answer right now. Please try again in a moment."
OTHER_COMPANY_MESSAGE = (
    "I can only answer questions about your own company's transport policy. "
    "For anything else, please contact your transport desk."
)


def other_company_aliases(conn: sqlite3.Connection, company_id: str) -> list[str]:
    """Names that identify other tenants: id, full name, and first word of the name."""
    aliases: list[str] = []
    for row in conn.execute("SELECT id, name FROM companies WHERE id != ?", (company_id,)):
        for alias in (row["name"], row["name"].split()[0], row["id"]):
            if len(alias) >= 3 and alias not in aliases:
                aliases.append(alias)
    return aliases


def _call_llm(budget: LLMBudget, messages: list[dict], excerpts: dict[str, str], aliases: list[str],
              settings: Settings, timings: RequestTimings) -> ValidatedAnswer | None:
    """Step 5-6. Returns a validated answer, or None if no usable reply arrived in time."""
    attempt_messages = messages
    while budget.calls < budget.max_calls:
        remaining = settings.request_deadline_s - timings.elapsed_s()
        if remaining <= 0 or (budget.calls > 0 and remaining < settings.min_time_for_retry_s):
            log.info("no time left for another LLM call (%.2fs remaining)", remaining)
            return None
        try:
            with timings.stage("llm"):
                raw = budget.complete(attempt_messages, timeout_s=remaining)
            return guards.validate_llm_output(raw, excerpts, aliases, settings.max_citations)
        except LLMRateLimited:
            log.warning("LLM rate limited; not retrying")
            return None
        except (LLMTimeout, LLMUnavailable) as exc:
            log.warning("LLM call %d failed: %s", budget.calls, exc)
        except MalformedOutput as exc:
            log.warning("LLM call %d returned malformed output: %s", budget.calls, exc)
            attempt_messages = messages + [{"role": "user", "content": prompts.JSON_REMINDER}]
    return None


def _citations(validated: ValidatedAnswer, labels: dict[str, RetrievedChunk]) -> list[Citation]:
    quote_norm = guards.normalise(validated.quote)
    out = []
    for label in validated.cited_labels:
        chunk = labels[label]
        out.append(Citation(
            document=chunk.doc_title,
            version=chunk.version_no,
            section=f"{chunk.section_id} {chunk.section_title}",
            chunk_id=chunk.chunk_id,
            quote=validated.quote if quote_norm in guards.normalise(chunk.text) else None,
        ))
    return out


def ask(
    conn: sqlite3.Connection,
    store: VectorStore,
    llm: LLMClient,
    settings: Settings,
    user: CurrentUser,
    question: str,
    request_id: str,
) -> AskResponse:
    timings = RequestTimings()
    budget = LLMBudget(llm, settings.max_llm_calls)
    top_score: float | None = None

    def finish(status: str, answer: str, citations: list[Citation] | None = None, reason: str = "") -> AskResponse:
        citations = citations or []
        log.info("ask %s company=%s outcome=%s llm_calls=%d reason=%s total_ms=%.0f",
                 request_id, user.company_id, status, budget.calls, reason, timings.total_ms())
        log_query(conn, request_id=request_id, user_id=user.user_id, company_id=user.company_id,
                  question=question, outcome=status, llm_calls=budget.calls, top_score=top_score,
                  cited_chunk_ids=[c.chunk_id for c in citations], timings=timings)
        return AskResponse(status=status, answer=answer, citations=citations, request_id=request_id)

    # 1-2. input checks (QuestionRejected propagates -> 422)
    question = guards.clean_question(question, settings.max_question_chars)
    aliases = other_company_aliases(conn, user.company_id)
    named = guards.mentioned_company(question, aliases)
    if named:
        return finish("not_covered", OTHER_COMPANY_MESSAGE, reason=f"question names another company ({named})")

    # 3. retrieval
    try:
        with timings.stage("retrieval"):
            result = retrieve(conn, store, settings, user.company_id, question)
    except RetrievalUnavailable as exc:
        log.warning("retrieval unavailable: %s", exc)
        return finish("try_again", TRY_AGAIN_MESSAGE, reason="retrieval unavailable")
    top_score = result.top_similarity
    if not result.chunks:
        return finish("not_covered", NOT_COVERED_MESSAGE, reason="no chunk above relevance threshold")

    # 4-6. prompt, LLM, validation
    messages, labels = prompts.build_messages(question, result.chunks)
    excerpts = {label: prompts.escape(chunk.text) for label, chunk in labels.items()}
    try:
        validated = _call_llm(budget, messages, excerpts, aliases, settings, timings)
    except Exception:
        # A bug or an unexpected provider error must still give the user a clear reply.
        log.exception("unexpected error during LLM call")
        validated = None

    if validated is None:
        return finish("try_again", TRY_AGAIN_MESSAGE, reason="no usable LLM reply")
    if validated.status == "not_covered":
        return finish("not_covered", NOT_COVERED_MESSAGE, reason=validated.reason)

    # 7. citations from metadata
    return finish("answered", validated.answer, _citations(validated, labels), reason="ok")
