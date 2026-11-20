"""Retrieval, restricted to one company and its active policy version(s).

Isolation layers:
  1. collection = policy_<company_id>, where company_id comes from the verified JWT
  2. Chroma filter: version_id IN the active versions read from SQLite
     (the keyword index is built from SQLite rows with the same company + version filter)
  3. post-check: every candidate must match the company and an active version,
     in both its Chroma metadata and its SQLite row; anything else is dropped
     and logged. This should never fire. It exists so a bug in 1-2 cannot leak.

Ranking:
  dense   embedding similarity only
  hybrid  dense + BM25 keyword ranking, merged with Reciprocal Rank Fusion

Relevance gate (question level): if even the best chunk's embedding similarity
is below settings.min_similarity, the question is treated as off-topic and the
caller replies "not covered" without calling the LLM. Otherwise the top
settings.max_citations chunks by ranking go to the LLM, so it can never see
(and therefore never cite) more than that.
"""

import logging
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, replace

from app.config import Settings
from app.keyword_search import BM25Index
from app.vectorstore import VectorStore

log = logging.getLogger(__name__)

RRF_K = 60  # standard constant from the RRF paper; dampens the weight of top ranks

# Chroma calls are synchronous; running them in a pool lets us stop waiting
# after retrieval_timeout_s and tell the user to try again instead of hanging.
_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="retrieval")


class RetrievalUnavailable(Exception):
    """The vector store failed or was too slow. The user gets 'try again'."""


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    company_id: str
    version_id: int
    version_no: int
    doc_title: str
    section_id: str
    section_title: str
    text: str  # sanitised text from SQLite, the same text the LLM will see
    similarity: float | None  # embedding similarity (1 - cosine distance); None if found only by keywords


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]  # sent to the LLM: best first, at most max_citations; [] = no evidence
    candidates: list[RetrievedChunk]  # every isolation-checked candidate in ranked order (diagnostics)
    dropped: int  # candidates removed by the post-check (expected: always 0)
    top_similarity: float | None  # best embedding similarity; what the relevance gate looks at


def active_version_ids(conn: sqlite3.Connection, company_id: str) -> list[int]:
    rows = conn.execute(
        """SELECT v.id FROM policy_versions v
           JOIN policy_documents d ON d.id = v.document_id
           WHERE d.company_id = ? AND v.status = 'active'""",
        (company_id,),
    ).fetchall()
    return [r["id"] for r in rows]


def _query_with_timeout(store: VectorStore, company_id: str, question: str, n: int,
                        version_ids: list[int], timeout_s: float) -> list[dict]:
    future = _pool.submit(store.query, company_id, question, n, version_ids)
    try:
        return future.result(timeout=timeout_s)
    except FutureTimeout:
        future.cancel()
        raise RetrievalUnavailable(f"vector search took longer than {timeout_s}s")
    except Exception as exc:
        raise RetrievalUnavailable(f"vector search failed: {type(exc).__name__}: {str(exc)[:300]}") from exc


def _active_chunks(conn: sqlite3.Connection, company_id: str, active: list[int]) -> dict[str, RetrievedChunk]:
    """All chunks of the company's active version(s), from SQLite (the source of truth)."""
    marks = ",".join("?" * len(active))
    rows = conn.execute(
        f"""SELECT c.id, c.company_id, c.version_id, c.section_id, c.section_title, c.text,
                   v.version_no, d.title AS doc_title
            FROM chunks c
            JOIN policy_versions v ON v.id = c.version_id
            JOIN policy_documents d ON d.id = v.document_id
            WHERE c.company_id = ? AND c.version_id IN ({marks})
            ORDER BY c.id""",
        (company_id, *active),
    )
    return {
        r["id"]: RetrievedChunk(r["id"], r["company_id"], r["version_id"], r["version_no"], r["doc_title"],
                                r["section_id"], r["section_title"], r["text"], similarity=None)
        for r in rows
    }


# Keyword indexes are rebuilt only when the active versions change: the cache key
# contains the version ids, so publishing a new version invalidates it by itself.
_bm25_cache: dict[tuple[str, tuple[int, ...]], tuple[BM25Index, list[str]]] = {}
_bm25_lock = threading.Lock()


def _keyword_ranking(chunks: dict[str, RetrievedChunk], company_id: str, active: list[int],
                     question: str, n: int) -> list[str]:
    key = (company_id, tuple(sorted(active)))
    with _bm25_lock:
        if key not in _bm25_cache:
            for old in [k for k in _bm25_cache if k[0] == company_id]:
                del _bm25_cache[old]
            ids = list(chunks)
            index = BM25Index([f"{chunks[i].section_title}\n{chunks[i].text}" for i in ids])
            _bm25_cache[key] = (index, ids)
        index, ids = _bm25_cache[key]
    return [ids[i] for i in index.top(question, n)]


def _rrf(rankings: list[list[str]]) -> list[str]:
    """Reciprocal Rank Fusion: score(id) = sum of 1 / (RRF_K + rank) over the rankings."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores, key=lambda cid: (-scores[cid], cid))


def retrieve(
    conn: sqlite3.Connection,
    store: VectorStore,
    settings: Settings,
    company_id: str,
    question: str,
) -> RetrievalResult:
    """Search the company's active policy. An empty `chunks` list means 'no evidence'."""
    active = active_version_ids(conn, company_id)
    if not active:
        return RetrievalResult(chunks=[], candidates=[], dropped=0, top_similarity=None)

    hits = _query_with_timeout(store, company_id, question, settings.retrieval_candidates,
                               active, settings.retrieval_timeout_s)
    rows = _active_chunks(conn, company_id, active)

    # Post-check: a dense hit must match both its metadata and an active SQLite row.
    dense: list[RetrievedChunk] = []
    dropped = 0
    for hit in hits:
        meta = hit["metadata"] or {}
        row = rows.get(hit["id"])
        if (row is None or meta.get("company_id") != company_id
                or meta.get("version_id") != row.version_id or row.version_id not in active):
            dropped += 1
            log.warning("SECURITY: dropped chunk %s for company %s (metadata=%s)", hit["id"], company_id, meta)
            continue
        dense.append(replace(row, similarity=1.0 - hit["distance"]))
    dense.sort(key=lambda c: c.similarity, reverse=True)
    top_similarity = dense[0].similarity if dense else None

    if settings.retrieval_mode == "hybrid":
        by_id = {**rows, **{c.chunk_id: c for c in dense}}  # keep similarities where known
        keyword = _keyword_ranking(rows, company_id, active, question, settings.retrieval_candidates)
        ranked = [by_id[cid] for cid in _rrf([[c.chunk_id for c in dense], keyword])]
    else:
        ranked = dense

    on_topic = top_similarity is not None and top_similarity >= settings.min_similarity
    chunks = ranked[: settings.max_citations] if on_topic else []
    return RetrievalResult(chunks=chunks, candidates=ranked, dropped=dropped, top_similarity=top_similarity)
