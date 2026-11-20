"""Timing helpers and the query log used for p95 latency and LLM-call counts."""

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class RequestTimings:
    """Milliseconds per pipeline stage for one request."""

    stages: dict[str, float] = field(default_factory=dict)
    _start: float = field(default_factory=time.perf_counter)

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - t0) * 1000

    def elapsed_s(self) -> float:
        return time.perf_counter() - self._start

    def total_ms(self) -> float:
        return self.elapsed_s() * 1000


def log_query(conn, *, request_id: str, user_id: int, company_id: str, question: str, outcome: str,
              llm_calls: int, top_score: float | None, cited_chunk_ids: list[str],
              timings: RequestTimings) -> None:
    """One row per /ask request. The eval runner reads llm_calls and timings from here."""
    conn.execute(
        """INSERT INTO query_log (ts, request_id, user_id, company_id, question, outcome, llm_calls,
                                  top_score, cited_chunk_ids, retrieval_ms, llm_ms, total_ms)
           VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (request_id, user_id, company_id, question, outcome, llm_calls, top_score,
         ",".join(cited_chunk_ids), timings.stages.get("retrieval"), timings.stages.get("llm"),
         timings.total_ms()),
    )


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100). Simple and easy to explain."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]
