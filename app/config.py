"""All configuration in one place.

Every tunable number the viva might ask you to change (threshold, top-k,
citation limit, timeouts) lives here and can be overridden by an environment
variable of the same name (case-insensitive), e.g. MAX_CITATIONS=2.
"""

import logging
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)

_PLACEHOLDER_PREFIX = "change-me"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- runtime ---------------------------------------------------------
    app_env: Literal["dev", "test", "prod"] = "dev"
    data_dir: Path = Path("data/runtime")  # SQLite file + Chroma folder live here
    seed_on_startup: bool = True  # demo companies and users
    seed_policies_on_startup: bool = True  # ingest every policies/<company>/*.v1.md not yet ingested
    policies_dir: Path = Path("policies")

    # --- auth ------------------------------------------------------------
    # Empty => a random secret is generated at startup (tokens die on restart).
    # This keeps `docker compose up` working without committing any secret.
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_issuer: str = "commute-assistant"
    jwt_audience: str = "commute-assistant-api"
    jwt_ttl_minutes: int = 30

    # --- LLM ---------------------------------------------------------------
    llm_provider: Literal["fake", "groq"] = "fake"
    groq_api_key: str = ""
    # Chosen by measurement on our own prompts (Nov 2026): fastest of the available
    # models (300-510 ms) and correct on all 6 benchmark questions. Alternatives:
    # openai/gpt-oss-20b or openai/gpt-oss-120b with GROQ_REASONING_EFFORT=low.
    groq_model: str = "qwen/qwen3.8-27b"
    groq_reasoning_effort: str = "none"  # qwen: none|default; gpt-oss: low|medium|high; "" = don't send
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # --- business limits (enforced in Python, never by the LLM) -----------
    max_llm_calls: int = 2
    max_citations: int = 3
    max_question_chars: int = 500
    max_policy_bytes: int = 200_000

    # --- retrieval ---------------------------------------------------------
    # hybrid = embeddings + BM25 keywords (RRF). On the eval set: 39/40 vs 38/40 for
    # dense, hit@1 0.83 -> 0.90, retrieval cost +2-6 ms p95. See EVALUATION.md.
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"
    retrieval_candidates: int = 8
    # Question-level gate: below this best-chunk similarity, reply "not covered"
    # without calling the LLM. Chosen with `python -m evaluation.threshold_sweep`:
    #   0.20 -> 0 answers lost, 2/8 unanswerable stopped early
    #   0.25 -> 0 answers lost, 4/8 stopped early   <- chosen
    #   0.30 -> 1 answer lost,  5/8;  0.35 -> 4 answers lost, 7/8
    # Lowest answerable question scores 0.294; 0.25 keeps a margin instead of fitting
    # this eval set exactly. A wrongly gated question gets no answer at all, while an
    # ungated unanswerable one costs one LLM call that can still say "not covered".
    min_similarity: float = 0.25

    # --- latency budget (seconds) -----------------------------------------
    request_deadline_s: float = 2.7
    retrieval_timeout_s: float = 1.0  # typical warm retrieval is ~10 ms; this only catches a stuck store
    min_time_for_retry_s: float = 1.0

    # --- derived paths ------------------------------------------------------
    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / "chroma"

    @field_validator("jwt_secret")
    @classmethod
    def _reject_placeholder_secret(cls, v: str) -> str:
        if v.startswith(_PLACEHOLDER_PREFIX):
            raise ValueError("JWT_SECRET is still the placeholder from .env.example; set a real value or leave it empty")
        if v and len(v) < 32:
            raise ValueError("JWT_SECRET must be at least 32 characters")
        return v

    def effective_jwt_secret(self) -> str:
        return self.jwt_secret or _ephemeral_secret()


@lru_cache
def _ephemeral_secret() -> str:
    log.warning("JWT_SECRET not set: using a random per-process secret (tokens are invalid after restart)")
    return secrets.token_urlsafe(48)


@lru_cache
def get_settings() -> Settings:
    return Settings()
