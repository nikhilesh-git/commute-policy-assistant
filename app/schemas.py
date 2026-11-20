"""Request/response models.

`extra="forbid"` on request bodies: a client that sends `company_id` (or any
unknown field) gets a 422 instead of being silently ignored.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- auth -------------------------------------------------------------------

class TokenRequest(StrictModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


# --- ask ----------------------------------------------------------------------

class AskRequest(StrictModel):
    # Length is re-checked in guards.py against settings.max_question_chars.
    question: str = Field(min_length=1, max_length=2000)


class Citation(BaseModel):
    document: str
    version: int
    section: str
    chunk_id: str
    quote: str | None = None


AnswerStatus = Literal["answered", "not_covered", "try_again"]


class AskResponse(BaseModel):
    status: AnswerStatus
    answer: str
    citations: list[Citation] = []
    request_id: str


# --- admin ---------------------------------------------------------------------

class PolicyUploadRequest(StrictModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1)


class IngestionResult(BaseModel):
    version_id: int
    version_no: int
    status: str
    chunk_count: int
    warnings: list[str] = []


class PolicyVersionInfo(BaseModel):
    document: str
    slug: str
    version_no: int
    status: str
    chunk_count: int | None
    created_at: str
    activated_at: str | None
