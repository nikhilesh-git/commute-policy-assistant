"""Admin endpoints. The company is always the admin's own (from the JWT)."""

import sqlite3

from fastapi import APIRouter, Depends, Path

from app.auth import CurrentUser, require_admin
from app.config import Settings, get_settings
from app.deps import get_db, get_vectorstore
from app.ingestion import ingest_policy_version
from app.schemas import IngestionResult, PolicyUploadRequest, PolicyVersionInfo
from app.vectorstore import VectorStore

router = APIRouter(prefix="/admin", tags=["admin"])

SLUG = Path(pattern=r"^[a-z0-9][a-z0-9-]{1,60}$")


@router.post("/policies/{slug}/versions", response_model=IngestionResult, status_code=201)
def upload_policy_version(
    body: PolicyUploadRequest,
    slug: str = SLUG,
    admin: CurrentUser = Depends(require_admin),
    conn: sqlite3.Connection = Depends(get_db),
    store: VectorStore = Depends(get_vectorstore),
    settings: Settings = Depends(get_settings),
) -> IngestionResult:
    return ingest_policy_version(conn, store, settings, admin.company_id, slug, body.title, body.content)


@router.get("/policies", response_model=list[PolicyVersionInfo])
def list_policy_versions(
    admin: CurrentUser = Depends(require_admin),
    conn: sqlite3.Connection = Depends(get_db),
) -> list[PolicyVersionInfo]:
    rows = conn.execute(
        """SELECT d.title AS document, d.slug, v.version_no, v.status, v.chunk_count,
                  v.created_at, v.activated_at
           FROM policy_versions v JOIN policy_documents d ON d.id = v.document_id
           WHERE d.company_id = ?
           ORDER BY d.slug, v.version_no""",
        (admin.company_id,),
    ).fetchall()
    return [PolicyVersionInfo(**dict(r)) for r in rows]
