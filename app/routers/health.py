import sqlite3

from fastapi import APIRouter, Depends, Request

from app.deps import get_db

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request, conn: sqlite3.Connection = Depends(get_db)) -> dict:
    active = conn.execute("SELECT COUNT(*) AS n FROM policy_versions WHERE status = 'active'").fetchone()["n"]
    return {
        "status": "ok",
        "database": True,
        "vectorstore": request.app.state.vectorstore.is_alive(),
        "llm_provider": request.app.state.llm.name,
        "active_policies": active,
    }
