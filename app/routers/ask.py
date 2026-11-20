import sqlite3
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app import answer
from app.auth import CurrentUser, get_current_user
from app.config import Settings, get_settings
from app.deps import get_db, get_llm, get_vectorstore
from app.llm import LLMClient
from app.schemas import AskRequest, AskResponse
from app.vectorstore import VectorStore

router = APIRouter(tags=["ask"])


@router.post(
    "/ask",
    response_model=AskResponse,
    responses={503: {"model": AskResponse, "description": "Temporary failure: try again"}},
)
def ask(
    body: AskRequest,
    user: CurrentUser = Depends(get_current_user),  # company comes from here, nowhere else
    conn: sqlite3.Connection = Depends(get_db),
    store: VectorStore = Depends(get_vectorstore),
    llm: LLMClient = Depends(get_llm),
    settings: Settings = Depends(get_settings),
):
    result = answer.ask(conn, store, llm, settings, user, body.question, request_id=str(uuid.uuid4()))
    if result.status == "try_again":
        return JSONResponse(status_code=503, content=result.model_dump(), headers={"Retry-After": "5"})
    return result
