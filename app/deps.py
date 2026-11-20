"""Request-scoped dependencies: how routes get the DB, vector store and LLM.

Long-lived objects are created once in main.lifespan and stored on app.state.
Tests swap them (e.g. a FakeLLM) by setting app.state before making requests.
"""

import sqlite3
from collections.abc import Iterator

from fastapi import Depends, Request

from app.config import Settings, get_settings
from app.db import connect
from app.llm import LLMClient
from app.vectorstore import VectorStore


def get_db(settings: Settings = Depends(get_settings)) -> Iterator[sqlite3.Connection]:
    conn = connect(settings.sqlite_path)
    try:
        yield conn
    finally:
        conn.close()


def get_vectorstore(request: Request) -> VectorStore:
    return request.app.state.vectorstore


def get_llm(request: Request) -> LLMClient:
    return request.app.state.llm
