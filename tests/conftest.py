"""Shared fixtures. Every test gets a fresh, temporary SQLite + Chroma and a FakeLLM.

No test touches the network or needs an API key. The embedding model is the
real local ONNX model (deterministic, free), so retrieval tests are real.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.db import connect, init_db
from app.llm import FakeLLM
from app.seed import DEMO_PASSWORD, seed
from app.vectorstore import VectorStore

TEST_SECRET = "test-secret-" + "x" * 40
POLICIES = Path(__file__).resolve().parent.parent / "policies"


def read_policy(relative: str) -> str:
    return (POLICIES / relative).read_text(encoding="utf-8")


@pytest.fixture
def settings_env(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("JWT_SECRET", TEST_SECRET)
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("POLICIES_DIR", str(POLICIES))
    # Pin everything a developer's local .env might set, so tests behave the same everywhere.
    monkeypatch.setenv("RETRIEVAL_MODE", "hybrid")
    monkeypatch.setenv("GROQ_MODEL", "test-model")
    # Off by default so most tests stay fast; tests that need policies ingest them explicitly.
    monkeypatch.setenv("SEED_POLICIES_ON_STARTUP", "false")
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


# --- service-level fixtures (no HTTP) ---------------------------------------

@pytest.fixture
def conn(settings_env):
    init_db(settings_env.sqlite_path)
    c = connect(settings_env.sqlite_path)
    seed(c)
    yield c
    c.close()


@pytest.fixture
def store(settings_env):
    return VectorStore(settings_env.chroma_path)


# --- HTTP-level fixtures ---------------------------------------------------------

@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def client(settings_env, fake_llm):
    from app.main import create_app

    app = create_app()
    with TestClient(app) as c:
        app.state.llm = fake_llm  # replace whatever lifespan built
        yield c


@pytest.fixture
def seeded_client(settings_env, fake_llm, monkeypatch):
    """Like `client`, but all three companies' v1 policies are ingested at startup."""
    from app.main import create_app

    monkeypatch.setenv("SEED_POLICIES_ON_STARTUP", "true")
    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as c:
        app.state.llm = fake_llm
        yield c


def make_login(client):
    def _login(username: str) -> dict[str, str]:
        r = client.post("/auth/token", json={"username": username, "password": DEMO_PASSWORD})
        assert r.status_code == 200, r.text
        return {"Authorization": f"Bearer {r.json()['access_token']}"}

    return _login


@pytest.fixture
def login(client):
    """login("alice@nimbus") -> {"Authorization": "Bearer ..."}"""

    def _login(username: str) -> dict[str, str]:
        r = client.post("/auth/token", json={"username": username, "password": DEMO_PASSWORD})
        assert r.status_code == 200, r.text
        return {"Authorization": f"Bearer {r.json()['access_token']}"}

    return _login
