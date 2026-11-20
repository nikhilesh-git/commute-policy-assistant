"""FastAPI entry point: builds the app and wires long-lived objects at startup."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.db import connect, init_db
from app.guards import QuestionRejected
from app.ingestion import IngestionConflict, IngestionError, IngestionFailed, startup_cleanup
from app.llm import build_llm_client
from app.routers import admin, ask, health, token
from app.seed import seed, seed_policies
from app.vectorstore import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.effective_jwt_secret()  # fail fast / warn about the secret at startup

    init_db(settings.sqlite_path)
    app.state.vectorstore = VectorStore(settings.chroma_path)
    app.state.llm = build_llm_client(settings)

    conn = connect(settings.sqlite_path)
    try:
        cleanup = startup_cleanup(conn, app.state.vectorstore)
        log.info("startup cleanup: %s", cleanup)
        if settings.seed_on_startup:
            seed(conn)
        if settings.seed_policies_on_startup:
            try:
                ingested = seed_policies(conn, app.state.vectorstore, settings)
                log.info("seeded policies: %s", ingested or "none needed")
            except IngestionError:
                # Keep serving: /health shows active_policies, so the problem is visible.
                log.exception("policy seeding failed")
    finally:
        conn.close()

    try:
        app.state.vectorstore.warm_up()  # load the embedding model now, not on the first question
    except Exception:
        log.exception("embedding warm-up failed; the first question will be slower")

    log.info("ready: env=%s llm=%s retrieval=%s", settings.app_env, app.state.llm.name, settings.retrieval_mode)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Commute Policy Assistant", version="0.1.0", lifespan=lifespan)
    for r in (health.router, token.router, ask.router, admin.router):
        app.include_router(r)

    @app.exception_handler(QuestionRejected)
    async def _bad_question(_: Request, exc: QuestionRejected):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(IngestionFailed)
    async def _ingestion_rolled_back(_: Request, exc: IngestionFailed):
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.exception_handler(IngestionConflict)
    async def _conflict(_: Request, exc: IngestionConflict):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(IngestionError)
    async def _ingestion_failed(_: Request, exc: IngestionError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    # Scaffold only: stubs raise NotImplementedError until their milestone lands.
    @app.exception_handler(NotImplementedError)
    async def _not_implemented(_: Request, exc: NotImplementedError):
        return JSONResponse(status_code=501, content={"detail": f"not implemented yet: {exc}"})

    return app


app = create_app()
