"""Versioning and atomicity: once v2 is active only v2 exists for readers,
and a failed ingestion never leaves a half-updated policy."""

import pytest

from app.ingestion import IngestionConflict, IngestionError, IngestionFailed, ingest_policy_version, startup_cleanup
from app.vectorstore import VectorStore
from tests.conftest import read_policy

NIMBUS_V1 = read_policy("nimbus/transport-policy.v1.md")
NIMBUS_V2 = read_policy("nimbus/transport-policy.v2.md")
TITLE = "Nimbus Tech Employee Transport Policy"


def ingest(conn, store, settings, content, company="nimbus", slug="transport-policy"):
    return ingest_policy_version(conn, store, settings, company, slug, TITLE, content)


def versions(conn, company="nimbus"):
    return {
        r["version_no"]: dict(r)
        for r in conn.execute(
            """SELECT v.id, v.version_no, v.status, v.chunk_count, v.error FROM policy_versions v
               JOIN policy_documents d ON d.id = v.document_id WHERE d.company_id = ?""",
            (company,),
        )
    }


def chunk_rows(conn, version_id):
    return conn.execute("SELECT COUNT(*) AS n FROM chunks WHERE version_id = ?", (version_id,)).fetchone()["n"]


class FlakyStore(VectorStore):
    """Real store that raises on the Nth add() call, after earlier batches were really written."""

    def __init__(self, path, fail_on_call: int):
        super().__init__(path)
        self.fail_on_call = fail_on_call
        self.calls = 0

    def add(self, *args, **kwargs):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise ConnectionError("simulated vector store failure")
        super().add(*args, **kwargs)


# --- happy path ------------------------------------------------------------------

def test_first_version_becomes_active(conn, store, settings_env):
    result = ingest(conn, store, settings_env, NIMBUS_V1)
    assert (result.version_no, result.status) == (1, "active")
    assert result.chunk_count == 17
    assert store.count_version("nimbus", result.version_id) == 17
    assert chunk_rows(conn, result.version_id) == 17


def test_new_version_supersedes_old_one(conn, store, settings_env):
    v1 = ingest(conn, store, settings_env, NIMBUS_V1)
    v2 = ingest(conn, store, settings_env, NIMBUS_V2)

    vs = versions(conn)
    assert vs[1]["status"] == "superseded"
    assert vs[2]["status"] == "active"
    assert [v for v in vs.values() if v["status"] == "active"] == [vs[2]]
    assert store.count_version("nimbus", v2.version_id) == 17

    # Superseded vectors are kept until cleanup, then removed.
    assert store.count_version("nimbus", v1.version_id) == 17
    startup_cleanup(conn, store)
    assert store.count_version("nimbus", v1.version_id) == 0
    assert store.count_version("nimbus", v2.version_id) == 17


def test_vectors_go_only_to_the_owning_company_collection(conn, store, settings_env):
    result = ingest(conn, store, settings_env, NIMBUS_V1)
    assert store.count_version("orion", result.version_id) == 0
    assert store.count_version("kestrel", result.version_id) == 0


# --- conflicts and invalid input: nothing written ---------------------------------

def test_identical_content_is_rejected(conn, store, settings_env):
    ingest(conn, store, settings_env, NIMBUS_V1)
    with pytest.raises(IngestionConflict, match="identical"):
        ingest(conn, store, settings_env, NIMBUS_V1)
    assert list(versions(conn)) == [1]


def test_concurrent_ingestion_is_rejected(conn, store, settings_env):
    ingest(conn, store, settings_env, NIMBUS_V1)
    conn.execute("UPDATE policy_versions SET status = 'staging' WHERE version_no = 1")  # simulate in-flight
    with pytest.raises(IngestionConflict, match="being ingested"):
        ingest(conn, store, settings_env, NIMBUS_V2)


def test_invalid_document_writes_nothing(conn, store, settings_env):
    with pytest.raises(IngestionError):
        ingest(conn, store, settings_env, "no headings at all")
    assert versions(conn) == {}


def test_oversized_document_is_rejected(conn, store, settings_env, monkeypatch):
    monkeypatch.setattr(settings_env, "max_policy_bytes", 100)
    with pytest.raises(IngestionError, match="larger than"):
        ingest(conn, store, settings_env, NIMBUS_V1)
    assert versions(conn) == {}


# --- failure halfway: rollback ------------------------------------------------------

def test_failure_midway_keeps_old_version_and_leaves_no_trace(conn, store, settings_env):
    v1 = ingest(conn, store, settings_env, NIMBUS_V1)

    flaky = FlakyStore(settings_env.chroma_path, fail_on_call=2)  # batch 1 written, batch 2 fails
    with pytest.raises(IngestionFailed, match="previous version is still active"):
        ingest(conn, flaky, settings_env, NIMBUS_V2)

    vs = versions(conn)
    assert vs[1]["status"] == "active"  # v1 still answers
    assert vs[2]["status"] == "failed"
    assert "simulated vector store failure" in vs[2]["error"]
    assert store.count_version("nimbus", vs[2]["id"]) == 0  # the written batch was rolled back
    assert chunk_rows(conn, vs[2]["id"]) == 0
    assert store.count_version("nimbus", v1.version_id) == 17

    # The failure does not block a later retry.
    retry = ingest(conn, store, settings_env, NIMBUS_V2)
    assert (retry.version_no, retry.status) == (3, "active")


def test_failure_at_the_commit_point_rolls_back_sqlite_too(conn, store, settings_env):
    ingest(conn, store, settings_env, NIMBUS_V1)
    # Make the LAST statement of the activation transaction fail, after the chunk
    # rows were inserted and v1 was set to 'superseded' in that same transaction.
    conn.execute(
        """CREATE TRIGGER fail_activation BEFORE UPDATE OF status ON policy_versions
           WHEN NEW.status = 'active' BEGIN SELECT RAISE(ABORT, 'simulated failure at commit'); END"""
    )
    with pytest.raises(IngestionFailed):
        ingest(conn, store, settings_env, NIMBUS_V2)

    vs = versions(conn)
    assert vs[1]["status"] == "active"  # the 'superseded' update was rolled back
    assert vs[2]["status"] == "failed"
    assert chunk_rows(conn, vs[2]["id"]) == 0  # the chunk inserts were rolled back
    assert store.count_version("nimbus", vs[2]["id"]) == 0


def test_startup_recovers_an_interrupted_ingestion(conn, store, settings_env):
    v1 = ingest(conn, store, settings_env, NIMBUS_V1)
    # Simulate a process crash after vectors were written but before activation.
    conn.execute(
        "INSERT INTO policy_versions (document_id, version_no, status, content_sha256, created_at) "
        "SELECT document_id, 2, 'staging', 'x', 'now' FROM policy_versions WHERE id = ?",
        (v1.version_id,),
    )
    orphan_id = versions(conn)[2]["id"]
    store.add("nimbus", ids=[f"{orphan_id}:0"], documents=["orphan"],
              metadatas=[{"company_id": "nimbus", "version_id": orphan_id}])

    report = startup_cleanup(conn, store)

    assert report["recovered_staging"] == 1
    assert versions(conn)[2]["status"] == "failed"
    assert store.count_version("nimbus", orphan_id) == 0
    assert versions(conn)[1]["status"] == "active"
