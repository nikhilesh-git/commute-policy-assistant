"""SQLite: the source of truth (users, policy versions, chunks, query log).

Chroma is only an index. "Which policy version is live" is decided here,
because SQLite has transactions and Chroma does not.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
  id   TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS users (
  id            INTEGER PRIMARY KEY,
  username      TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  company_id    TEXT NOT NULL REFERENCES companies(id),
  role          TEXT NOT NULL CHECK (role IN ('employee', 'admin'))
);

CREATE TABLE IF NOT EXISTS policy_documents (
  id         INTEGER PRIMARY KEY,
  company_id TEXT NOT NULL REFERENCES companies(id),
  slug       TEXT NOT NULL,
  title      TEXT NOT NULL,
  UNIQUE (company_id, slug)
);

CREATE TABLE IF NOT EXISTS policy_versions (
  id             INTEGER PRIMARY KEY,
  document_id    INTEGER NOT NULL REFERENCES policy_documents(id),
  version_no     INTEGER NOT NULL,
  status         TEXT NOT NULL CHECK (status IN ('staging', 'active', 'superseded', 'failed')),
  content_sha256 TEXT NOT NULL,
  chunk_count    INTEGER,
  error          TEXT,
  created_at     TEXT NOT NULL,
  activated_at   TEXT,
  UNIQUE (document_id, version_no)
);

-- The database itself guarantees at most one active version per document.
CREATE UNIQUE INDEX IF NOT EXISTS one_active_version
  ON policy_versions(document_id) WHERE status = 'active';

CREATE TABLE IF NOT EXISTS chunks (
  id            TEXT PRIMARY KEY,          -- '<version_id>:<ordinal>', same id in Chroma
  version_id    INTEGER NOT NULL REFERENCES policy_versions(id),
  company_id    TEXT NOT NULL,             -- denormalised for the isolation post-check
  section_id    TEXT NOT NULL,
  section_title TEXT NOT NULL,
  ordinal       INTEGER NOT NULL,
  text          TEXT NOT NULL,
  flagged       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS chunks_by_version ON chunks(version_id);

CREATE TABLE IF NOT EXISTS query_log (
  id              INTEGER PRIMARY KEY,
  ts              TEXT NOT NULL,
  request_id      TEXT NOT NULL,
  user_id         INTEGER,
  company_id      TEXT,
  question        TEXT,
  outcome         TEXT,                    -- answered | not_covered | try_again | rejected
  llm_calls       INTEGER,
  top_score       REAL,
  cited_chunk_ids TEXT,
  retrieval_ms    REAL,
  llm_ms          REAL,
  total_ms        REAL
);
"""


def connect(path: Path) -> sqlite3.Connection:
    """Open a connection. One connection per request; WAL lets readers run during ingestion.

    isolation_level=None means autocommit: every statement commits on its own
    unless it runs inside an explicit `transaction(conn)` block. Multi-step
    writes are therefore always visible in the code as one block.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 3000")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """All-or-nothing block. IMMEDIATE takes the write lock up front, so two
    ingestions cannot interleave their check-then-insert steps."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def init_db(path: Path) -> None:
    conn = connect(path)
    try:
        conn.executescript(SCHEMA)
    finally:
        conn.close()
