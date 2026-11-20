"""Policy ingestion: Markdown -> sections -> chunks -> atomic version switch.

Atomicity:
  1. parse + validate + sanitise + chunk, all in memory   -> bad document: nothing written
  2. [transaction] reject identical content / concurrent ingestion, insert version as 'staging'
  3. write vectors to Chroma in batches, tagged with the new version_id
  4. verify the Chroma count for this version_id == number of chunks
  5. [transaction] insert chunk rows, old active -> 'superseded', new -> 'active'
     ^ the ONLY commit point that changes what employees see
  On any error in 3-5: delete this version's vectors, mark it 'failed', raise.

Retrieval always filters on the active version_id read from SQLite, so
vectors of a staging or failed version are never visible, even while they
exist. Superseded vectors are not deleted at the switch (a request that
started a millisecond earlier may still be using them); startup_cleanup()
removes them later.
"""

import hashlib
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from app.config import Settings
from app.db import transaction
from app.guards import sanitise_policy_text
from app.schemas import IngestionResult
from app.vectorstore import VectorStore

log = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 900
VECTOR_BATCH_SIZE = 8

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.+)$")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class IngestionError(Exception):
    """The document is invalid. Nothing was written. (HTTP 422)"""


class IngestionConflict(IngestionError):
    """Identical content, or another ingestion of this document is in progress. (HTTP 409)"""


class IngestionFailed(IngestionError):
    """Writing failed and was rolled back. The previous version is still active. (HTTP 500)"""


@dataclass(frozen=True)
class Section:
    section_id: str  # "4.1"
    title: str  # "Escort for women employees"
    body: str


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    section_id: str
    section_title: str
    text: str  # sanitised text, what the LLM will see
    flagged: bool  # injection-like text was removed from it


# --- 1. parsing and chunking (pure functions, no I/O) ------------------------

def parse_title(markdown: str) -> str | None:
    for line in markdown.splitlines():
        m = _HEADING.match(line)
        if m and len(m.group(1)) == 1:
            return m.group(2).strip()
    return None


def parse_sections(markdown: str) -> list[Section]:
    """Split a policy into numbered sections using its `##` / `###` headings.

    Rules (strict on purpose: a citation must point at an unambiguous section):
      - every `##`/`###` heading must be numbered, e.g. "## 4. Night shift", "### 4.1 Escort"
      - section numbers are unique
      - a heading with no text is only allowed if subsections follow it
      - text before the first numbered heading (title, version line) is not indexed
    """
    raw: list[tuple[str, str, list[str]]] = []
    for line in markdown.splitlines():
        m = _HEADING.match(line)
        if not m:
            if raw:
                raw[-1][2].append(line)
            continue
        level = len(m.group(1))
        if level == 1:
            continue  # document title
        if level > 3:
            raise IngestionError(f"only ## and ### headings are supported: {line!r}")
        num = _NUMBERED.match(m.group(2))
        if not num:
            raise IngestionError(f"heading must start with a section number: {line!r}")
        raw.append((num.group(1), num.group(2).strip(), []))

    if not raw:
        raise IngestionError("no numbered sections found")

    ids = [section_id for section_id, _, _ in raw]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise IngestionError(f"duplicate section numbers: {', '.join(duplicates)}")

    sections = []
    for i, (section_id, title, body_lines) in enumerate(raw):
        body = "\n".join(body_lines).strip()
        has_children = i + 1 < len(raw) and raw[i + 1][0].startswith(section_id + ".")
        if not body:
            if has_children:
                continue  # a heading that only groups subsections
            raise IngestionError(f"section {section_id} '{title}' is empty")
        sections.append(Section(section_id, title, body))
    return sections


def _pack(pieces: list[str], max_chars: int, sep: str) -> list[str]:
    """Greedily join pieces into strings no longer than max_chars."""
    out: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}{sep}{piece}" if current else piece
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                out.append(current)
            current = piece
    if current:
        out.append(current)
    return out


def _split_text(text: str, max_chars: int) -> list[str]:
    """Split on paragraphs, then sentences, then (last resort) hard cuts."""
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    for para in _PARAGRAPH_BREAK.split(text):
        if len(para) <= max_chars:
            pieces.append(para)
            continue
        for sentence in _SENTENCE_END.split(para):
            pieces.extend(sentence[i:i + max_chars] for i in range(0, len(sentence), max_chars))
    return _pack(pieces, max_chars, sep="\n\n")


def chunk_sections(sections: list[Section], max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """One chunk per section; a long section is split but keeps its section id."""
    chunks: list[Chunk] = []
    for s in sections:
        clean, flagged = sanitise_policy_text(s.body)
        for piece in _split_text(clean, max_chars):
            chunks.append(Chunk(len(chunks), s.section_id, s.title, piece, flagged))
    return chunks


def prepare_document(markdown: str, max_chars: int = MAX_CHUNK_CHARS) -> tuple[list[Chunk], list[str]]:
    """Step 1: everything that can fail on bad input, before any write."""
    chunks = chunk_sections(parse_sections(markdown), max_chars)
    flagged = sorted({c.section_id for c in chunks if c.flagged})
    warnings = [f"instruction-like text removed from section {sid}" for sid in flagged]
    return chunks, warnings


# --- 2-5. the versioned write ---------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _create_staging(conn: sqlite3.Connection, company_id: str, slug: str, title: str, sha: str) -> tuple[int, int, int]:
    """Step 2. Returns (document_id, version_id, version_no)."""
    with transaction(conn):
        conn.execute(
            """INSERT INTO policy_documents (company_id, slug, title) VALUES (?, ?, ?)
               ON CONFLICT (company_id, slug) DO UPDATE SET title = excluded.title""",
            (company_id, slug, title),
        )
        document_id = conn.execute(
            "SELECT id FROM policy_documents WHERE company_id = ? AND slug = ?", (company_id, slug)
        ).fetchone()["id"]

        statuses = {
            r["status"]: r["content_sha256"]
            for r in conn.execute(
                "SELECT status, content_sha256 FROM policy_versions "
                "WHERE document_id = ? AND status IN ('active', 'staging')",
                (document_id,),
            )
        }
        if "staging" in statuses:
            raise IngestionConflict("another version of this document is being ingested")
        if statuses.get("active") == sha:
            raise IngestionConflict("content is identical to the active version")

        version_no = conn.execute(
            "SELECT COALESCE(MAX(version_no), 0) + 1 AS n FROM policy_versions WHERE document_id = ?",
            (document_id,),
        ).fetchone()["n"]
        cur = conn.execute(
            """INSERT INTO policy_versions (document_id, version_no, status, content_sha256, created_at)
               VALUES (?, ?, 'staging', ?, ?)""",
            (document_id, version_no, sha, _now()),
        )
        return document_id, cur.lastrowid, version_no


def _write_vectors(store: VectorStore, company_id: str, document_id: int, version_id: int,
                   version_no: int, title: str, chunks: list[Chunk]) -> None:
    """Step 3. The section heading is embedded with the text so short questions match it."""
    for start in range(0, len(chunks), VECTOR_BATCH_SIZE):
        batch = chunks[start:start + VECTOR_BATCH_SIZE]
        store.add(
            company_id,
            ids=[f"{version_id}:{c.ordinal}" for c in batch],
            documents=[f"{title} > {c.section_id} {c.section_title}\n{c.text}" for c in batch],
            metadatas=[{
                "company_id": company_id,
                "document_id": document_id,
                "version_id": version_id,
                "version_no": version_no,
                "doc_title": title,
                "section_id": c.section_id,
                "section_title": c.section_title,
                "ordinal": c.ordinal,
                "flagged": c.flagged,
            } for c in batch],
        )


def _activate(conn: sqlite3.Connection, company_id: str, document_id: int, version_id: int,
              chunks: list[Chunk]) -> None:
    """Step 5. The single commit point."""
    with transaction(conn):
        conn.executemany(
            """INSERT INTO chunks (id, version_id, company_id, section_id, section_title, ordinal, text, flagged)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [(f"{version_id}:{c.ordinal}", version_id, company_id, c.section_id, c.section_title,
              c.ordinal, c.text, int(c.flagged)) for c in chunks],
        )
        conn.execute(
            "UPDATE policy_versions SET status = 'superseded' WHERE document_id = ? AND status = 'active'",
            (document_id,),
        )
        conn.execute(
            "UPDATE policy_versions SET status = 'active', activated_at = ?, chunk_count = ? WHERE id = ?",
            (_now(), len(chunks), version_id),
        )


def _mark_failed(conn: sqlite3.Connection, store: VectorStore, company_id: str, version_id: int,
                 exc: Exception) -> None:
    """Rollback for the Chroma side. Best effort: leftovers are invisible and swept at startup."""
    try:
        store.delete_version(company_id, version_id)
    except Exception:
        log.exception("could not delete vectors of failed version %s; startup cleanup will retry", version_id)
    conn.execute(
        "UPDATE policy_versions SET status = 'failed', error = ? WHERE id = ?",
        (f"{type(exc).__name__}: {exc}"[:500], version_id),
    )


def ingest_policy_version(
    conn: sqlite3.Connection,
    store: VectorStore,
    settings: Settings,
    company_id: str,
    slug: str,
    title: str,
    content: str,
) -> IngestionResult:
    """Ingest a new version of a company's policy document. See module docstring."""
    if len(content.encode("utf-8")) > settings.max_policy_bytes:
        raise IngestionError(f"document is larger than {settings.max_policy_bytes} bytes")

    chunks, warnings = prepare_document(content)
    sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    document_id, version_id, version_no = _create_staging(conn, company_id, slug, title, sha)

    try:
        _write_vectors(store, company_id, document_id, version_id, version_no, title, chunks)
        stored = store.count_version(company_id, version_id)
        if stored != len(chunks):
            raise RuntimeError(f"vector store holds {stored} chunks, expected {len(chunks)}")
        _activate(conn, company_id, document_id, version_id, chunks)
    except Exception as exc:
        log.exception("ingestion of %s/%s v%s failed; rolling back", company_id, slug, version_no)
        _mark_failed(conn, store, company_id, version_id, exc)
        raise IngestionFailed(
            f"ingestion of version {version_no} failed and was rolled back; the previous version is still active"
        ) from exc

    log.info("activated %s/%s v%s (%d chunks)", company_id, slug, version_no, len(chunks))
    return IngestionResult(
        version_id=version_id, version_no=version_no, status="active",
        chunk_count=len(chunks), warnings=warnings,
    )


# --- startup maintenance ---------------------------------------------------------

def startup_cleanup(conn: sqlite3.Connection, store: VectorStore) -> dict[str, int]:
    """Recover from a crash mid-ingestion and garbage-collect dead vectors.

    - 'staging' rows can only exist here if the process died mid-ingestion
      (single-instance assumption): mark them failed.
    - vectors of failed and superseded versions are deleted. Their chunk rows
      stay in SQLite as an audit trail.
    """
    stale = conn.execute("UPDATE policy_versions SET status = 'failed', error = 'interrupted' "
                         "WHERE status = 'staging'").rowcount
    dead = conn.execute(
        """SELECT v.id, d.company_id FROM policy_versions v
           JOIN policy_documents d ON d.id = v.document_id
           WHERE v.status IN ('failed', 'superseded')"""
    ).fetchall()
    for row in dead:
        store.delete_version(row["company_id"], row["id"])
    if stale:
        log.warning("marked %d interrupted ingestion(s) as failed", stale)
    return {"recovered_staging": stale, "purged_versions": len(dead)}
