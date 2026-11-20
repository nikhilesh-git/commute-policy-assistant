"""Idempotent demo data: companies, users and each company's version-1 policy.

Demo credentials are intentionally public (they only exist inside this local
demo database). Real secrets (JWT_SECRET, GROQ_API_KEY) come from the environment.

Only `*.v1.md` files are seeded. Later versions (e.g. Nimbus v2) are uploaded
through the admin API, which is how the version switch is demonstrated.
"""

import logging
import sqlite3

from app.auth import hash_password
from app.config import Settings
from app.ingestion import ingest_policy_version, parse_title
from app.vectorstore import VectorStore

log = logging.getLogger(__name__)

COMPANIES = [
    ("nimbus", "Nimbus Tech"),
    ("orion", "Orion Finserv"),
    ("kestrel", "Kestrel Health"),
]

DEMO_PASSWORD = "demo1234"

USERS = [  # (username, company_id, role)
    ("alice@nimbus", "nimbus", "employee"),
    ("admin@nimbus", "nimbus", "admin"),
    ("bob@orion", "orion", "employee"),
    ("admin@orion", "orion", "admin"),
    ("chitra@kestrel", "kestrel", "employee"),
    ("admin@kestrel", "kestrel", "admin"),
]

SEED_SUFFIX = ".v1.md"


def seed(conn: sqlite3.Connection) -> None:
    conn.executemany("INSERT OR IGNORE INTO companies (id, name) VALUES (?, ?)", COMPANIES)
    for username, company_id, role in USERS:
        exists = conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
        if not exists:
            conn.execute(
                "INSERT INTO users (username, password_hash, company_id, role) VALUES (?, ?, ?, ?)",
                (username, hash_password(DEMO_PASSWORD), company_id, role),
            )


def seed_policies(conn: sqlite3.Connection, store: VectorStore, settings: Settings) -> list[str]:
    """Ingest policies/<company_id>/<slug>.v1.md for documents that have no version yet."""
    ingested = []
    known = {r["id"] for r in conn.execute("SELECT id FROM companies")}
    for path in sorted(settings.policies_dir.glob(f"*/*{SEED_SUFFIX}")):
        company_id, slug = path.parent.name, path.name.removesuffix(SEED_SUFFIX)
        if company_id not in known:
            log.warning("skipping %s: unknown company %r", path, company_id)
            continue
        already = conn.execute(
            """SELECT 1 FROM policy_versions v JOIN policy_documents d ON d.id = v.document_id
               WHERE d.company_id = ? AND d.slug = ? AND v.status IN ('active', 'superseded')""",
            (company_id, slug),
        ).fetchone()
        if already:
            continue
        content = path.read_text(encoding="utf-8")
        ingest_policy_version(conn, store, settings, company_id, slug, parse_title(content) or slug, content)
        ingested.append(f"{company_id}/{slug}")
    return ingested
