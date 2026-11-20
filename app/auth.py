"""JWT issuing and verification (mock issuer) plus FastAPI auth dependencies.

The company a user belongs to comes ONLY from a verified token. No request
body or path parameter anywhere in the API carries a company id.
"""

import hashlib
import hmac
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import Settings, get_settings
from app.deps import get_db

_PBKDF2_ROUNDS = 200_000
_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class CurrentUser:
    user_id: int
    username: str
    company_id: str
    role: str


# --- passwords (stdlib only; demo-grade) -----------------------------------

def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), _PBKDF2_ROUNDS)
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    salt, expected = stored.split("$", 1)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), _PBKDF2_ROUNDS)
    return hmac.compare_digest(digest.hex(), expected)


# --- tokens ---------------------------------------------------------------

def create_token(user: CurrentUser, settings: Settings) -> str:
    now = datetime.now(timezone.utc)
    claims = {
        "sub": str(user.user_id),
        "username": user.username,
        "company_id": user.company_id,
        "role": user.role,
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_ttl_minutes),
    }
    return jwt.encode(claims, settings.effective_jwt_secret(), algorithm=settings.jwt_algorithm)


def decode_token(token: str, settings: Settings) -> dict:
    """Verify signature, expiry, issuer, audience and required claims.

    `algorithms=[...]` is explicit: this is what blocks `alg: none` and
    algorithm-confusion tokens.
    """
    return jwt.decode(
        token,
        settings.effective_jwt_secret(),
        algorithms=[settings.jwt_algorithm],
        audience=settings.jwt_audience,
        issuer=settings.jwt_issuer,
        options={"require": ["sub", "company_id", "role", "exp", "iat", "iss", "aud"]},
    )


def authenticate(conn: sqlite3.Connection, username: str, password: str) -> CurrentUser | None:
    row = conn.execute(
        "SELECT id, username, password_hash, company_id, role FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    if row is None or not verify_password(password, row["password_hash"]):
        return None
    return CurrentUser(row["id"], row["username"], row["company_id"], row["role"])


# --- FastAPI dependencies -------------------------------------------------

def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status.HTTP_401_UNAUTHORIZED, detail, headers={"WWW-Authenticate": "Bearer"})


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings),
    conn: sqlite3.Connection = Depends(get_db),
) -> CurrentUser:
    if creds is None:
        raise _unauthorized("missing bearer token")
    try:
        claims = decode_token(creds.credentials, settings)
    except jwt.PyJWTError:
        raise _unauthorized("invalid or expired token")

    # The company in the token must still exist (deleted tenant => no access).
    if conn.execute("SELECT 1 FROM companies WHERE id = ?", (claims["company_id"],)).fetchone() is None:
        raise _unauthorized("unknown company")
    if claims["role"] not in ("employee", "admin"):
        raise _unauthorized("unknown role")

    return CurrentUser(int(claims["sub"]), claims.get("username", ""), claims["company_id"], claims["role"])


def require_admin(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin role required")
    return user
