"""Mock JWT issuer. In a real system this would be the company's identity provider."""

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth import authenticate, create_token
from app.config import Settings, get_settings
from app.deps import get_db
from app.schemas import TokenRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token", response_model=TokenResponse)
def issue_token(
    body: TokenRequest,
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    user = authenticate(conn, body.username, body.password)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid username or password")
    return TokenResponse(access_token=create_token(user, settings), expires_in=settings.jwt_ttl_minutes * 60)
