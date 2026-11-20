from datetime import datetime, timedelta, timezone

import jwt
import pytest

from tests.conftest import TEST_SECRET


def _claims(**overrides):
    now = datetime.now(timezone.utc)
    claims = {
        "sub": "1", "company_id": "nimbus", "role": "employee",
        "iss": "commute-assistant", "aud": "commute-assistant-api",
        "iat": now, "exp": now + timedelta(minutes=5),
    }
    claims.update(overrides)
    return claims


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_login_success_returns_token_with_company(client, login):
    headers = login("alice@nimbus")
    claims = jwt.decode(headers["Authorization"].split()[1], options={"verify_signature": False})
    assert claims["company_id"] == "nimbus"
    assert claims["role"] == "employee"


def test_login_wrong_password(client):
    r = client.post("/auth/token", json={"username": "alice@nimbus", "password": "nope"})
    assert r.status_code == 401


@pytest.mark.parametrize("headers", [{}, _bearer("not-a-jwt")])
def test_ask_requires_valid_token(client, headers):
    assert client.post("/ask", json={"question": "hi"}, headers=headers).status_code == 401


def test_token_signed_with_other_secret_rejected(client):
    token = jwt.encode(_claims(), "another-secret-" + "y" * 40, algorithm="HS256")
    assert client.post("/ask", json={"question": "hi"}, headers=_bearer(token)).status_code == 401


def test_alg_none_token_rejected(client):
    token = jwt.encode(_claims(), key=None, algorithm="none")
    assert client.post("/ask", json={"question": "hi"}, headers=_bearer(token)).status_code == 401


def test_expired_token_rejected(client):
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    token = jwt.encode(_claims(iat=past, exp=past + timedelta(minutes=1)), TEST_SECRET, algorithm="HS256")
    assert client.post("/ask", json={"question": "hi"}, headers=_bearer(token)).status_code == 401


def test_token_without_company_rejected(client):
    claims = _claims()
    del claims["company_id"]
    token = jwt.encode(claims, TEST_SECRET, algorithm="HS256")
    assert client.post("/ask", json={"question": "hi"}, headers=_bearer(token)).status_code == 401


def test_token_for_unknown_company_rejected(client):
    token = jwt.encode(_claims(company_id="ghostco"), TEST_SECRET, algorithm="HS256")
    assert client.post("/ask", json={"question": "hi"}, headers=_bearer(token)).status_code == 401


def test_client_cannot_send_company_id(client, login):
    r = client.post("/ask", json={"question": "hi", "company_id": "orion"}, headers=login("alice@nimbus"))
    assert r.status_code == 422


def test_employee_cannot_use_admin_endpoints(client, login):
    assert client.get("/admin/policies", headers=login("alice@nimbus")).status_code == 403


def test_admin_sees_only_own_company(client, login):
    r = client.get("/admin/policies", headers=login("admin@nimbus"))
    assert r.status_code == 200
    assert r.json() == []  # no policies ingested yet; becomes a real isolation test in M3
