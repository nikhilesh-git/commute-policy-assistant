"""Admin endpoints over HTTP: role checks and company taken from the token."""

from tests.conftest import read_policy

ORION = read_policy("orion/transport-policy.v1.md")
NIMBUS_V1 = read_policy("nimbus/transport-policy.v1.md")


def upload(client, headers, content, slug="transport-policy", title="Policy"):
    return client.post(f"/admin/policies/{slug}/versions", json={"title": title, "content": content}, headers=headers)


def test_admin_upload_creates_active_version_with_warnings(client, login):
    r = upload(client, login("admin@orion"), ORION)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "active" and body["version_no"] == 1
    assert body["warnings"] == ["instruction-like text removed from section 6.2"]


def test_employee_cannot_upload(client, login):
    assert upload(client, login("bob@orion"), ORION).status_code == 403


def test_company_comes_from_token_not_request(client, login):
    # The Nimbus admin uploads; it can only ever land in Nimbus.
    assert upload(client, login("admin@nimbus"), NIMBUS_V1).status_code == 201
    assert len(client.get("/admin/policies", headers=login("admin@nimbus")).json()) == 1
    assert client.get("/admin/policies", headers=login("admin@orion")).json() == []


def test_body_cannot_carry_company(client, login):
    r = client.post(
        "/admin/policies/transport-policy/versions",
        json={"title": "x", "content": NIMBUS_V1, "company_id": "orion"},
        headers=login("admin@nimbus"),
    )
    assert r.status_code == 422


def test_invalid_document_is_422_and_duplicate_is_409(client, login):
    headers = login("admin@nimbus")
    assert upload(client, headers, "no sections here").status_code == 422
    assert upload(client, headers, NIMBUS_V1).status_code == 201
    assert upload(client, headers, NIMBUS_V1).status_code == 409


def test_bad_slug_is_rejected(client, login):
    assert upload(client, login("admin@nimbus"), NIMBUS_V1, slug="../etc").status_code in (404, 422)


def test_startup_seeds_v1_policies_for_all_companies(settings_env, monkeypatch, fake_llm):
    from fastapi.testclient import TestClient

    from app.config import get_settings
    from app.main import create_app

    monkeypatch.setenv("SEED_POLICIES_ON_STARTUP", "true")
    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        assert c.get("/health").json()["active_policies"] == 3
    # Restart: seeding is idempotent.
    with TestClient(create_app()) as c:
        assert c.get("/health").json()["active_policies"] == 3
