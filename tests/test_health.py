def test_health_reports_components(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["database"] is True
    assert body["vectorstore"] is True
    assert body["llm_provider"] == "fake"
