from avalanche_forecast.webapp import app


def test_dashboard_loads():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"Local research dashboard" in response.data
    assert b"Sentinel tile model results" in response.data
    assert b"Model input imagery reference" in response.data


def test_dashboard_rejects_missing_manifest():
    client = app.test_client()
    response = client.post("/api/train", data={"manifest": "missing.csv", "horizon_hours": "6"})
    assert response.status_code == 400


def test_status_shows_unconfigured_official_report(monkeypatch):
    monkeypatch.delenv("OFFICIAL_REPORT_URL", raising=False)
    client = app.test_client()
    body = client.get("/api/status").get_json()
    assert body["official_report"]["configured"] is False


def test_sar_preview_rejects_missing_sample():
    client = app.test_client()
    response = client.get("/api/sar-preview")
    assert response.status_code == 400
