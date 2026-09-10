from avalanche_forecast.webapp import app


def test_dashboard_loads():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"Local research dashboard" in response.data


def test_dashboard_rejects_missing_manifest():
    client = app.test_client()
    response = client.post("/api/train", data={"manifest": "missing.csv", "horizon_hours": "6"})
    assert response.status_code == 400
