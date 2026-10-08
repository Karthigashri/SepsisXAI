from fastapi.testclient import TestClient

from sepsispulse.api import app


def test_model_absence_is_reported_without_returning_fake_predictions(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(
        "SEPSISPULSE_MODEL_PATH", str(tmp_path / "missing-model.joblib")
    )

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 503
        assert health.json()["model_status"] == "unavailable"

        response = client.post(
            "/api/predict/json",
            json={
                "patient_id": "test-patient",
                "records": [{"ICULOS": hour} for hour in range(1, 7)],
            },
        )
        assert response.status_code == 503
        assert "model artifact not found" in response.json()["detail"]


def test_dashboard_includes_required_research_disclaimer(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(
        "SEPSISPULSE_MODEL_PATH", str(tmp_path / "missing-model.joblib")
    )
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert (
            "Research/Clinical Decision-Support Prototype — Not for Medical Diagnosis."
            in response.text
        )
