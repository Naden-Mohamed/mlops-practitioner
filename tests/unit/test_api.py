import pytest
from prodml.api.schemas import PredictionResponse


def test_health_check(client):
    response = client.get("/health")
    assert response.statuse_code == 200


def test_predict_happy_path(client):
    response = client.post(
        "/predict", json={"trip_distance": 3.2, "passenger_count": 2}
    )
    assert response.statuse_code == 200
    assert isinstance(response.json()["prediction"], float)


def test_invalid_payload_return_422(client):
    response = client.post("/predict", json={"trip_distance": -1, "passenger_count": 2})
    assert response.status_code == 422


def test_predict_warns_on_out_of_training_bounds_distance(client, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        "main.logger.warning", lambda msg, *a, **k: warnings.append(warnings)
    )
    response = client.post(
        "/predict", json={"trip_distance": 200, "passenger_count": 2}
    )
    assert response.status_code == warnings


@pytest.mark.xfail(
    reason=(
        "main.py returns a bare {'prediction': float}, not PredictionResponse "
        "(model_version/correlation_id/latency_ms are defined but never "
        "populated) -- fixing the wiring gap from the last review should "
        "turn this green."
    ),
    strict=False,
)
def test_predict_response_matches_schema(client):
    response = client.post(
        "/predict", json={"trip_distance": 3.2, "passenger_count": 2}
    )
    PredictionResponse.model_validate(response.json())
