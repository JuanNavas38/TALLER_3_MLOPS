import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

VALID_PENGUIN = {
    "island": "Biscoe",
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181.0,
    "body_mass_g": 3750.0,
    "sex": "male",
}


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_rf():
    response = client.post("/predict/rf", json=VALID_PENGUIN)
    assert response.status_code == 200
    data = response.json()
    assert data["model_used"] == "rf"
    assert data["prediction"] in ["Adelie", "Chinstrap", "Gentoo"]


def test_predict_svm():
    response = client.post("/predict/svm", json=VALID_PENGUIN)
    assert response.status_code == 200
    data = response.json()
    assert data["model_used"] == "svm"
    assert data["prediction"] in ["Adelie", "Chinstrap", "Gentoo"]


def test_predict_invalid_model_type():
    response = client.post("/predict/xgboost", json=VALID_PENGUIN)
    assert response.status_code == 422


def test_predict_invalid_island():
    penguin = {**VALID_PENGUIN, "island": "Atlantis"}
    response = client.post("/predict/rf", json=penguin)
    assert response.status_code == 400


def test_predict_invalid_sex():
    penguin = {**VALID_PENGUIN, "sex": "unknown"}
    response = client.post("/predict/rf", json=penguin)
    assert response.status_code == 400


def test_predict_out_of_range_bill_length():
    penguin = {**VALID_PENGUIN, "bill_length_mm": -5.0}
    response = client.post("/predict/rf", json=penguin)
    assert response.status_code == 422
