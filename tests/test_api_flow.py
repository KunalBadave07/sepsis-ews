# tests/test_api_flow.py
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def get_token(username="nurse_jane", password="demo123"):
    resp = client.post("/v1/auth/token", data={"username": username, "password": password})
    assert resp.status_code == 200
    return resp.json()["access_token"]


def test_predict_and_acknowledge_flow():
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}

    vitals = {
        "heart_rate": 118, "resp_rate": 26, "sbp": 88, "map_bp": 58,
        "temp_c": 38.6, "spo2": 91, "hr_rolling_mean": 105, "hr_rolling_std": 8,
        "map_rolling_mean": 65, "map_rolling_std": 6, "shock_index": 1.34,
    }
    resp = client.post("/v1/predict?patient_id=p000001", json=vitals, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "treatment" not in body
    assert 0.0 <= body["probability"] <= 1.0

    ack_resp = client.post(
        f"/v1/predict/{body['prediction_id']}/acknowledge",
        json={"disposition": "reviewed_no_action"},
        headers=headers,
    )
    assert ack_resp.status_code == 200


def test_predict_requires_auth():
    vitals = {"heart_rate": 80}
    resp = client.post("/v1/predict?patient_id=p000001", json=vitals)
    assert resp.status_code == 401