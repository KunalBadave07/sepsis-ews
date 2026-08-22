# Sprint 3, Milestone 3 — Guardrailed FastAPI Serving Layer
### Every file, every command, in order.

This is where "advisory only, structurally enforced" and "human-in-the-loop" stop being promises in a document and become code you can point to.

---

## Step 1 — Install what you need
```
pip install "python-jose[cryptography]" "passlib[bcrypt]" python-multipart
```
(SQLite is built into Python — no install needed for the audit log.)

## Step 2 — The response schema that CANNOT contain a treatment field

New File → `api/schemas/predict.py`:
```python
# api/schemas/predict.py
from pydantic import BaseModel, ConfigDict


class ShapFeature(BaseModel):
    feature: str
    shap_value: float


class PredictionResponse(BaseModel):
    """
    This is a hard architectural boundary, not a style choice: this
    schema has NO field for treatment, medication, or dosage, and
    `model_config` forbids extra fields — meaning nothing downstream
    can smuggle one in later without this file being deliberately
    edited and reviewed. See tests/test_no_treatment_field.py.
    """
    model_config = ConfigDict(extra="forbid")

    prediction_id: str
    patient_id: str
    probability: float
    risk_tier: str  # "stable" | "at_risk" | "deteriorating" — NEVER a treatment
    top_features: list[ShapFeature]
    model_version: str
    latency_ms: float


class AcknowledgeRequest(BaseModel):
    disposition: str  # e.g. "reviewed_no_action", "escalated", "false_alarm"
    clinician_note: str | None = None
```

## Step 3 — Prove the boundary with a real test (not just a comment)

New File → `tests/test_no_treatment_field.py`:
```python
# tests/test_no_treatment_field.py
import pytest
from pydantic import ValidationError
from api.schemas.predict import PredictionResponse, ShapFeature

VALID_KWARGS = dict(
    prediction_id="pred_123", patient_id="p000001", probability=0.42,
    risk_tier="at_risk",
    top_features=[ShapFeature(feature="shock_index", shap_value=0.3)],
    model_version="v1", latency_ms=12.4,
)

def test_schema_has_no_treatment_field():
    assert "treatment" not in PredictionResponse.model_fields
    assert "medication" not in PredictionResponse.model_fields
    assert "dosage" not in PredictionResponse.model_fields

def test_schema_rejects_smuggled_treatment_field():
    with pytest.raises(ValidationError):
        PredictionResponse(**VALID_KWARGS, treatment="give 500mg of X")
```
Run:
```
pytest tests/test_no_treatment_field.py -v
```
Both should pass. This is a small test, but it's the one you point to when someone asks "how do you know your system can't accidentally suggest a drug?" — you don't say "we designed it carefully," you say "here's a test that fails the build if anyone adds that field."

## Step 4 — Minimal auth (JWT + role scaffold)

New File → `api/security/auth.py`:
```python
# api/security/auth.py
"""
Minimal JWT auth for the demo. NOTE: hardcoded demo users and a
hardcoded secret key are ONLY acceptable because this is a local
prototype. Real deployment would use a real identity provider (hospital
SSO) — this is flagged again in PRODUCTION_NOTES.md.
"""
from datetime import datetime, timedelta
from jose import jwt, JWTError
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

SECRET_KEY = "dev-only-secret-never-use-in-real-deployment"
ALGORITHM = "HS256"

# demo user store: username -> (password, role)
FAKE_USERS = {
    "nurse_jane": {"password": "demo123", "role": "clinician"},
    "admin_sam": {"password": "demo123", "role": "admin"},
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/auth/token")


def create_token(username: str, role: str) -> str:
    payload = {
        "sub": username,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=8),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"username": payload["sub"], "role": payload["role"]}
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                             detail="Invalid or expired token")


def require_role(*allowed_roles: str):
    def checker(user: dict = Depends(get_current_user)) -> dict:
        if user["role"] not in allowed_roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                                 detail=f"Requires one of roles: {allowed_roles}")
        return user
    return checker
```

## Step 5 — The audit log (SQLite, append-only)

New File → `monitoring/audit/audit_log.py`:
```python
# monitoring/audit/audit_log.py
import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path("monitoring/audit/audit.db")


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id TEXT PRIMARY KEY,
            patient_id TEXT,
            probability REAL,
            risk_tier TEXT,
            top_features TEXT,
            model_version TEXT,
            latency_ms REAL,
            requested_by TEXT,
            created_at TEXT,
            disposition TEXT,
            clinician_note TEXT,
            acknowledged_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(prediction_id, patient_id, probability, risk_tier,
                    top_features, model_version, latency_ms, requested_by):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO predictions
           (prediction_id, patient_id, probability, risk_tier, top_features,
            model_version, latency_ms, requested_by, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (prediction_id, patient_id, probability, risk_tier,
         json.dumps(top_features), model_version, latency_ms,
         requested_by, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def log_acknowledgment(prediction_id, disposition, clinician_note):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        """UPDATE predictions SET disposition = ?, clinician_note = ?,
           acknowledged_at = ? WHERE prediction_id = ?""",
        (disposition, clinician_note, datetime.utcnow().isoformat(), prediction_id),
    )
    conn.commit()
    updated = cur.rowcount
    conn.close()
    return updated > 0


def get_prediction(prediction_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM predictions WHERE prediction_id = ?",
                        (prediction_id,)).fetchone()
    conn.close()
    return dict(row) if row else None
```

## Step 6 — The predict router

New File → `api/routers/predict.py`:
```python
# api/routers/predict.py
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from api.schemas.predict import PredictionResponse, AcknowledgeRequest, ShapFeature
from api.security.auth import get_current_user, require_role
from ml.explainability.shap_pipeline import SepsisExplainer
from monitoring.audit import audit_log

router = APIRouter(prefix="/v1", tags=["predict"])

_explainer = None  # lazy-loaded singleton, avoids reloading the model per request


def get_explainer() -> SepsisExplainer:
    global _explainer
    if _explainer is None:
        _explainer = SepsisExplainer()
    return _explainer


def risk_tier_from_probability(p: float) -> str:
    if p < 0.2:
        return "stable"
    elif p < 0.5:
        return "at_risk"
    return "deteriorating"


class VitalsInput(dict):
    pass


@router.post("/predict", response_model=PredictionResponse)
def predict(patient_id: str, vitals: dict,
            user: dict = Depends(get_current_user)):
    start = time.perf_counter()
    explainer = get_explainer()

    try:
        result = explainer.explain_row(vitals, top_n=5)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Bad vitals payload: {e}")

    latency_ms = (time.perf_counter() - start) * 1000
    prediction_id = str(uuid.uuid4())
    risk_tier = risk_tier_from_probability(result["probability"])

    top_features = [ShapFeature(feature=f["feature"], shap_value=f["shap_value"])
                     for f in result["top_features"]]

    audit_log.log_prediction(
        prediction_id=prediction_id, patient_id=patient_id,
        probability=result["probability"], risk_tier=risk_tier,
        top_features=[f.model_dump() for f in top_features],
        model_version="lightgbm_v1", latency_ms=latency_ms,
        requested_by=user["username"],
    )

    return PredictionResponse(
        prediction_id=prediction_id, patient_id=patient_id,
        probability=result["probability"], risk_tier=risk_tier,
        top_features=top_features, model_version="lightgbm_v1",
        latency_ms=round(latency_ms, 2),
    )


@router.post("/predict/{prediction_id}/acknowledge")
def acknowledge(prediction_id: str, ack: AcknowledgeRequest,
                 user: dict = Depends(require_role("clinician", "admin"))):
    existing = audit_log.get_prediction(prediction_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Prediction not found")

    audit_log.log_acknowledgment(prediction_id, ack.disposition, ack.clinician_note)
    return {"status": "acknowledged", "prediction_id": prediction_id,
            "acknowledged_by": user["username"]}
```

## Step 7 — Auth token endpoint + main app

New File → `api/routers/auth.py`:
```python
# api/routers/auth.py
from fastapi import APIRouter, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import Depends
from api.security.auth import FAKE_USERS, create_token

router = APIRouter(prefix="/v1/auth", tags=["auth"])


@router.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = FAKE_USERS.get(form_data.username)
    if not user or user["password"] != form_data.password:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_token(form_data.username, user["role"])
    return {"access_token": token, "token_type": "bearer"}
```

New File → `api/main.py`:
```python
# api/main.py
from fastapi import FastAPI
from api.routers import predict, auth
from monitoring.audit.audit_log import init_db

app = FastAPI(title="Sepsis-EWS API", version="0.1.0")

init_db()

app.include_router(auth.router)
app.include_router(predict.router)


@app.get("/v1/health")
def health():
    return {"status": "ok"}
```

## Step 8 — Run it and test through the interactive docs

```
uvicorn api.main:app --reload --port 8000
```
Open `http://localhost:8000/docs`.

1. Click **`/v1/auth/token`** → "Try it out" → username `nurse_jane`, password `demo123` → Execute. Copy the `access_token` from the response.
2. Click the green **"Authorize"** button at the top of the page, paste the token, click Authorize.
3. Click **`/v1/predict`** → "Try it out" → fill in a `patient_id` and a `vitals` JSON body like:
```json
{
  "heart_rate": 118, "resp_rate": 26, "sbp": 88, "map_bp": 58,
  "temp_c": 38.6, "spo2": 91, "hr_rolling_mean": 105, "hr_rolling_std": 8,
  "map_rolling_mean": 65, "map_rolling_std": 6, "shock_index": 1.34
}
```
Execute. You should get back a `PredictionResponse` with a `prediction_id`, probability, risk tier, top features, and latency — no treatment field anywhere in sight.

4. Copy the `prediction_id` from that response, then call **`/v1/predict/{prediction_id}/acknowledge`** with a body like `{"disposition": "reviewed_no_action", "clinician_note": "vitals trending down on recheck"}`.

## Step 9 — Verify the audit trail actually recorded it

```
python -c "from monitoring.audit.audit_log import get_prediction; import json; print(json.dumps(get_prediction('PASTE_YOUR_PREDICTION_ID_HERE'), indent=2))"
```
You should see the full row: the original prediction, who requested it, and now the disposition and acknowledgment timestamp filled in. That round-trip — predict → log → acknowledge → reconstructable history — is your answer to the liability/guardrails question from the clinical framework document.

## Step 10 — One more automated test tying it together

New File → `tests/test_api_flow.py`:
```python
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
```
Run:
```
pytest tests/ -v
```
Everything from every prior sprint should still be green alongside these new ones.

Commit:
```
git add .
git commit -m "Milestone 3: guardrailed FastAPI layer with auth, audit log, acknowledgment flow"
git push
```

---

## Milestone 3 — Definition of Done
- [ ] `PredictionResponse` structurally cannot contain a treatment/medication/dosage field — proven by a real test, not just design intent
- [ ] Every prediction requires authentication; acknowledgment requires a clinician/admin role specifically
- [ ] Every prediction is logged to the audit trail with measured (not asserted) latency
- [ ] Acknowledgment updates the same audit record — full predict-to-disposition trail is reconstructable
- [ ] `pytest tests/` — everything from Sprints 1-3 passes together, not just the newest tests in isolation

Once this is solid, ping me for Milestone 4: the FHIR-shaped adapter and `PRODUCTION_NOTES.md` — the last piece of Sprint 3.
