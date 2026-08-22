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