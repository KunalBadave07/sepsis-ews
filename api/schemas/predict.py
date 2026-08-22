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