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