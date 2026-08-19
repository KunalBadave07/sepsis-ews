# tests/test_integration.py
from pipeline.validation.schema import VitalReading
from pipeline.features.transforms import compute_features
from collections import deque
from datetime import datetime

def test_feature_pipeline_produces_shock_index():
    buffer = deque(maxlen=8)
    reading = {
        "patient_id": "test_patient",
        "timestamp": datetime.utcnow().isoformat(),
        "heart_rate": 110, "resp_rate": 22, "sbp": 90,
        "map_bp": 60, "temp_c": 38.5, "spo2": 94,
    }
    buffer.append(reading)
    features = compute_features("test_patient", buffer)
    assert features["shock_index"] > 1.0  # HR/SBP > 1 indicates possible shock