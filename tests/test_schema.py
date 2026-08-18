# tests/test_schema.py
from datetime import datetime
from pipeline.validation.schema import VitalReading

def test_valid_reading():
    r = VitalReading(
        patient_id="p000001",
        timestamp=datetime.utcnow(),
        heart_rate=88, resp_rate=18, sbp=120, map_bp=85,
        temp_c=37.0, spo2=98, wbc=9.2, lactate=1.1,
    )
    assert r.heart_rate == 88

def test_invalid_heart_rate_rejected():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        VitalReading(
            patient_id="p000001",
            timestamp=datetime.utcnow(),
            heart_rate=999,  # out of range on purpose
            resp_rate=18, sbp=120, map_bp=85, temp_c=37.0, spo2=98,
        )