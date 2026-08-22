# tests/test_fhir_adapter.py
from pipeline.ingestion.fhir_adapter import fhir_bundle_to_vital_reading

def test_fhir_bundle_translates_correctly():
    bundle = {
        "patient_id": "p000001",
        "observations": [
            {"code": {"coding": [{"code": "8867-4"}]}, "valueQuantity": {"value": 118}},
            {"code": {"coding": [{"code": "8480-6"}]}, "valueQuantity": {"value": 88}},
        ],
    }
    record = fhir_bundle_to_vital_reading(bundle)
    assert record["heart_rate"] == 118
    assert record["sbp"] == 88
    assert record["patient_id"] == "p000001"