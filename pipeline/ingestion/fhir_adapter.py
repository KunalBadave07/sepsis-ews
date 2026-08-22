# pipeline/ingestion/fhir_adapter.py
"""
Translates a simplified FHIR Observation-style payload into our internal
VitalReading schema. This is a SIMULATED adapter — it demonstrates the
correct integration pattern (LOINC-coded observations, not free-text
column names) without connecting to any real EHR system. See
PRODUCTION_NOTES.md for what real EHR integration would additionally require.
"""
from datetime import datetime

# LOINC codes are the real-world standard identifiers hospitals use for
# each vital sign type — this mapping is what a real FHIR integration
# would also need, just pointed at a live EHR instead of simulated input.
LOINC_MAP = {
    "8867-4": "heart_rate",
    "9279-1": "resp_rate",
    "8480-6": "sbp",
    "8478-0": "map_bp",
    "8310-5": "temp_c",
    "59408-5": "spo2",
}


def fhir_bundle_to_vital_reading(bundle: dict) -> dict:
    """
    bundle: a simplified FHIR Bundle containing Observation resources,
    e.g.:
    {
      "patient_id": "p000001",
      "observations": [
        {"code": {"coding": [{"code": "8867-4"}]}, "valueQuantity": {"value": 118}},
        {"code": {"coding": [{"code": "8480-6"}]}, "valueQuantity": {"value": 88}}
      ]
    }
    """
    record = {
        "patient_id": bundle["patient_id"],
        "timestamp": datetime.utcnow().isoformat(),
    }
    for obs in bundle.get("observations", []):
        code = obs["code"]["coding"][0]["code"]
        field = LOINC_MAP.get(code)
        if field:
            record[field] = obs["valueQuantity"]["value"]
    return record