# tests/test_drift_injection.py
"""
Deliberately feeds the online model a stable relationship for a while,
then flips the relationship (simulating real concept drift — e.g., a
change in hospital protocol that changes what vitals pattern predicts
sepsis), and asserts the ADWIN detector actually notices.
"""
import random
from ml.online.adwin_model import OnlineDriftSensor


def make_stable_sample(rng):
    """High heart rate + low BP => sepsis label 1. Stable rule."""
    hr = rng.uniform(60, 140)
    sbp = rng.uniform(70, 140)
    label = 1 if (hr > 100 and sbp < 100) else 0
    return {"heart_rate": hr, "sbp": sbp}, label


def make_drifted_sample(rng):
    """
    Same feature ranges, but the RULE that defines sepsis has flipped —
    now it's LOW heart rate + HIGH BP that signals risk. This is a
    concept drift injection: same feature distribution, different
    input-output relationship.
    """
    hr = rng.uniform(60, 140)
    sbp = rng.uniform(70, 140)
    label = 1 if (hr < 100 and sbp > 100) else 0
    return {"heart_rate": hr, "sbp": sbp}, label


def test_adwin_detects_injected_concept_drift():
    rng = random.Random(42)
    sensor = OnlineDriftSensor()

    # Phase 1: 500 samples of the STABLE relationship — let the model learn it
    for _ in range(500):
        x, y = make_stable_sample(rng)
        sensor.step(x, y)

    drift_before_injection = len(sensor.drift_events)

    # Phase 2: 500 samples of the DRIFTED relationship
    drift_found_in_phase_2 = False
    for _ in range(500):
        x, y = make_drifted_sample(rng)
        result = sensor.step(x, y)
        if result["drift_detected"]:
            drift_found_in_phase_2 = True

    assert drift_found_in_phase_2, (
        "ADWIN failed to detect an injected concept drift within 500 "
        "samples — either the detector is mis-configured or the "
        "injection isn't actually changing model error rate enough."
    )
    print(f"\nDrift events before injection: {drift_before_injection}")
    print(f"Drift correctly detected after injecting concept drift: {drift_found_in_phase_2}")