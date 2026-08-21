# tests/test_swadt.py
import numpy as np
from monitoring.retrain_trigger.swadt import SWADT

FEATURES = ["important_feature", "unimportant_feature"]


def run_scenario(drift_which: str, window_size=200, n_windows=4, drift_magnitude=4.0):
    """
    drift_which: 'important' or 'unimportant' — which feature actually
    shifts distribution starting from window 2 onward. The other feature
    stays stable throughout.

    SHAP importance is fixed and fed manually: 'important_feature' always
    reports a high mean |SHAP| value, 'unimportant_feature' always reports
    a near-zero one — simulating a feature the model barely relies on.
    """
    rng = np.random.default_rng(42)
    swadt = SWADT(FEATURES, window_size=window_size, lam=1.5, k=2.0, psi_cap=0.6)

    results = []
    for w in range(n_windows):
        for _ in range(window_size):
            important_val = rng.normal(0, 1)
            unimportant_val = rng.normal(0, 1)

            if w >= 1:  # from the 2nd window onward, inject drift
                if drift_which == "important":
                    important_val += drift_magnitude
                else:
                    unimportant_val += drift_magnitude

            shap_vals = {
                "important_feature": 0.8,   # consistently high importance
                "unimportant_feature": 0.02,  # consistently near-zero importance
            }
            feature_vals = {
                "important_feature": important_val,
                "unimportant_feature": unimportant_val,
            }

            result = swadt.observe(feature_vals, shap_vals)
            if result is not None:
                results.append(result)

    return results


def test_swadt_reacts_more_to_important_feature_drift():
    important_drift_results = run_scenario("important")
    unimportant_drift_results = run_scenario("unimportant")

    # compare the trigger score in the FIRST post-drift window (index 0,
    # since index -1 in reference-building isn't counted — the first
    # returned result IS the first real comparison window)
    important_score = important_drift_results[0]["trigger_score"]
    unimportant_score = unimportant_drift_results[0]["trigger_score"]

    print(f"\nTrigger score when IMPORTANT feature drifts:   {important_score:.4f} "
          f"(triggered={important_drift_results[0]['triggered']})")
    print(f"Trigger score when UNIMPORTANT feature drifts: {unimportant_score:.4f} "
          f"(triggered={unimportant_drift_results[0]['triggered']})")

    assert important_score > unimportant_score, (
        "SWADT should weight drift in an important feature more heavily "
        "than identical-magnitude drift in an unimportant feature — this "
        "is the entire point of the mechanism. If this fails, the "
        "importance-weighting logic in _evaluate_window is broken."
    )

    # the stronger claim: importance-weighted drift should actually cross
    # the adaptive threshold, while unimportant drift should be suppressed
    assert important_drift_results[0]["triggered"] is True, (
        "Expected the important-feature drift scenario to fire the trigger."
    )