# tests/test_model.py
import os
import pytest

MODEL_EXISTS = os.path.exists("ml/registry/latest_model.pkl")

@pytest.mark.skipif(not MODEL_EXISTS, reason="model not trained yet")
def test_explainer_returns_valid_probability():
    from ml.explainability.shap_pipeline import SepsisExplainer
    explainer = SepsisExplainer()
    row = {
        "heart_rate": 80, "resp_rate": 16, "sbp": 120, "map_bp": 85,
        "temp_c": 37.0, "spo2": 98, "hr_rolling_mean": 80, "hr_rolling_std": 2,
        "map_rolling_mean": 85, "map_rolling_std": 2, "shock_index": 0.67,
    }
    result = explainer.explain_row(row)
    assert 0.0 <= result["probability"] <= 1.0
    assert len(result["top_features"]) == 5
    assert result["latency_ms"] < 200  # generous CI margin; local target is 50ms