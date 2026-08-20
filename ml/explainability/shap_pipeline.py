# ml/explainability/shap_pipeline.py
import time
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans

MODEL_PATH = "ml/registry/latest_model.pkl"
DATA_PATH = "data/processed/training_features.parquet"
FEATURE_COLS = [
    "heart_rate", "resp_rate", "sbp", "map_bp", "temp_c", "spo2",
    "hr_rolling_mean", "hr_rolling_std", "map_rolling_mean",
    "map_rolling_std", "shock_index",
]


class SepsisExplainer:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        df = pd.read_parquet(DATA_PATH).dropna(subset=FEATURE_COLS)

        # K-means-summarize the background set — using the full training
        # set for SHAP background is too slow; 50 representative points
        # is the standard production pattern.
        background_raw = df[FEATURE_COLS].sample(min(2000, len(df)), random_state=42)
        kmeans = KMeans(n_clusters=50, random_state=42, n_init=10).fit(background_raw)
        self.background = pd.DataFrame(kmeans.cluster_centers_, columns=FEATURE_COLS)

        self.explainer = shap.TreeExplainer(self.model, self.background)

    def explain_row(self, row: dict, top_n: int = 5) -> dict:
        start = time.perf_counter()

        x = pd.DataFrame([row])[FEATURE_COLS]
        shap_values = self.explainer.shap_values(x)

        # LightGBM binary classifier: shap_values may be a list [class0, class1]
        # or a single array depending on shap version — handle both
        values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

        contributions = list(zip(FEATURE_COLS, values))
        contributions.sort(key=lambda t: abs(t[1]), reverse=True)
        top_features = [{"feature": f, "shap_value": float(v)} for f, v in contributions[:top_n]]

        prob = float(self.model.predict_proba(x)[0][1])
        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "probability": prob,
            "top_features": top_features,
            "latency_ms": round(elapsed_ms, 2),
        }


if __name__ == "__main__":
    explainer = SepsisExplainer()

    # test with one made-up "at risk" patient row
    test_row = {
        "heart_rate": 118, "resp_rate": 26, "sbp": 88, "map_bp": 58,
        "temp_c": 38.6, "spo2": 91, "hr_rolling_mean": 105, "hr_rolling_std": 8,
        "map_rolling_mean": 65, "map_rolling_std": 6, "shock_index": 1.34,
    }

    result = explainer.explain_row(test_row)
    print(f"Predicted probability: {result['probability']:.3f}")
    print(f"Latency: {result['latency_ms']} ms")
    print("Top contributing features:")
    for f in result["top_features"]:
        direction = "increases" if f["shap_value"] > 0 else "decreases"
        print(f"  {f['feature']}: {direction} risk (SHAP={f['shap_value']:.4f})")