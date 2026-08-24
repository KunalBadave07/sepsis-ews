# monitoring/retrain_trigger/run_swadt_live.py
"""
Replays historical feature data through SWADT using REAL SHAP importances
from your Sprint 2 explainer, instead of synthetic fixed values.
"""
import sys
sys.path.append('/app')
import pandas as pd
from monitoring.retrain_trigger.swadt import SWADT
from ml.explainability.shap_pipeline import SepsisExplainer

FEATURE_COLS = [
    "heart_rate", "resp_rate", "sbp", "map_bp", "temp_c", "spo2",
    "hr_rolling_mean", "hr_rolling_std", "map_rolling_mean",
    "map_rolling_std", "shock_index",
]
DATA_PATH = "data/processed/training_features.parquet"


def main():
    df = pd.read_parquet(DATA_PATH).dropna(subset=FEATURE_COLS)
    df = df.sort_values(["patient_id", "ICULOS"]).head(5000)  # keep it fast for a first real run

    explainer = SepsisExplainer()
    swadt = SWADT(FEATURE_COLS, window_size=250, lam=1.5, k=2.5, psi_cap=1.0)

    trigger_count = 0
    for i, row in df.iterrows():
        feature_vals = {col: row[col] for col in FEATURE_COLS}

        explanation = explainer.explain_row(feature_vals, top_n=len(FEATURE_COLS))
        shap_map = {item["feature"]: abs(item["shap_value"]) for item in explanation["top_features"]}

        result = swadt.observe(feature_vals, shap_map)
        if result and result["triggered"]:
            trigger_count += 1
            print(f"[SWADT TRIGGER] score={result['trigger_score']:.3f} "
                  f"threshold={result['threshold']:.3f} "
                  f"top_contributors={result['top_contributors']}")

    print(f"\nTotal SWADT triggers over {len(df)} real samples: {trigger_count}")


if __name__ == "__main__":
    main()