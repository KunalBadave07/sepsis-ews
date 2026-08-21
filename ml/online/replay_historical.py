# ml/online/replay_historical.py
"""
Replays the Sprint 2 training parquet through the online model, IN
TEMPORAL ORDER PER PATIENT, to prove the mechanism runs cleanly on
real data before we deliberately try to break it in Step 4.
"""
import pandas as pd
from ml.online.adwin_model import OnlineDriftSensor

DATA_PATH = "data/processed/training_features.parquet"
FEATURE_COLS = [
    "heart_rate", "resp_rate", "sbp", "map_bp", "temp_c", "spo2",
    "hr_rolling_mean", "hr_rolling_std", "map_rolling_mean",
    "map_rolling_std", "shock_index",
]

def main():
    df = pd.read_parquet(DATA_PATH).dropna(subset=FEATURE_COLS)
    df = df.sort_values(["patient_id", "ICULOS"])  # CRITICAL: temporal order

    sensor = OnlineDriftSensor()

    for _, row in df.iterrows():
        x = {col: row[col] for col in FEATURE_COLS}
        y = int(row["SepsisLabel"])
        result = sensor.step(x, y)

        if result["drift_detected"]:
            print(f"  [DRIFT] at sample {result['n_seen']}, "
                  f"running accuracy={result['running_accuracy']:.3f}")

    print(f"\nProcessed {sensor.n_seen} samples.")
    print(f"Final running accuracy: {sensor.accuracy.get():.3f}")
    print(f"Total drift events on real historical data: {len(sensor.drift_events)}")

if __name__ == "__main__":
    main()