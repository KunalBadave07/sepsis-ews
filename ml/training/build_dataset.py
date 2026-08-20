# ml/training/build_dataset.py
"""
Reads every PhysioNet .psv file, computes the same rolling features
your streaming pipeline computes (rolling mean/std, shock index),
and writes one big Parquet file ready for training.
"""
import glob
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/training_setA")
OUT_PATH = Path("data/processed/training_features.parquet")

COLUMN_MAP = {
    "HR": "heart_rate", "Resp": "resp_rate", "SBP": "sbp",
    "MAP": "map_bp", "Temp": "temp_c", "O2Sat": "spo2",
    "WBC": "wbc", "Lactate": "lactate",
}



def build_patient_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|")
    df = df.rename(columns=COLUMN_MAP)
    df["patient_id"] = path.stem

    # rolling stats, computed causally (no future leakage — window looks BACKWARD only)
    df["hr_rolling_mean"] = df["heart_rate"].rolling(window=8, min_periods=1).mean()
    df["hr_rolling_std"] = df["heart_rate"].rolling(window=8, min_periods=1).std().fillna(0)
    df["map_rolling_mean"] = df["map_bp"].rolling(window=8, min_periods=1).mean()
    df["map_rolling_std"] = df["map_bp"].rolling(window=8, min_periods=1).std().fillna(0)
    df["shock_index"] = df["heart_rate"] / df["sbp"].replace(0, pd.NA)

    keep_cols = [
        "patient_id", "ICULOS", "heart_rate", "resp_rate", "sbp", "map_bp",
        "temp_c", "spo2", "wbc", "lactate", "hr_rolling_mean", "hr_rolling_std",
        "map_rolling_mean", "map_rolling_std", "shock_index", "SepsisLabel",
    ]
    return df[keep_cols]


def main():
    files = sorted(glob.glob(str(RAW_DIR / "*.psv")))
    print(f"Found {len(files)} patient files.")

    all_frames = []
    for i, f in enumerate(files):
        try:
            all_frames.append(build_patient_features(Path(f)))
        except Exception as e:
            print(f"  skipped {f}: {e}")
        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{len(files)}")

    full = pd.concat(all_frames, ignore_index=True)
    full = full.dropna(subset=["heart_rate", "resp_rate", "sbp", "map_bp", "temp_c", "spo2"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(full)} rows across {full['patient_id'].nunique()} patients to {OUT_PATH}")
    print(f"Sepsis-positive rate: {full['SepsisLabel'].mean():.3%}")


if __name__ == "__main__":
    main()