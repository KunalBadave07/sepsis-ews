# Sprint 2 Walkthrough — Core Model & Explainability
### Every file, every command, in order.

Reminder before you start: activate your venv every new terminal session (`venv\Scripts\Activate.ps1` on Windows). You do NOT need Redpanda/Redis running for most of this sprint — training happens offline, on the raw `.psv` files directly. Only spin Docker back up at the very end when we test the explainability latency.

---

## DAY 1-3: Build the Offline Training Dataset

Your streaming pipeline computes features one row at a time as data arrives live. Training needs the opposite: the *entire* historical dataset, features computed, all at once. So we write a separate batch version — same clinical logic, different mode.

### Step 1 — Create the batch dataset builder
New File → `ml/training/build_dataset.py`:
```python
# ml/training/build_dataset.py
"""
Reads every PhysioNet .psv file, computes the same rolling features
your streaming pipeline computes (rolling mean/std, shock index),
and writes one big Parquet file ready for training.
"""
import glob
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/training")
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
```

### Step 2 — Run it
```
python ml/training/build_dataset.py
```
This will take a couple minutes depending on how many files you downloaded. Watch for the final print line — the **sepsis-positive rate** should land somewhere around 2-8%. If it prints something wildly different (like 50%), something's wrong with how `SepsisLabel` is being read — stop and check `data/processed/training_features.parquet` didn't get built from garbage.

**Why this matters, don't skip reading this:** notice the rolling windows are computed with `.rolling(window=8, min_periods=1)` — meaning at row 3 of a patient's stay, it only averages the 3 rows that exist so far, never peeking into that patient's future. This is the single most common bug in clinical ML projects (accidentally leaking future data backward) and you just avoided it on purpose. That's worth remembering for an interview answer.

---

## DAY 1-3 (continued): Train/Test Split by Patient

### Step 3 — Why you can't just do a random row split
If you randomly split *rows*, hour 5 and hour 6 of the same patient could end up in different sets — the model partially "sees" that patient during training. You must split by **whole patient**, never by row.

### Step 4 — Create the training script
New File → `ml/training/train.py`:
```python
# ml/training/train.py
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import optuna
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import average_precision_score, fbeta_score, brier_score_loss

DATA_PATH = "data/processed/training_features.parquet"
FEATURE_COLS = [
    "heart_rate", "resp_rate", "sbp", "map_bp", "temp_c", "spo2",
    "hr_rolling_mean", "hr_rolling_std", "map_rolling_mean",
    "map_rolling_std", "shock_index",
]
LABEL_COL = "SepsisLabel"

mlflow.set_experiment("sepsis-ews")


def load_split():
    df = pd.read_parquet(DATA_PATH)
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df["patient_id"]))
    return df.iloc[train_idx], df.iloc[test_idx]


def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "objective": "binary",
        "metric": "average_precision",
        "verbosity": -1,
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5, 50),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
    model = lgb.LGBMClassifier(**params, n_estimators=300)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(30, verbose=False)])
    preds = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, preds)


def main():
    train_df, test_df = load_split()

    # further split train into train/val for Optuna
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=1)
    tr_idx, val_idx = next(splitter.split(train_df, groups=train_df["patient_id"]))
    tr, val = train_df.iloc[tr_idx], train_df.iloc[val_idx]

    X_tr, y_tr = tr[FEATURE_COLS], tr[LABEL_COL]
    X_val, y_val = val[FEATURE_COLS], val[LABEL_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[LABEL_COL]

    print("Running Optuna search (20 trials — grab a coffee, ~5-10 min)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, X_tr, y_tr, X_val, y_val), n_trials=20)

    print("Best params:", study.best_params)

    with mlflow.start_run(run_name="lightgbm_sepsis_v1"):
        mlflow.log_params(study.best_params)

        final_model = lgb.LGBMClassifier(**study.best_params, n_estimators=300)
        final_model.fit(X_tr, y_tr)

        test_probs = final_model.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= 0.5).astype(int)

        pr_auc = average_precision_score(y_test, test_probs)
        fbeta2 = fbeta_score(y_test, test_preds, beta=2)
        brier = brier_score_loss(y_test, test_probs)

        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("f_beta_2", fbeta2)
        mlflow.log_metric("brier_score", brier)

        mlflow.lightgbm.log_model(final_model, artifact_path="model",
                                   registered_model_name="sepsis_lightgbm")

        print(f"\nFINAL TEST METRICS")
        print(f"  PR-AUC:       {pr_auc:.4f}")
        print(f"  F-beta(2):    {fbeta2:.4f}")
        print(f"  Brier score:  {brier:.4f}")

        import joblib
        joblib.dump(final_model, "ml/registry/latest_model.pkl")
        print("Model also saved locally to ml/registry/latest_model.pkl")


if __name__ == "__main__":
    main()
```

### Step 5 — Create the registry folder if it doesn't exist
```
mkdir ml/registry
```

### Step 6 — Start MLflow's UI in its own terminal tab
```
mlflow ui --port 5000
```
Leave this running. Open `http://localhost:5000` in your browser — keep it open, you'll watch your run appear here.

### Step 7 — Run training
In a **second** terminal tab (venv activated):
```
python ml/training/train.py
```
This will take several minutes — Optuna is trying 20 different hyperparameter combinations. Watch the printed trial progress. When it finishes, you'll see the FINAL TEST METRICS block.

**What "good" looks like here:** PR-AUC somewhere in the 0.25-0.45 range is realistic and respectable for this dataset with these features — this is NOT a 95%-accuracy problem, and if you see something suspiciously high like PR-AUC > 0.9, that's a red flag for data leakage, not a reason to celebrate. Go back and check your rolling windows and train/test split before trusting a number that good.

### Step 8 — Confirm it landed in MLflow
Refresh `http://localhost:5000`. Click into your run. You should see the logged params, the three metrics, and a registered model artifact. Click "Models" in the left nav — you should see `sepsis_lightgbm` registered.

Commit:
```
git add .
git commit -m "Day 1-3: offline training pipeline with Optuna + MLflow working"
git push
```
(Your `.gitignore` from Phase 1 already excludes `mlruns/` and `data/raw/` — confirm `data/processed/` isn't accidentally huge before committing; if the parquet file is large, add `data/processed/` to `.gitignore` too and just keep the pipeline script committed, not the generated data.)

---

## DAY 4-8: SHAP Explainability Pipeline

### Step 9 — Install SHAP if needed
```
pip install shap
```

### Step 10 — Build the explainability pipeline
New File → `ml/explainability/shap_pipeline.py`:
```python
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
```

### Step 11 — Run it
```
python ml/explainability/shap_pipeline.py
```
You should see a predicted probability, a latency number, and a ranked list of features with SHAP values.

**The number that actually matters here is `latency_ms`.** Your architecture spec commits to sub-50ms explainability. Run this a few times — the first run is always slower (model/explainer warming up), so look at run 2 and 3. If you're consistently over 50ms, the fix is almost always reducing the KMeans background from 50 clusters down to 20-30 — fewer background points means less computation per explanation, at a small cost to explanation stability. Try it and re-measure before assuming something's broken.

### Step 12 — Sanity-check the explanation makes clinical sense
Look at your printed output. For the test row above (high heart rate, low blood pressure, high temp, low O2), you should see `shock_index`, `heart_rate`, or `map_bp` showing up as top contributors *increasing* risk. If instead something like `resp_rate` barely moving the needle is somehow your #1 feature for an obviously deteriorating patient, don't just shrug — that's worth a second look at your training data before you trust this model's explanations in Sprint 3's API.

Commit:
```
git add .
git commit -m "Day 4-8: SHAP explainability pipeline with cached background, latency verified"
git push
```

---

## DAY 9-10: Buffer, Tests, and Sprint Close-Out

### Step 13 — Write one automated test
New File → `tests/test_model.py`:
```python
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
```
Run:
```
pytest tests/ -v
```

### Step 14 — Use remaining buffer days deliberately
- Re-run training with a different `random_state` in the split and compare PR-AUC — if it swings wildly between runs, your model is unstable on this small a dataset, and that's worth noting honestly in your README rather than hiding it.
- Test `explain_row()` on 3-4 different made-up patient scenarios (clearly healthy, clearly deteriorating, ambiguous) and read the explanations like a skeptical clinician would — this is genuinely good practice for the interview question "walk me through how you validated your explainability output."
- Don't start Sprint 3 early. SWADT (Sprint 3) directly consumes the SHAP pipeline you just built — if it's shaky, SWADT inherits that shakiness.

### Step 15 — Close out
```
git add .
git commit -m "Sprint 2 complete: trained model + SHAP explainability verified end to end"
git push
```

---

## You're Done With Sprint 2 When...
- [ ] `training_features.parquet` built with causal (no-future-leak) rolling features
- [ ] Train/test split is by patient, not by row
- [ ] MLflow shows a registered `sepsis_lightgbm` model with PR-AUC, F-beta, and Brier logged
- [ ] PR-AUC is realistic (not suspiciously perfect) — you've sanity-checked, not just accepted the number
- [ ] `SepsisExplainer.explain_row()` returns a probability + top-5 SHAP features
- [ ] You've verified latency is at or near sub-50ms, and know what to do if it isn't
- [ ] At least one automated test passes

Once every box is checked, ping me and we'll do Sprint 3 — that's where SWADT stops being a paper and starts being real code.
