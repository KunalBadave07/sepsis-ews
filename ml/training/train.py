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