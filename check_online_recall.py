# check_online_recall.py
# Run this from your project root: python check_online_recall.py
# Proves (or disproves) whether the 98% accuracy is real signal or a
# majority-class illusion.

import pandas as pd
from river import tree, drift, metrics
from ml.online.shadow_runner import OnlineDriftSensor

DATA_PATH = "data/processed/training_features.parquet"
FEATURE_COLS = [
    "heart_rate", "resp_rate", "sbp", "map_bp", "temp_c", "spo2",
    "hr_rolling_mean", "hr_rolling_std", "map_rolling_mean",
    "map_rolling_std", "shock_index",
]

df = pd.read_parquet(DATA_PATH).dropna(subset=FEATURE_COLS)
df = df.sort_values(["patient_id", "ICULOS"])

# model = tree.HoeffdingTreeClassifier()
# adwin = drift.ADWIN()
model = OnlineDriftSensor()

acc = metrics.Accuracy()
recall = metrics.Recall()      # of actual sepsis cases, how many did we catch?
precision = metrics.Precision()  # of predicted sepsis, how many were real?
report = metrics.ClassificationReport()

for _, row in df.iterrows():
    x = {col: row[col] for col in FEATURE_COLS}
    y = int(row["SepsisLabel"])

    y_pred = model.model.predict_one(x)
    if y_pred is None:
        y_pred = 0

    acc.update(y, y_pred)
    recall.update(y, y_pred)
    precision.update(y, y_pred)
    report.update(y, y_pred)

    model.model.learn_one(x, y)

print(f"Accuracy:  {acc.get():.4f}")
print(f"Recall (sepsis caught / sepsis actually present): {recall.get():.4f}")
print(f"Precision: {precision.get():.4f}")
print()
print(report)