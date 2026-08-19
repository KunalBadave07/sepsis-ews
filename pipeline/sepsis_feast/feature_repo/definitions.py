# pipeline/sepsis_feast/definitions.py
from datetime import timedelta
from feast import Entity, FeatureView, Field, PushSource, FileSource
from feast.types import Float64, String
import pandas as pd
from pathlib import Path

patient = Entity(name="patient_id", join_keys=["patient_id"])

# a tiny placeholder parquet so Feast has an offline schema reference
placeholder_path = Path(__file__).parent / "placeholder.parquet"
if not placeholder_path.exists():
    pd.DataFrame({
        "patient_id": ["p000000"],
        "event_timestamp": [pd.Timestamp.utcnow()],
        "heart_rate": [80.0], "resp_rate": [16.0], "sbp": [120.0],
        "map_bp": [85.0], "temp_c": [37.0], "spo2": [98.0],
        "hr_rolling_mean": [80.0], "hr_rolling_std": [2.0],
        "map_rolling_mean": [85.0], "map_rolling_std": [2.0],
        "shock_index": [0.67],
    }).to_parquet(placeholder_path)

batch_source = FileSource(
    path=str(placeholder_path),
    timestamp_field="event_timestamp",
)

push_source = PushSource(name="vitals_push_source", batch_source=batch_source)

vitals_fv = FeatureView(
    name="patient_vitals",
    entities=[patient],
    ttl=timedelta(hours=24),
    schema=[
        Field(name="heart_rate", dtype=Float64),
        Field(name="resp_rate", dtype=Float64),
        Field(name="sbp", dtype=Float64),
        Field(name="map_bp", dtype=Float64),
        Field(name="temp_c", dtype=Float64),
        Field(name="spo2", dtype=Float64),
        Field(name="hr_rolling_mean", dtype=Float64),
        Field(name="hr_rolling_std", dtype=Float64),
        Field(name="map_rolling_mean", dtype=Float64),
        Field(name="map_rolling_std", dtype=Float64),
        Field(name="shock_index", dtype=Float64),
    ],
    source=push_source,
    online=True,
)