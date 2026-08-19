# pipeline/features/transforms.py
import json
from collections import defaultdict, deque

import pandas as pd
import polars as pl
from kafka import KafkaConsumer, KafkaProducer
from feast import FeatureStore
from feast.data_source import PushMode

# Initialize Feast connection
store = FeatureStore(repo_path="pipeline/sepsis_feast/feature_repo")

BOOTSTRAP = "localhost:9092"
IN_TOPIC = "vitals.clean"
OUT_TOPIC = "vitals.features"
WINDOW_SIZE = 8  # last 8 readings ~ 8 hours of data

# per-patient rolling buffer of recent readings
buffers: dict[str, deque] = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))


def compute_features(patient_id: str, buffer: deque) -> dict:
    df = pl.DataFrame(list(buffer))

    latest = df.tail(1).to_dicts()[0]

    features = {
        "patient_id": patient_id,
        "timestamp": latest["timestamp"],
        "heart_rate": latest["heart_rate"],
        "resp_rate": latest["resp_rate"],
        "sbp": latest["sbp"],
        "map_bp": latest["map_bp"],
        "temp_c": latest["temp_c"],
        "spo2": latest["spo2"],
        # rolling stats (over whatever window we have so far)
        "hr_rolling_mean": df["heart_rate"].mean(),
        "hr_rolling_std": df["heart_rate"].std() or 0.0,
        "map_rolling_mean": df["map_bp"].mean(),
        "map_rolling_std": df["map_bp"].std() or 0.0,
        # clinically meaningful composite feature (safe from NoneTypes)
        "shock_index": latest["heart_rate"] / latest["sbp"] if latest.get("sbp") else 0.0,
    }
    return features


def run():
    consumer = KafkaConsumer(
        IN_TOPIC,
        bootstrap_servers=BOOTSTRAP,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id="feature-consumer",
    )
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    print("Feature engine running. Waiting for clean messages...")
    
    # INDENTATION FIXED: The loop processes one message at a time
    for msg in consumer:
        reading = msg.value
        pid = reading["patient_id"]
        buffers[pid].append(reading)

        # 1. Compute features
        features = compute_features(pid, buffers[pid])
        
        # 2. Forward to downstream Kafka topic
        producer.send(OUT_TOPIC, value=features)
        
        # 3. Push real-time features to Feast Online Store (Redis)
        # (Updated to pd.Timestamp.now(tz="UTC") to prevent Pandas deprecation warnings)
        push_df = pd.DataFrame([{**features, "event_timestamp": pd.Timestamp.now(tz="UTC")}])
        store.push("vitals_push_source", push_df, to=PushMode.ONLINE)

        print(f"[FEATURES] {pid} shock_index={features['shock_index']:.2f} "
              f"hr_mean={features['hr_rolling_mean']:.1f}")


if __name__ == "__main__":
    run()