# ml/online/shadow_runner.py
"""
Consumes the SAME vitals.clean stream from Sprint 1, running the online
drift sensor in shadow mode. NOTE: this uses SepsisLabel if present in
the simulated replay data as a stand-in for a "ground truth" signal —
in real deployment this label would arrive with clinical delay, which
is a known, documented limitation (see the honesty flag at the top of
this walkthrough).
"""
import json
from kafka import KafkaConsumer
from ml.online.adwin_model import OnlineDriftSensor

BOOTSTRAP = "localhost:9092"
IN_TOPIC = "vitals.clean"

FEATURE_COLS = ["heart_rate", "resp_rate", "sbp", "map_bp", "temp_c", "spo2"]


def run():
    consumer = KafkaConsumer(
        IN_TOPIC,
        bootstrap_servers=BOOTSTRAP,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id="shadow-drift-sensor",
    )
    sensor = OnlineDriftSensor()

    print("Shadow drift sensor running against live stream (Ctrl+C to stop)...")
    for msg in consumer:
        reading = msg.value
        x = {col: reading.get(col, 0.0) for col in FEATURE_COLS}
        # NOTE: live stream has no label — this is a placeholder using
        # a naive proxy rule ONLY so the mechanism has something to run
        # against in shadow mode. This is explicitly NOT how a real
        # production shadow model would get its ground truth.
        proxy_label = 1 if (x["heart_rate"] > 100 and x["sbp"] < 100) else 0

        result = sensor.step(x, proxy_label)
        if result["drift_detected"]:
            print(f"  [SHADOW DRIFT] at sample {result['n_seen']}")


if __name__ == "__main__":
    run()
    