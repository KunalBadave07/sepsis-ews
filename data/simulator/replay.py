# data/simulator/replay.py
import json
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
from kafka import KafkaProducer

TOPIC = "vitals.raw"
BOOTSTRAP = "localhost:9092"

# PhysioNet column -> our schema field
COLUMN_MAP = {
    "HR": "heart_rate",
    "Resp": "resp_rate",
    "SBP": "sbp",
    "MAP": "map_bp",
    "Temp": "temp_c",
    "O2Sat": "spo2",
    "WBC": "wbc",
    "Lactate": "lactate",
}


def build_producer():
    return KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )


def replay_file(path: Path, speed: float = 0.3):
    """speed = seconds between rows. Lower = faster playback."""
    patient_id = path.stem  # e.g. "p000001"
    df = pd.read_csv(path, sep="|")

    producer = build_producer()
    
    # 1. Determine total ICU stay duration
    max_hours = df["ICULOS"].max() if "ICULOS" in df.columns else len(df)
    
    # 2. Set the start time in the past so the final row lands exactly at "now"
    base_time = datetime.now(timezone.utc) - timedelta(hours=int(max_hours))

    sent, skipped = 0, 0
    for i, row in df.iterrows():
        record = {"patient_id": patient_id}
        
        # 3. Add hours to the historical base_time
        current_hour = int(row.get("ICULOS", i))
        record["timestamp"] = (base_time + timedelta(hours=current_hour)).isoformat()

        for src_col, dst_field in COLUMN_MAP.items():
            val = row.get(src_col)
            if pd.notna(val):
                record[dst_field] = float(val)
            # If missing (NaN), we omit the key. 
            # Our updated Pydantic schema will safely default these to None.

        producer.send(TOPIC, value=record)
        sent += 1
        time.sleep(speed)

    producer.flush()
    print(f"Done. Sent {sent} rows for patient {patient_id}.")


if __name__ == "__main__":
    # Usage: python data/simulator/replay.py data/raw/training_setA/p000001.psv
    if len(sys.argv) < 2:
        print("Usage: python data/simulator/replay.py <path-to-psv-file>")
        sys.exit(1)
    replay_file(Path(sys.argv[1]))