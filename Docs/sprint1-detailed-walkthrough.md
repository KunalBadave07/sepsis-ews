# Sprint 1 Walkthrough — Ingestion & Feature Store
### Every file, every command, in order. Don't skip ahead.

Before anything: open your terminal, `cd` into `sepsis-ews`, activate your venv (`venv\Scripts\Activate.ps1` on Windows or `source venv/bin/activate` on Mac), and confirm Redpanda + Redis are running:
```
docker compose -f infra/docker-compose.yml up -d
docker ps
```
You should see `redpanda` and `redis` in the list with status "Up." If you don't, stop here and fix that first — nothing below works without them.

---

## DAY 1-2: Pydantic Validation Contract

### Step 1 — Create the schema file
In VS Code's file explorer, right-click `pipeline/validation/` → **New File** → name it `schema.py`.

Paste this exactly:
```python
# pipeline/validation/schema.py
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class VitalReading(BaseModel):
    patient_id: str
    timestamp: datetime
    heart_rate: float = Field(ge=0, le=300)
    resp_rate: float = Field(ge=0, le=80)
    sbp: float = Field(ge=0, le=300)
    map_bp: float = Field(ge=0, le=250)
    temp_c: float = Field(ge=25, le=45)
    spo2: float = Field(ge=0, le=100)
    wbc: float | None = Field(default=None, ge=0, le=100)
    lactate: float | None = Field(default=None, ge=0, le=30)

    @field_validator("timestamp")
    @classmethod
    def not_future(cls, v):
        if v > datetime.utcnow():
            raise ValueError("future timestamp — clock skew or bad sim data")
        return v
```
Save it. Nothing to run yet — this file is just a definition, not a script.

### Step 2 — Create the `__init__.py` files
Pydantic isn't the issue here — Python needs to treat your folders as importable packages. Create empty files (right-click folder → New File, leave content blank) at:
```
pipeline/__init__.py
pipeline/validation/__init__.py
pipeline/ingestion/__init__.py
pipeline/features/__init__.py
```
This is the "silly" step people forget and then get `ModuleNotFoundError` an hour later.

### Step 3 — Test the schema standalone before touching Kafka
New File → `tests/test_schema.py`:
```python
# tests/test_schema.py
from datetime import datetime
from pipeline.validation.schema import VitalReading

def test_valid_reading():
    r = VitalReading(
        patient_id="p000001",
        timestamp=datetime.utcnow(),
        heart_rate=88, resp_rate=18, sbp=120, map_bp=85,
        temp_c=37.0, spo2=98, wbc=9.2, lactate=1.1,
    )
    assert r.heart_rate == 88

def test_invalid_heart_rate_rejected():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        VitalReading(
            patient_id="p000001",
            timestamp=datetime.utcnow(),
            heart_rate=999,  # out of range on purpose
            resp_rate=18, sbp=120, map_bp=85, temp_c=37.0, spo2=98,
        )
```
Run it from the project root:
```
pytest tests/test_schema.py -v
```
You should see **2 passed**. If you see an import error instead, you're either not in the project root or forgot an `__init__.py` from Step 2.

---

## DAY 1-2 (continued): The Simulator

### Step 4 — Install kafka-python if it's not already
```
pip install kafka-python
```

### Step 5 — Create the replay simulator
New File → `data/simulator/replay.py`. This reads one PhysioNet `.psv` file and streams it row-by-row into Redpanda, simulating a live ICU monitor.

```python
# data/simulator/replay.py
import json
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta

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
    base_time = datetime.utcnow()

    sent, skipped = 0, 0
    for i, row in df.iterrows():
        record = {"patient_id": patient_id}
        # timestamp: synthetic, one hour per row (matches ICULOS hourly cadence)
        record["timestamp"] = (base_time + timedelta(hours=int(row.get("ICULOS", i)))).isoformat()

        for src_col, dst_field in COLUMN_MAP.items():
            val = row.get(src_col)
            if pd.notna(val):
                record[dst_field] = float(val)
            # if missing (NaN), just omit the key — the validator will
            # reject the row if a required field is missing, which is
            # CORRECT behavior: it should land in the dead-letter queue.

        producer.send(TOPIC, value=record)
        sent += 1
        time.sleep(speed)

    producer.flush()
    print(f"Done. Sent {sent} rows for patient {patient_id}.")


if __name__ == "__main__":
    # Usage: python data/simulator/replay.py data/raw/training/p000001.psv
    if len(sys.argv) < 2:
        print("Usage: python data/simulator/replay.py <path-to-psv-file>")
        sys.exit(1)
    replay_file(Path(sys.argv[1]))
```

### Step 6 — Find a real file to test with
In terminal:
```
ls data/raw/training | head -5
```
(Windows PowerShell: `Get-ChildItem data/raw/training | Select-Object -First 5`)

Copy one filename from the output, e.g. `p000001.psv`.

### Step 7 — Run it
```
python data/simulator/replay.py data/raw/training/p000001.psv
```
You should see it print `Done. Sent X rows...` after a short wait (X rows × 0.3 sec each). If it hangs forever with no output and no error, Redpanda likely isn't reachable — go back and confirm `docker ps` shows it running.

**Troubleshooting note:** if you get a `NoBrokersAvailable` error, your Redpanda container's advertised address doesn't match what your script expects. Check `infra/docker-compose.yml` still has `--advertise-kafka-addr PLAINTEXT://localhost:9092` exactly as given in the Phase 0 plan, then `docker compose -f infra/docker-compose.yml restart redpanda`.

---

## DAY 1-2 (continued): The Validation Consumer

### Step 8 — Create the consumer
New File → `pipeline/ingestion/consumer.py`:
```python
# pipeline/ingestion/consumer.py
import json
from datetime import datetime

from kafka import KafkaConsumer, KafkaProducer
from pydantic import ValidationError

from pipeline.validation.schema import VitalReading

BOOTSTRAP = "localhost:9092"
IN_TOPIC = "vitals.raw"
CLEAN_TOPIC = "vitals.clean"
DLQ_TOPIC = "vitals.dlq"


def run():
    consumer = KafkaConsumer(
        IN_TOPIC,
        bootstrap_servers=BOOTSTRAP,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id="validation-consumer",
    )
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    print("Consumer running. Waiting for messages... (Ctrl+C to stop)")
    clean_count, dlq_count = 0, 0

    for msg in consumer:
        raw = msg.value
        try:
            reading = VitalReading(**raw)
            producer.send(CLEAN_TOPIC, value=reading.model_dump(mode="json"))
            clean_count += 1
            print(f"[CLEAN] {reading.patient_id} @ {reading.timestamp} "
                  f"HR={reading.heart_rate}")
        except ValidationError as e:
            dlq_count += 1
            producer.send(DLQ_TOPIC, value={
                "raw": raw,
                "error": str(e),
                "rejected_at": datetime.utcnow().isoformat(),
            })
            print(f"[DLQ] rejected: {e.errors()[0]['msg']}")

        if (clean_count + dlq_count) % 20 == 0:
            print(f"--- running totals: clean={clean_count} dlq={dlq_count} ---")


if __name__ == "__main__":
    run()
```

### Step 9 — Run the full loop (two terminals, side by side)
Open **two terminal tabs**, both with venv activated.

**Tab 1** (start the consumer first, it needs to be listening):
```
python pipeline/ingestion/consumer.py
```

**Tab 2** (fire the simulator):
```
python data/simulator/replay.py data/raw/training/p000001.psv
```

Watch Tab 1. You should see a mix of `[CLEAN]` lines and occasionally `[DLQ]` lines (real ICU data has missing values — seeing some DLQ hits is *correct*, not a bug, it means your validation is doing its job). If you see **only** DLQ lines and zero CLEAN, something's wrong with the field mapping — double check `COLUMN_MAP` in `replay.py` matches your `.psv` file's actual header row (`head -1 data/raw/training/p000001.psv`).

Once you see clean messages flowing, stop both with `Ctrl+C`, and commit:
```
git add .
git commit -m "Day 1-2: ingestion + Pydantic validation working end to end"
git push
```

---

## DAY 3-4: Polars Feature Engineering

### Step 10 — Create the feature transform
New File → `pipeline/features/transforms.py`. This consumes `vitals.clean`, keeps a rolling per-patient buffer, and computes windowed features using Polars.

```python
# pipeline/features/transforms.py
import json
from collections import defaultdict, deque

import polars as pl
from kafka import KafkaConsumer, KafkaProducer

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
        # clinically meaningful composite feature
        "shock_index": latest["heart_rate"] / latest["sbp"] if latest["sbp"] else 0.0,
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
    for msg in consumer:
        reading = msg.value
        pid = reading["patient_id"]
        buffers[pid].append(reading)

        features = compute_features(pid, buffers[pid])
        producer.send(OUT_TOPIC, value=features)
        print(f"[FEATURES] {pid} shock_index={features['shock_index']:.2f} "
              f"hr_mean={features['hr_rolling_mean']:.1f}")


if __name__ == "__main__":
    run()
```

### Step 11 — Run all three stages together
Three terminal tabs now:
- Tab 1: `python pipeline/ingestion/consumer.py`
- Tab 2: `python pipeline/features/transforms.py`
- Tab 3: `python data/simulator/replay.py data/raw/training/p000001.psv`

Watch Tab 2 — you should see `[FEATURES]` lines with a `shock_index` and rolling means printing as the simulator plays through the patient file. Once confirmed, `Ctrl+C` everything, commit:
```
git add .
git commit -m "Day 3-4: Polars rolling feature engineering working"
git push
```

---

## DAY 5-7: Feast + Redis (Online Feature Store)

This is the fiddliest part of the sprint — go slow here.

### Step 12 — Install and initialize Feast
```
pip install feast[redis]
cd pipeline
feast init -t local sepsis_feast
```
This creates a `pipeline/sepsis_feast/` folder with example files. You're going to replace them.

### Step 13 — Configure the feature store
Open `pipeline/sepsis_feast/feature_store.yaml`, replace its full content with:
```yaml
project: sepsis_feast
registry: registry.db
provider: local
online_store:
  type: redis
  connection_string: "localhost:6379"
entity_key_serialization_version: 2
```

### Step 14 — Define the entity and push source
In `pipeline/sepsis_feast/`, create a new file `definitions.py` (delete the auto-generated example `.py` file Feast created for you — it references a demo dataset you don't have):
```python
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
```

### Step 15 — Apply the feature definitions
From inside `pipeline/sepsis_feast/`:
```
feast apply
```
You should see output confirming it registered `patient_vitals` and the `patient_id` entity. If it errors about Redis, confirm `docker ps` still shows `redis` running.

### Step 16 — Wire the feature transform script to push into Feast
Back in `pipeline/features/transforms.py`, add near the top:
```python
from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path="pipeline/sepsis_feast")
```
And inside `run()`, right after `producer.send(OUT_TOPIC, value=features)`, add:
```python
push_df = pd.DataFrame([{**features, "event_timestamp": pd.Timestamp.utcnow()}])
store.push("vitals_push_source", push_df, to=PushMode.ONLINE)
```
Add the import at the top: `from feast.data_source import PushMode`.

### Step 17 — Verify features actually land in Redis via Feast
New File (temporary, for testing) → `pipeline/features/check_online_store.py`:
```python
from feast import FeatureStore

store = FeatureStore(repo_path="pipeline/sepsis_feast")

result = store.get_online_features(
    features=[
        "patient_vitals:heart_rate",
        "patient_vitals:shock_index",
        "patient_vitals:hr_rolling_mean",
    ],
    entity_rows=[{"patient_id": "p000001"}],
).to_dict()

print(result)
```
Run the full chain again (consumer → transforms → simulator, three tabs as in Step 11), let it run for a few seconds, `Ctrl+C` everything, then run:
```
python pipeline/features/check_online_store.py
```
You should see a dictionary printed back with real numbers for `heart_rate`, `shock_index`, etc. — that's your proof the entire loop works: **raw stream → validated → feature-engineered → stored in the online feature store, queryable by patient ID.**

Commit:
```
git add .
git commit -m "Day 5-7: Feast + Redis online store wired end to end"
git push
```

---

## DAY 8-10: Integration Test & Buffer

### Step 18 — Write one real automated test for the whole loop
New File → `tests/test_integration.py`:
```python
# tests/test_integration.py
from pipeline.validation.schema import VitalReading
from pipeline.features.transforms import compute_features
from collections import deque
from datetime import datetime

def test_feature_pipeline_produces_shock_index():
    buffer = deque(maxlen=8)
    reading = {
        "patient_id": "test_patient",
        "timestamp": datetime.utcnow().isoformat(),
        "heart_rate": 110, "resp_rate": 22, "sbp": 90,
        "map_bp": 60, "temp_c": 38.5, "spo2": 94,
    }
    buffer.append(reading)
    features = compute_features("test_patient", buffer)
    assert features["shock_index"] > 1.0  # HR/SBP > 1 indicates possible shock
```
Run:
```
pytest tests/ -v
```
All tests should pass. This is the test your future CI/CD pipeline (Phase 6) will run automatically on every push.

### Step 19 — Use the remaining days as buffer, on purpose
Don't start Sprint 2 early just because you have days left. Instead:
- Replay **three or four different patient files**, not just `p000001`, to catch edge cases (a patient with heavier missing data will expose bugs the first file didn't).
- Read back through `consumer.py` and `transforms.py` and add comments explaining *why*, not just *what* — future you (and any recruiter skimming your GitHub) will thank you.
- Confirm `docker compose down` and `docker compose up -d` still bring the whole thing back cleanly after a restart — a pipeline that only works if you never turn your laptop off isn't actually done.

### Step 20 — Close out the sprint
```
git add .
git commit -m "Sprint 1 complete: full ingestion-to-online-store loop verified"
git push
```

---

## You're Done With Sprint 1 When...
- [ ] Simulator streams a `.psv` file into Redpanda
- [ ] Consumer validates every row, routes bad rows to DLQ, good rows to `vitals.clean`
- [ ] Feature transform computes rolling stats + shock index via Polars
- [ ] Features land in Redis via Feast and are queryable by `patient_id`
- [ ] At least one automated test passes in `tests/`
- [ ] Everything survives a `docker compose down && docker compose up -d` restart

If every box is checked, you're ready for Sprint 2 (the model). If even one isn't, stay here — a model trained on a shaky pipeline is a model you'll be debugging blind later.
