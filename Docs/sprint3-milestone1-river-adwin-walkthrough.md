# Sprint 3, Milestone 1 — River + ADWIN Online Drift Sensor
### Every file, every command, in order.

### One honesty flag before you write a line of code
In a real hospital, the sepsis label for a given patient-hour isn't known instantly — it's confirmed retrospectively (a clinician diagnoses sepsis, and that gets charted, sometimes hours after the vitals that predicted it). A truly live online learner would have to handle *delayed labels*. For this milestone, we sidestep that honestly: we replay historical, already-labeled data in temporal order to prove the drift-detection *mechanism* works. Note this explicitly in your own head (and later in `PRODUCTION_NOTES.md`) — it's a real limitation of the demo, not something to gloss over.

Activate your venv. You don't need Docker running for this milestone — it's pure Python against your Sprint 2 parquet file.

---

## Step 1 — Confirm River is installed
```
pip show river
```
If nothing prints, install it:
```
pip install river
```

## Step 2 — Build the online model wrapper
New File → `ml/online/adwin_model.py`:
```python
# ml/online/adwin_model.py
"""
An online learner (HoeffdingTreeClassifier) paired with an explicit ADWIN
drift detector watching the model's own error stream. This is deliberately
kept separate from river's built-in AdaptiveRandomForestClassifier drift
handling, because we want VISIBILITY into exactly when and why a drift
event fires — that visibility is what SWADT (Milestone 2) will consume.
"""
from river import tree, drift, metrics


class OnlineDriftSensor:
    def __init__(self):
        self.model = tree.HoeffdingTreeClassifier()
        self.adwin = drift.ADWIN()
        self.accuracy = metrics.Accuracy()
        self.n_seen = 0
        self.drift_events = []  # log of (index, accuracy_at_drift)

    def step(self, x: dict, y_true: int) -> dict:
        """
        One online learning step: predict, score, learn, check for drift.
        Returns a dict describing what happened this step.
        """
        y_pred = self.model.predict_one(x)
        if y_pred is None:
            y_pred = 0  # cold start — model hasn't seen enough data yet

        error = 0 if y_pred == y_true else 1
        self.accuracy.update(y_true, y_pred)

        # feed the error signal into ADWIN — ADWIN watches for a change
        # in the DISTRIBUTION of this stream, not the raw values
        self.adwin.update(error)
        drifted = self.adwin.drift_detected

        self.model.learn_one(x, y_true)
        self.n_seen += 1

        if drifted:
            self.drift_events.append((self.n_seen, self.accuracy.get()))

        return {
            "n_seen": self.n_seen,
            "prediction": y_pred,
            "error": error,
            "running_accuracy": self.accuracy.get(),
            "drift_detected": drifted,
        }
```

## Step 3 — Test it against real historical data (no drift expected yet)
New File → `ml/online/replay_historical.py`:
```python
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
```

Run it:
```
python ml/online/replay_historical.py
```

**What to actually look for:** some drift events firing on real historical data is normal and even expected — real multi-patient data genuinely shifts as you move from one patient's baseline physiology to another's. What you're checking here isn't "zero drift events," it's "does this run to completion without crashing, and does the number of drift events feel reasonable (a handful, not literally every 10 samples)." If it's firing constantly, that's a signal ADWIN's default sensitivity is too aggressive for this data — note it, we'll tune it in Milestone 2 rather than randomly guessing now.

Commit:
```
git add .
git commit -m "Milestone 1a: online HoeffdingTree + ADWIN running on historical data"
git push
```

---

## Step 4 — Prove ADWIN Actually Detects Drift (Synthetic Injection Test)

This is the step that matters most. Running clean on real data proves nothing is broken. You need to prove the detector actually *fires when it should*.

New File → `tests/test_drift_injection.py`:
```python
# tests/test_drift_injection.py
"""
Deliberately feeds the online model a stable relationship for a while,
then flips the relationship (simulating real concept drift — e.g., a
change in hospital protocol that changes what vitals pattern predicts
sepsis), and asserts the ADWIN detector actually notices.
"""
import random
from ml.online.adwin_model import OnlineDriftSensor


def make_stable_sample(rng):
    """High heart rate + low BP => sepsis label 1. Stable rule."""
    hr = rng.uniform(60, 140)
    sbp = rng.uniform(70, 140)
    label = 1 if (hr > 100 and sbp < 100) else 0
    return {"heart_rate": hr, "sbp": sbp}, label


def make_drifted_sample(rng):
    """
    Same feature ranges, but the RULE that defines sepsis has flipped —
    now it's LOW heart rate + HIGH BP that signals risk. This is a
    concept drift injection: same feature distribution, different
    input-output relationship.
    """
    hr = rng.uniform(60, 140)
    sbp = rng.uniform(70, 140)
    label = 1 if (hr < 100 and sbp > 100) else 0
    return {"heart_rate": hr, "sbp": sbp}, label


def test_adwin_detects_injected_concept_drift():
    rng = random.Random(42)
    sensor = OnlineDriftSensor()

    # Phase 1: 500 samples of the STABLE relationship — let the model learn it
    for _ in range(500):
        x, y = make_stable_sample(rng)
        sensor.step(x, y)

    drift_before_injection = len(sensor.drift_events)

    # Phase 2: 500 samples of the DRIFTED relationship
    drift_found_in_phase_2 = False
    for _ in range(500):
        x, y = make_drifted_sample(rng)
        result = sensor.step(x, y)
        if result["drift_detected"]:
            drift_found_in_phase_2 = True

    assert drift_found_in_phase_2, (
        "ADWIN failed to detect an injected concept drift within 500 "
        "samples — either the detector is mis-configured or the "
        "injection isn't actually changing model error rate enough."
    )
    print(f"\nDrift events before injection: {drift_before_injection}")
    print(f"Drift correctly detected after injecting concept drift: {drift_found_in_phase_2}")
```

Run it:
```
pytest tests/test_drift_injection.py -v -s
```
(The `-s` flag shows the print output so you can see the actual counts, not just pass/fail.)

**If this test fails:** don't loosen the assertion to make it pass — that defeats the entire point. Instead, check: (1) is 500 samples actually enough for the HoeffdingTree to have "learned" the stable rule first — try increasing Phase 1 to 1000 samples; (2) is the drifted rule different enough to actually spike the error rate — a subtle drift is genuinely harder to detect, which is realistic, but for this proof-of-mechanism test we want an obvious flip. Debug it for real before moving on — a drift detector you haven't proven fires is a drift detector you can't trust in Milestone 2.

Commit once it passes:
```
git add .
git commit -m "Milestone 1b: verified ADWIN fires on injected concept drift"
git push
```

---

## Step 5 — Shadow Mode Against the Live Stream (Optional but Recommended)

This wires your online sensor into the actual Kafka pipeline from Sprint 1, running silently alongside your batch model — exactly as designed in the architecture spec.

New File → `ml/online/shadow_runner.py`:
```python
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
```

To test this, run your Sprint 1 three-terminal chain again (consumer, this shadow runner instead of `transforms.py`, and the simulator) — or run it alongside `transforms.py` in a fourth tab. Either way, this step is optional for Milestone 1's core proof — the synthetic injection test in Step 4 is the one that actually matters for your portfolio narrative. Don't burn a lot of time perfecting shadow-mode wiring if Step 4 already passes cleanly.

---

## Milestone 1 — Definition of Done
- [ ] `OnlineDriftSensor` runs cleanly against real historical data without crashing
- [ ] The synthetic injection test in Step 4 **passes for real** — ADWIN provably fires on an injected concept drift, not just in theory
- [ ] You understand and can explain, out loud, why the shadow-mode label problem (Step 5's proxy label) is a real limitation, not swept under the rug
- [ ] Everything committed and pushed

When Step 4's test is green, you're ready for Milestone 2: SWADT. That's where this drift signal gets fused with the SHAP importance trajectory from Sprint 2 into the actual urgency score from your paper. Ping me when you're there.
