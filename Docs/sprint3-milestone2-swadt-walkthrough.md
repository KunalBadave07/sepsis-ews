# Sprint 3, Milestone 2 — SWADT: Making Your Paper Real
### Every file, every command, in order.

This is the part of the whole project that's actually, provably yours. Go slow. Read the code, don't just paste it.

---

## Step 1 — Create the SWADT module

New File → `monitoring/retrain_trigger/swadt.py`:

```python
# monitoring/retrain_trigger/swadt.py
"""
Implements the SWADT algorithm from the paper:
- S_d(t): per-feature drift statistic (PSI between a fixed reference
  window and the current rolling window)
- Δφ_d(t): directional change in that feature's SHAP importance
- U_d(t) = S_d(t) * (1 + λ * max(0, Δφ_d(t))): per-feature urgency
- T(t): importance-weighted aggregate trigger score
- τ(t): adaptive threshold (EWMA + k * EWSTD of trigger score history)
"""
import numpy as np


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two samples of the same feature."""
    breakpoints = np.quantile(reference, np.linspace(0, 1, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    ref_pct = np.clip(ref_counts / max(len(reference), 1), 1e-4, None)
    cur_pct = np.clip(cur_counts / max(len(current), 1), 1e-4, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


class SWADT:
    def __init__(self, feature_names: list[str], window_size: int = 200,
                 lam: float = 1.5, k: float = 2.5, psi_cap: float = 1.0):
        self.feature_names = feature_names
        self.window_size = window_size
        self.lam = lam            # SHAP-velocity sensitivity coefficient
        self.k = k                # threshold width (in std-devs)
        self.psi_cap = psi_cap    # normalize raw PSI into roughly [0,1]

        self.reference_buffer = {f: [] for f in feature_names}
        self.current_buffer = {f: [] for f in feature_names}
        self.shap_buffer = {f: [] for f in feature_names}
        self.prev_shap_importance = {f: 0.0 for f in feature_names}

        self.reference_ready = False
        self.trigger_score_history: list[float] = []

    def observe(self, feature_values: dict, shap_abs_values: dict):
        """
        Call this once per incoming sample. Returns None most of the time
        (still filling a window); returns a result dict once per completed
        window comparison.
        """
        for f in self.feature_names:
            self.current_buffer[f].append(feature_values[f])
            self.shap_buffer[f].append(abs(shap_abs_values.get(f, 0.0)))

        window_full = all(len(v) >= self.window_size for v in self.current_buffer.values())
        if not window_full:
            return None

        if not self.reference_ready:
            # first full window becomes the permanent baseline distribution
            self.reference_buffer = {f: list(vals) for f, vals in self.current_buffer.items()}
            self._reset_current_window()
            self.reference_ready = True
            return None

        result = self._evaluate_window()
        self._reset_current_window()
        return result

    def _reset_current_window(self):
        self.current_buffer = {f: [] for f in self.feature_names}
        self.shap_buffer = {f: [] for f in self.feature_names}

    def _evaluate_window(self) -> dict:
        urgencies, importances = {}, {}

        for f in self.feature_names:
            ref = np.array(self.reference_buffer[f])
            cur = np.array(self.current_buffer[f])

            raw_psi = compute_psi(ref, cur)
            s_d = float(np.clip(raw_psi / self.psi_cap, 0.0, 1.0))

            phi_now = float(np.mean(self.shap_buffer[f])) if self.shap_buffer[f] else 0.0
            phi_prev = self.prev_shap_importance[f]
            delta_phi = (phi_now - phi_prev) / (phi_prev + 1e-6)

            urgencies[f] = s_d * (1 + self.lam * max(0.0, delta_phi))
            importances[f] = phi_now
            self.prev_shap_importance[f] = phi_now

        total_importance = sum(importances.values()) + 1e-6
        weights = {f: importances[f] / total_importance for f in self.feature_names}

        trigger_score = sum(weights[f] * urgencies[f] for f in self.feature_names)

        if len(self.trigger_score_history) >= 5:
            arr = np.array(self.trigger_score_history)
            ewma, ewstd = arr.mean(), (arr.std() or 0.01)
        else:
            ewma, ewstd = 0.0, 0.5  # generous default until enough history exists

        threshold = ewma + self.k * ewstd
        triggered = trigger_score > threshold

        self.trigger_score_history.append(trigger_score)
        if len(self.trigger_score_history) > 50:
            self.trigger_score_history.pop(0)

        top_contributors = sorted(
            [(f, weights[f] * urgencies[f]) for f in self.feature_names],
            key=lambda pair: pair[1], reverse=True,
        )[:3]

        return {
            "trigger_score": trigger_score,
            "threshold": threshold,
            "triggered": triggered,
            "top_contributors": top_contributors,
        }
```

---

## Step 2 — THE test that actually validates your paper

This is the one test in this entire project you should be proudest of. It doesn't just check the code runs — it proves the core thesis: **SWADT should barely react when an unimportant feature drifts, and should react strongly when an important feature drifts by the same amount.**

New File → `tests/test_swadt.py`:
```python
# tests/test_swadt.py
import numpy as np
from monitoring.retrain_trigger.swadt import SWADT

FEATURES = ["important_feature", "unimportant_feature"]


def run_scenario(drift_which: str, window_size=200, n_windows=4, drift_magnitude=4.0):
    """
    drift_which: 'important' or 'unimportant' — which feature actually
    shifts distribution starting from window 2 onward. The other feature
    stays stable throughout.

    SHAP importance is fixed and fed manually: 'important_feature' always
    reports a high mean |SHAP| value, 'unimportant_feature' always reports
    a near-zero one — simulating a feature the model barely relies on.
    """
    rng = np.random.default_rng(42)
    swadt = SWADT(FEATURES, window_size=window_size, lam=1.5, k=2.0, psi_cap=0.6)

    results = []
    for w in range(n_windows):
        for _ in range(window_size):
            important_val = rng.normal(0, 1)
            unimportant_val = rng.normal(0, 1)

            if w >= 1:  # from the 2nd window onward, inject drift
                if drift_which == "important":
                    important_val += drift_magnitude
                else:
                    unimportant_val += drift_magnitude

            shap_vals = {
                "important_feature": 0.8,   # consistently high importance
                "unimportant_feature": 0.02,  # consistently near-zero importance
            }
            feature_vals = {
                "important_feature": important_val,
                "unimportant_feature": unimportant_val,
            }

            result = swadt.observe(feature_vals, shap_vals)
            if result is not None:
                results.append(result)

    return results


def test_swadt_reacts_more_to_important_feature_drift():
    important_drift_results = run_scenario("important")
    unimportant_drift_results = run_scenario("unimportant")

    # compare the trigger score in the FIRST post-drift window (index 0,
    # since index -1 in reference-building isn't counted — the first
    # returned result IS the first real comparison window)
    important_score = important_drift_results[0]["trigger_score"]
    unimportant_score = unimportant_drift_results[0]["trigger_score"]

    print(f"\nTrigger score when IMPORTANT feature drifts:   {important_score:.4f} "
          f"(triggered={important_drift_results[0]['triggered']})")
    print(f"Trigger score when UNIMPORTANT feature drifts: {unimportant_score:.4f} "
          f"(triggered={unimportant_drift_results[0]['triggered']})")

    assert important_score > unimportant_score, (
        "SWADT should weight drift in an important feature more heavily "
        "than identical-magnitude drift in an unimportant feature — this "
        "is the entire point of the mechanism. If this fails, the "
        "importance-weighting logic in _evaluate_window is broken."
    )

    # the stronger claim: importance-weighted drift should actually cross
    # the adaptive threshold, while unimportant drift should be suppressed
    assert important_drift_results[0]["triggered"] is True, (
        "Expected the important-feature drift scenario to fire the trigger."
    )
```

Run it:
```
pytest tests/test_swadt.py -v -s
```

**Read the printed scores, don't just check the pass/fail.** You should see the "important feature drifts" trigger score noticeably higher than the "unimportant feature drifts" one — and ideally the important one crosses its threshold while the unimportant one doesn't. If both assertions pass but the printed scores are nearly identical, the mechanism technically passes but isn't demonstrating much — try increasing `lam` (the SHAP-velocity sensitivity) or making the importance gap between the two features starker (e.g., 0.9 vs 0.01) and rerun until the separation is convincing, not just barely-there.

**If it fails:** the most common cause is `psi_cap` being set too high or too low for your synthetic drift magnitude, which flattens `s_d` toward 0 or 1 for both features, erasing the distinction the importance-weighting is supposed to create. Print the raw `s_d` value per feature inside `_evaluate_window` temporarily to see what's actually happening before changing constants blindly.

Commit once this passes convincingly:
```
git add .
git commit -m "Milestone 2: SWADT implemented, proven importance-aware via differential drift test"
git push
```

---

## Step 3 — Wire SWADT to Real Data (Sprint 1 features + Sprint 2 SHAP)

Now connect it to the real pipeline instead of synthetic data.

New File → `monitoring/retrain_trigger/run_swadt_live.py`:
```python
# monitoring/retrain_trigger/run_swadt_live.py
"""
Replays historical feature data through SWADT using REAL SHAP importances
from your Sprint 2 explainer, instead of synthetic fixed values.
"""
import pandas as pd
from monitoring.retrain_trigger.swadt import SWADT
from ml.explainability.shap_pipeline import SepsisExplainer

FEATURE_COLS = [
    "heart_rate", "resp_rate", "sbp", "map_bp", "temp_c", "spo2",
    "hr_rolling_mean", "hr_rolling_std", "map_rolling_mean",
    "map_rolling_std", "shock_index",
]
DATA_PATH = "data/processed/training_features.parquet"


def main():
    df = pd.read_parquet(DATA_PATH).dropna(subset=FEATURE_COLS)
    df = df.sort_values(["patient_id", "ICULOS"]).head(5000)  # keep it fast for a first real run

    explainer = SepsisExplainer()
    swadt = SWADT(FEATURE_COLS, window_size=250, lam=1.5, k=2.5, psi_cap=1.0)

    trigger_count = 0
    for i, row in df.iterrows():
        feature_vals = {col: row[col] for col in FEATURE_COLS}

        explanation = explainer.explain_row(feature_vals, top_n=len(FEATURE_COLS))
        shap_map = {item["feature"]: abs(item["shap_value"]) for item in explanation["top_features"]}

        result = swadt.observe(feature_vals, shap_map)
        if result and result["triggered"]:
            trigger_count += 1
            print(f"[SWADT TRIGGER] score={result['trigger_score']:.3f} "
                  f"threshold={result['threshold']:.3f} "
                  f"top_contributors={result['top_contributors']}")

    print(f"\nTotal SWADT triggers over {len(df)} real samples: {trigger_count}")


if __name__ == "__main__":
    main()
```

Run it:
```
python monitoring/retrain_trigger/run_swadt_live.py
```
This will run slower than the synthetic test (computing real SHAP values per row is the expensive part) — that's expected and fine for a one-time validation run, not something you'd do at full production throughput without batching.

**What you're checking:** does it run to completion, and when a trigger does fire, do the `top_contributors` it names make clinical sense (features like `map_bp` or `shock_index`, not something random like a feature that shouldn't matter much)? If the top contributors consistently look arbitrary, that's worth a closer look before trusting this on real data.

Commit:
```
git add .
git commit -m "Milestone 2: SWADT wired to real SHAP pipeline and historical data"
git push
```

---

## Milestone 2 — Definition of Done
- [ ] `SWADT` class implemented with real PSI, urgency, aggregate score, and adaptive threshold logic
- [ ] `test_swadt.py` passes, **and the printed scores show a real, convincing separation** between important-feature drift and unimportant-feature drift — not just a technical pass
- [ ] SWADT runs against real historical data + your actual SHAP pipeline without crashing
- [ ] Any real triggers that fire point to clinically sensible top-contributor features
- [ ] Everything committed and pushed

Once this is solid, you're ready for Milestone 3: wrapping this whole thing in the guardrailed FastAPI layer — auth, the acknowledgment endpoint, the audit log, and the schema that structurally cannot contain a treatment recommendation. Ping me when you're there.
