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
    