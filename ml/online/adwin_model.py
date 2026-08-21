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