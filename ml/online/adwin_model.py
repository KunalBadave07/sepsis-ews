# ml/online/adwin_model.py — UPDATED
from river import tree, drift, metrics, imblearn


class OnlineDriftSensor:
    def __init__(self):
        # RandomUnderSampler under-samples the majority class (0) ON THE FLY
        # so the learner actually sees a roughly balanced stream, instead
        # of collapsing to "always predict no sepsis."
        self.model = imblearn.RandomUnderSampler(
            classifier=tree.HoeffdingTreeClassifier(),
            desired_dist={0: 0.5, 1: 0.5},
            seed=42,
        )
        self.adwin = drift.ADWIN()
        self.accuracy = metrics.Accuracy()
        self.recall = metrics.Recall()
        self.precision = metrics.Precision()
        self.n_seen = 0
        self.drift_events = []

    def step(self, x: dict, y_true: int) -> dict:
        y_pred = self.model.predict_one(x)
        if y_pred is None:
            y_pred = 0

        error = 0 if y_pred == y_true else 1
        self.accuracy.update(y_true, y_pred)
        self.recall.update(y_true, y_pred)
        self.precision.update(y_true, y_pred)

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
            "running_recall": self.recall.get(),
            "running_precision": self.precision.get(),
            "drift_detected": drifted,
        }