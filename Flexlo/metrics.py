"""
Evaluation helpers: SI-SNRi per speaker-count level, and stopping-classifier
precision/recall/accuracy. These are what your validation loop and final
report should be built around, since "SI-SNRi per speaker-count level" is
literally how your project will be graded.
"""

import numpy as np
from .losses import si_snr


def compute_si_snri(estimate, target, mixture):
    """SI-SNR improvement: how much better is the separated output than just
    handing back the raw mixture unchanged. estimate/target/mixture: [B,1,L]
    or [1,1,L]. Returns a python float (mean over batch)."""
    improvement = si_snr(estimate, target) - si_snr(mixture, target)
    return improvement.mean().item()


class StoppingStats:
    """Accumulates TP/FP/FN/TN for the stopping classifier across many
    recursion steps and mixtures, then reports precision/recall/accuracy.

    Label convention (matches recursion_labels in dynamic_mixer.py):
        True  = "a real speaker remains in the residual, keep recursing"
        False = "residual is empty/noise, stop"
    Positive class = True (keep going).
    """

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, predicted_bool, true_bool):
        if predicted_bool and true_bool:
            self.tp += 1
        elif predicted_bool and not true_bool:
            self.fp += 1
        elif not predicted_bool and true_bool:
            self.fn += 1
        else:
            self.tn += 1

    @property
    def accuracy(self):
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total > 0 else float("nan")

    @property
    def precision(self):
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else float("nan")

    @property
    def recall(self):
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else float("nan")

    @property
    def f1(self):
        p, r = self.precision, self.recall
        if np.isnan(p) or np.isnan(r) or (p + r) == 0:
            return float("nan")
        return 2 * p * r / (p + r)

    def summary(self):
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "n_samples": self.tp + self.fp + self.fn + self.tn,
        }


def summarize_si_snri_list(si_snri_values):
    arr = np.array(si_snri_values, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": len(arr)}
