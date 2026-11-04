# -*- coding: utf-8 -*-
"""
Score calibration (Platt scaling).

The judge's raw score is a noisy proxy for true harmfulness; the 0.5 threshold
is arbitrary. If you have even a small set of human-labeled (score -> breach)
pairs, fit a logistic map P(breach) = sigma(w*score + b) and threshold the
CALIBRATED probability instead. This aligns the decision boundary with human
judgment and reduces both false negatives (missed breaches) and false positives.

Without labels this is a no-op passthrough, so it is always safe to leave on.
"""

from __future__ import annotations
import json
import os
from typing import List, Optional, Tuple

import numpy as np


class ScoreCalibrator:
    def __init__(self):
        self.w: float = 1.0
        self.b: float = 0.0
        self.fitted: bool = False

    def fit(self, scores: List[float], labels: List[int], iters: int = 500, lr: float = 0.1):
        """Fit sigma(w*score + b) to (score, breach-label) pairs by gradient descent."""
        if len(scores) < 4 or len(set(labels)) < 2:
            return self
        x = np.asarray(scores, dtype=float)
        y = np.asarray(labels, dtype=float)
        w, b = 1.0, 0.0
        for _ in range(iters):
            z = w * x + b
            p = 1.0 / (1.0 + np.exp(-z))
            gw = np.mean((p - y) * x)
            gb = np.mean(p - y)
            w -= lr * gw
            b -= lr * gb
        self.w, self.b, self.fitted = float(w), float(b), True
        return self

    def probability(self, score: float) -> float:
        if not self.fitted:
            return float(score)   # passthrough: raw score as a pseudo-probability
        return float(1.0 / (1.0 + np.exp(-(self.w * score + self.b))))

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"w": self.w, "b": self.b, "fitted": self.fitted}, f)

    @classmethod
    def load(cls, path: str) -> "ScoreCalibrator":
        c = cls()
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                c.w, c.b, c.fitted = d["w"], d["b"], d["fitted"]
            except Exception:
                pass
        return c
