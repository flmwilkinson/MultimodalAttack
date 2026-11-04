# -*- coding: utf-8 -*-
"""
Bayesian optimization over continuous COMET knobs.

The LLM-driven strategies pick from discrete buckets; BO treats the breach
score as a black-box f(x) over a CONTINUOUS knob vector and finds optima
between the buckets. Surrogate: a Gaussian Process (sklearn). Acquisition:
Expected Improvement, which balances exploiting the current best against
exploring high-uncertainty regions.

Knob vector x (all normalized to [0,1]):
    x[0] output_format   -> round to 1..8
    x[1] entanglement    -> continuous migration ratio in [0,1]
    x[2] target temperature in [0,1.5]

No botorch/cma dependency -- GP + EI in numpy/sklearn.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel


DIM = 3  # [format, entanglement_ratio, temperature]


def decode(x: np.ndarray) -> Dict:
    """Map a normalized [0,1]^DIM vector to COMET knob values."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    fmt = int(round(1 + x[0] * 7))               # 1..8
    ratio = float(x[1])                          # 0..1 entities migrated
    temperature = float(x[2] * 1.5)              # 0..1.5
    return {"output_format": fmt, "entanglement_ratio": ratio, "temperature": temperature}


class BayesOpt:
    """GP + Expected Improvement optimizer over [0,1]^DIM."""

    def __init__(self, seed: Optional[int] = None, n_candidates: int = 512):
        self.X: List[np.ndarray] = []
        self.y: List[float] = []
        self.rng = np.random.RandomState(seed)
        self.n_candidates = n_candidates
        kernel = (ConstantKernel(1.0, (1e-2, 1e2))
                  * Matern(length_scale=0.3, length_scale_bounds=(1e-2, 1e1), nu=2.5)
                  + WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-4, 1e0)))
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                           n_restarts_optimizer=2,
                                           random_state=seed)

    def suggest(self) -> np.ndarray:
        """Next point to evaluate. Random until we have a few observations,
        then Expected Improvement over a random candidate set."""
        if len(self.X) < 3:
            return self.rng.uniform(0, 1, size=DIM)
        self.gp.fit(np.array(self.X), np.array(self.y))
        best = max(self.y)
        cands = self.rng.uniform(0, 1, size=(self.n_candidates, DIM))
        mu, sigma = self.gp.predict(cands, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        imp = mu - best
        z = imp / sigma
        ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
        return cands[int(np.argmax(ei))]

    def observe(self, x: np.ndarray, score: float):
        self.X.append(np.asarray(x, dtype=float))
        self.y.append(float(score))

    def best(self) -> Tuple[Optional[np.ndarray], float]:
        if not self.y:
            return None, 0.0
        i = int(np.argmax(self.y))
        return self.X[i], self.y[i]
