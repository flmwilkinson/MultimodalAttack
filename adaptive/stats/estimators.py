# -*- coding: utf-8 -*-
"""
Point estimators and confidence intervals for attack-success-rate (ASR),
plus the Best-of-N power-law fit.

Why this matters for breach rate: an ASR of 3/10 is not "30%" -- it is
30% with a 95% Wilson interval of roughly [11%, 60%]. Reporting the
interval stops you chasing noise, and the power-law fit tells you whether
spending 10x more best-of-N samples will actually break through.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Callable

import numpy as np
from scipy import stats as _ss


# ---------------------------------------------------------------------------
# Confidence intervals for a binomial proportion (breach rate)
# ---------------------------------------------------------------------------

def wilson_interval(successes: int, n: int, z: float = 1.959963985) -> Tuple[float, float]:
    """Wilson score interval -- the right default for ASR (well-behaved at the
    extremes where breaches are rare and the normal approximation fails)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def clopper_pearson_interval(successes: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Clopper-Pearson (exact) interval -- conservative; use when you need a
    guaranteed coverage lower bound on ASR for a claim."""
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if successes == 0 else _ss.beta.ppf(alpha / 2, successes, n - successes + 1)
    hi = 1.0 if successes == n else _ss.beta.ppf(1 - alpha / 2, successes + 1, n - successes)
    return (float(lo), float(hi))


# ---------------------------------------------------------------------------
# pass@k -- unbiased estimator (Chen et al., 2021, "Evaluating LLMs on Code")
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of the probability that >=1 of k draws breaches,
    given c breaches observed in n>=k total samples.

    pass@k = 1 - C(n-c, k) / C(n, k), computed stably.
    """
    if k > n:
        raise ValueError("k must be <= n")
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    # Product form avoids overflow in the binomial coefficients.
    prob_none = 1.0
    for i in range(k):
        prob_none *= (n - c - i) / (n - i)
    return 1.0 - prob_none


# ---------------------------------------------------------------------------
# Best-of-N power-law fit (Hughes et al., 2024)
# ---------------------------------------------------------------------------

@dataclass
class PowerLawFit:
    """-log(ASR) = a * N^(-b)  <=>  log(-log(ASR)) = log(a) - b*log(N).

    Fit by OLS in log-log space. `predict_asr(N)` forecasts the ASR at a
    budget of N best-of-N samples; `n_for_asr(target)` inverts it to tell you
    how many samples you need to reach a target ASR.
    """
    a: float
    b: float
    r2: float
    n_points: int

    def predict_asr(self, n: int) -> float:
        if n <= 0:
            return 0.0
        neg_log_asr = self.a * (n ** (-self.b))
        return float(math.exp(-neg_log_asr))

    def n_for_asr(self, target_asr: float) -> float:
        target_asr = min(max(target_asr, 1e-6), 1 - 1e-6)
        neg_log = -math.log(target_asr)
        # neg_log = a * N^-b  =>  N = (a/neg_log)^(1/b)
        if self.b <= 0:
            return float("inf")
        return float((self.a / neg_log) ** (1.0 / self.b))


def fit_power_law(ns: Sequence[int], asrs: Sequence[float]) -> PowerLawFit | None:
    """Fit the BoN power law from observed (N, ASR) points. Needs >=2 points
    with 0 < ASR < 1. Returns None if it can't fit (e.g. all ASR==0 or ==1)."""
    xs, ys = [], []
    for n, asr in zip(ns, asrs):
        if n > 0 and 0.0 < asr < 1.0:
            xs.append(math.log(n))
            ys.append(math.log(-math.log(asr)))
    if len(xs) < 2:
        return None
    xs_a, ys_a = np.array(xs), np.array(ys)
    # log(-log(asr)) = log(a) - b*log(N)
    slope, intercept, r, _, _ = _ss.linregress(xs_a, ys_a)
    a = math.exp(intercept)
    b = -slope
    return PowerLawFit(a=a, b=b, r2=float(r * r), n_points=len(xs))
