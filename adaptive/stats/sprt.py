# -*- coding: utf-8 -*-
"""
Wald's Sequential Probability Ratio Test (SPRT).

Used to decide, as cheaply as possible, whether a config is "useful"
(per-attempt breach prob p >= p1) or "useless" (p <= p0). Instead of a fixed
number of attempts per config, keep sampling until the log-likelihood ratio
crosses a boundary, then stop and reallocate budget. On a stream of Bernoulli
breach outcomes this spends far fewer attempts on clearly-bad configs.
"""

from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class SPRT:
    """Sequential test of H0: p<=p0 vs H1: p>=p1 for a Bernoulli stream.

    alpha = P(accept H1 | H0 true), beta = P(accept H0 | H1 true).
    decision() returns 'continue', 'useful' (H1), or 'useless' (H0).
    """
    p0: float = 0.1          # "useless" breach prob
    p1: float = 0.5          # "useful" breach prob
    alpha: float = 0.05
    beta: float = 0.10
    llr: float = 0.0         # running log-likelihood ratio
    n: int = 0
    successes: int = 0

    def __post_init__(self):
        self._upper = math.log((1 - self.beta) / self.alpha)   # accept H1 above
        self._lower = math.log(self.beta / (1 - self.alpha))   # accept H0 below
        # Per-sample LLR increments.
        self._s1 = math.log(self.p1 / self.p0)                 # observed success
        self._s0 = math.log((1 - self.p1) / (1 - self.p0))     # observed failure

    def update(self, breached: bool) -> str:
        self.n += 1
        if breached:
            self.successes += 1
            self.llr += self._s1
        else:
            self.llr += self._s0
        return self.decision()

    def decision(self) -> str:
        if self.llr >= self._upper:
            return "useful"
        if self.llr <= self._lower:
            return "useless"
        return "continue"
