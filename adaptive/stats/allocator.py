# -*- coding: utf-8 -*-
"""
Suite-level budget allocator.

Given a set of objectives and a fixed TOTAL attempt budget, uniform allocation
is statistically wasteful: it keeps hammering hopeless objectives and
under-funds the ones that are one tweak from breaking. This allocator maintains
a Beta posterior over each objective's per-attempt breach probability (updated
with the continuous StrongREJECT score, so near-misses count), and always hands
the next attempt to the objective with the highest MARGINAL expected value:

    value(obj) = [objective not yet breached] * E[p_breach | history]

Already-breached objectives drop out. Hopeless ones (posterior mass near 0)
starve. The near-misses get fed. At a fixed total cost this maximizes the
expected number of distinct objectives breached.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class _ObjState:
    objective: str
    alpha: float = 1.0       # Beta prior (uniform)
    beta: float = 1.0
    attempts: int = 0
    breached: bool = False
    best_score: float = 0.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


class SuiteAllocator:
    """Marginal-value scheduler over a suite of objectives."""

    def __init__(self, objectives: List[str], total_budget: int,
                 min_per_objective: int = 1):
        self.total_budget = total_budget
        self.spent = 0
        self.min_per_objective = min_per_objective
        self.state: Dict[str, _ObjState] = {o: _ObjState(objective=o) for o in objectives}

    def _value(self, st: _ObjState) -> float:
        if st.breached:
            return -1.0
        # Guarantee a minimum probe per objective before letting the posterior
        # starve it (otherwise the uniform prior can be unlucky on attempt 1).
        if st.attempts < self.min_per_objective:
            return 10.0 + st.mean
        return st.mean

    def next_objective(self) -> Optional[str]:
        if self.spent >= self.total_budget:
            return None
        candidates = [st for st in self.state.values() if not st.breached]
        if not candidates:
            return None
        best = max(candidates, key=self._value)
        if self._value(best) < 0:
            return None
        return best.objective

    def update(self, objective: str, score: Optional[float], breached: bool):
        st = self.state[objective]
        st.attempts += 1
        self.spent += 1
        s = 0.0 if score is None else max(0.0, min(1.0, float(score)))
        # Continuous Beta update: near-misses nudge the posterior up.
        st.alpha += s
        st.beta += (1.0 - s)
        st.best_score = max(st.best_score, s)
        if breached:
            st.breached = True

    def report(self) -> List[Dict]:
        rows = []
        for st in self.state.values():
            lo = st.alpha / (st.alpha + st.beta)
            rows.append({
                "objective": st.objective[:60],
                "attempts": st.attempts,
                "breached": st.breached,
                "posterior_mean": round(lo, 3),
                "best_score": round(st.best_score, 3),
            })
        rows.sort(key=lambda r: (r["breached"], r["posterior_mean"]), reverse=True)
        return rows
