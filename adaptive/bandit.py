# -*- coding: utf-8 -*-
"""
Thompson-sampling bandit over the discrete COMET knob space.

The knobs (output_format x entanglement_strategy) form a finite set of arms.
A Beta-Bernoulli Thompson sampler learns which arms breach this target most
reliably -- faster and more principled than asking the LLM to pick -- and the
posterior persists per target model so learning carries across runs.

Continuous reward: the StrongREJECT score in [0,1] updates a Beta posterior as
alpha += score, beta += (1 - score). Sampling theta ~ Beta(alpha, beta) and
taking the argmax balances exploration and exploitation automatically.
"""

from __future__ import annotations
import os
import json
import random
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger("adaptive.bandit")

DEFAULT_BANDIT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "adaptive_memory", "bandit"
)

FORMATS = [4, 5, 6, 7, 8]
ENTANGLES = ["balanced", "shift_to_image", "shift_to_text", "split_entities"]


class KnobBandit:
    """Beta-Bernoulli Thompson sampler over (output_format, entanglement) arms."""

    def __init__(self, target_model: str, bandit_dir: str = DEFAULT_BANDIT_DIR,
                 seed: Optional[int] = None,
                 formats: Optional[List[int]] = None,
                 entangles: Optional[List[str]] = None):
        self.target_model = target_model
        self.bandit_dir = bandit_dir
        self.formats = formats or FORMATS
        self.entangles = entangles or ENTANGLES
        self._rng = random.Random(seed)
        os.makedirs(bandit_dir, exist_ok=True)
        self.posterior: Dict[str, List[float]] = self._load()

    # -- arm encoding --------------------------------------------------------

    @staticmethod
    def arm_key(output_format: int, entanglement: str) -> str:
        return f"{output_format}|{entanglement}"

    def _all_arms(self) -> List[Tuple[int, str]]:
        return [(f, e) for f in self.formats for e in self.entangles]

    # -- persistence ---------------------------------------------------------

    def _path(self) -> str:
        safe = self.target_model.replace("/", "_").replace("\\", "_").replace(":", "_")
        return os.path.join(self.bandit_dir, f"{safe}.json")

    def _load(self) -> Dict[str, List[float]]:
        p = self._path()
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def save(self):
        with open(self._path(), "w", encoding="utf-8") as f:
            json.dump(self.posterior, f, ensure_ascii=False, indent=2)

    # -- bandit core ---------------------------------------------------------

    def _ab(self, key: str) -> Tuple[float, float]:
        a, b = self.posterior.get(key, [1.0, 1.0])
        return a, b

    def select(self) -> Tuple[int, str]:
        """Thompson sampling: draw theta per arm, pick the argmax."""
        best_arm, best_theta = None, -1.0
        for f, e in self._all_arms():
            a, b = self._ab(self.arm_key(f, e))
            theta = self._rng.betavariate(a, b)
            if theta > best_theta:
                best_theta, best_arm = theta, (f, e)
        return best_arm

    def update(self, output_format: int, entanglement: str, reward: float):
        """Update the arm's Beta posterior with a continuous reward in [0,1]."""
        reward = max(0.0, min(1.0, float(reward)))
        key = self.arm_key(output_format, entanglement)
        a, b = self._ab(key)
        self.posterior[key] = [a + reward, b + (1.0 - reward)]

    def best_arms(self, k: int = 3) -> List[Tuple[str, float]]:
        """Top arms by posterior mean alpha/(alpha+beta), for reporting."""
        scored = []
        for f, e in self._all_arms():
            a, b = self._ab(self.arm_key(f, e))
            scored.append((self.arm_key(f, e), a / (a + b)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def summary(self) -> str:
        pulls = sum((a - 1) + (b - 1) for a, b in (self._ab(k) for k in self.posterior))
        top = self.best_arms(3)
        line = ", ".join(f"{k} (mean={m:.2f})" for k, m in top)
        return f"bandit[{self.target_model}] ~{int(pulls)} effective pulls | top: {line}"


class StructuredThompsonBandit(KnobBandit):
    """Correlated-arm Thompson sampling.

    Independent Beta arms waste samples because the arms share structure:
    format 7 and 8 are both adversarial-enhanced; the four entanglement
    strategies are related. After a pull we also apply a damped update to
    NEIGHBOURING arms (same format OR same entanglement), so a win on one arm
    informs its neighbours and the posterior converges in fewer pulls.
    """

    def __init__(self, *args, neighbor_weight: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.neighbor_weight = neighbor_weight

    def update(self, output_format: int, entanglement: str, reward: float):
        super().update(output_format, entanglement, reward)
        reward = max(0.0, min(1.0, float(reward)))
        w = self.neighbor_weight
        for f, e in self._all_arms():
            if (f == output_format) == (e == entanglement):
                continue  # skip self and fully-unrelated arms
            if f == output_format or e == entanglement:  # shares one dimension
                key = self.arm_key(f, e)
                a, b = self._ab(key)
                self.posterior[key] = [a + w * reward, b + w * (1.0 - reward)]


class LinUCBBandit:
    """Contextual linear bandit (LinUCB, Li et al. 2010).

    For the SUITE / cross-objective setting: the context x is a feature vector
    of the objective (so what works on one objective transfers to similar ones).
    Each arm keeps A = I + sum(x x^T) and b = sum(reward x); it picks the arm
    maximizing the upper confidence bound theta^T x + alpha * sqrt(x^T A^-1 x).
    """

    def __init__(self, arms: List[Tuple[int, str]], context_dim: int,
                 alpha: float = 1.0, seed: Optional[int] = None):
        import numpy as np
        self.arms = arms
        self.d = context_dim
        self.alpha = alpha
        self.A = {self._k(a): np.identity(context_dim) for a in arms}
        self.b = {self._k(a): np.zeros(context_dim) for a in arms}

    @staticmethod
    def _k(arm: Tuple[int, str]) -> str:
        return f"{arm[0]}|{arm[1]}"

    def select(self, context):
        import numpy as np
        x = np.asarray(context, dtype=float).reshape(-1)
        best_arm, best_ucb = self.arms[0], -1e9
        for arm in self.arms:
            k = self._k(arm)
            A_inv = np.linalg.inv(self.A[k])
            theta = A_inv @ self.b[k]
            ucb = float(theta @ x + self.alpha * np.sqrt(x @ A_inv @ x))
            if ucb > best_ucb:
                best_ucb, best_arm = ucb, arm
        return best_arm

    def update(self, arm: Tuple[int, str], context, reward: float):
        import numpy as np
        x = np.asarray(context, dtype=float).reshape(-1)
        k = self._k(arm)
        self.A[k] += np.outer(x, x)
        self.b[k] += max(0.0, min(1.0, float(reward))) * x
