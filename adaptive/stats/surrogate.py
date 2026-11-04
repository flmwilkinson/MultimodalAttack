# -*- coding: utf-8 -*-
"""
Online logistic surrogate: P(breach | config), trained on the run's own data.

Replaces "ask the LLM which knobs to try" with a data-driven acquisition
function. After each attempt we update the model with (config features ->
breached?), then score all candidate configs and pick the one that maximizes
predicted breach probability plus an exploration bonus. This is a cheap,
interpretable surrogate; features are PRE-observation (known before we spend a
target call), so it can actually steer the next choice.
"""

from __future__ import annotations
import hashlib
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import SGDClassifier


FORMATS = [1, 2, 3, 4, 5, 6, 7, 8]
ENTANGLES = ["balanced", "shift_to_image", "shift_to_text", "split_entities"]
_SCENE_BUCKETS = 8


def _scene_hash(vcs: Optional[Dict]) -> int:
    if not vcs:
        return 0
    s = (vcs.get("scene") or "") + "|" + (vcs.get("target_group") or "")
    if not s.strip():
        return 0
    h = int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16)
    return h % _SCENE_BUCKETS


def featurize(params: Dict) -> np.ndarray:
    """Fixed-length pre-observation feature vector for a config."""
    fmt = params.get("output_format", 5)
    ent = params.get("entanglement_strategy", "balanced")
    vcs = params.get("vcs_context")
    hints = params.get("entity_hints") or []
    query = params.get("query_rewrite", "") or ""

    v = np.zeros(len(FORMATS) + len(ENTANGLES) + _SCENE_BUCKETS + 2, dtype=float)
    off = 0
    try:
        v[off + FORMATS.index(int(fmt))] = 1.0
    except (ValueError, TypeError):
        pass
    off += len(FORMATS)
    if ent in ENTANGLES:
        v[off + ENTANGLES.index(ent)] = 1.0
    off += len(ENTANGLES)
    v[off + _scene_hash(vcs)] = 1.0
    off += _SCENE_BUCKETS
    v[off] = min(len(hints), 5) / 5.0           # entity-hint count (capped)
    v[off + 1] = min(len(query), 2000) / 2000.0  # query length (normalized)
    return v


class BreachSurrogate:
    """Online logistic P(breach | config) with an exploration bonus."""

    def __init__(self, explore_coef: float = 0.25, base_rate: float = 0.15,
                 seed: Optional[int] = None):
        self.model = SGDClassifier(loss="log_loss", alpha=1e-4,
                                   random_state=seed, warm_start=True)
        self.explore_coef = explore_coef
        self.base_rate = base_rate
        self._classes = np.array([0, 1])
        self._fitted = False
        self._seen_classes = set()
        self._counts: Dict[bytes, int] = {}

    def update(self, params: Dict, breached: bool):
        x = featurize(params).reshape(1, -1)
        y = np.array([1 if breached else 0])
        self.model.partial_fit(x, y, classes=self._classes)
        self._fitted = True
        self._seen_classes.add(int(y[0]))
        key = featurize(params).tobytes()
        self._counts[key] = self._counts.get(key, 0) + 1

    def predict_proba(self, params: Dict) -> float:
        # Need both classes before the logistic has a meaningful decision boundary.
        if not self._fitted or len(self._seen_classes) < 2:
            return self.base_rate
        x = featurize(params).reshape(1, -1)
        p = self.model.predict_proba(x)[0]
        idx = list(self.model.classes_).index(1)
        return float(p[idx])

    def acquisition(self, params: Dict) -> float:
        """Predicted breach prob + UCB-style exploration bonus (rarely-tried
        configs get a boost so we don't collapse onto one region too early)."""
        key = featurize(params).tobytes()
        n = self._counts.get(key, 0)
        bonus = self.explore_coef / np.sqrt(1.0 + n)
        return self.predict_proba(params) + bonus

    def rank(self, candidates: List[Dict]) -> List[Dict]:
        """Candidates sorted by acquisition, best first."""
        return sorted(candidates, key=self.acquisition, reverse=True)

    def best(self, candidates: List[Dict]) -> Dict:
        return max(candidates, key=self.acquisition)
