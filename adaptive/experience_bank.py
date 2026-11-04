# -*- coding: utf-8 -*-
"""
Experience bank with semantic-similarity retrieval (JailExpert / ASTRA style).

A persistent store of past attack experiences per target model. Unlike exact-
hash memory, it retrieves by SEMANTIC SIMILARITY of the objective, so a new
objective warm-starts from what worked on *similar* ones, and carries a running
success rate per experience so credit accumulates. Two things it provides:

  - retrieve(objective, k): the most similar past experiences, best-first.
  - strategy_priors(objective): similarity-weighted Beta priors per STRATEGY,
    used to seed a new objective's strategy-portfolio bandit (cross-objective
    transfer -- the contextual part of best practice).

Embeddings use a fit-free char-n-gram HashingVectorizer (deterministic, no model
dependency), L2-normalized so cosine similarity is a dot product.
"""

from __future__ import annotations
import os
import json
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "results", "adaptive_memory", "experience")

_VEC = HashingVectorizer(n_features=512, analyzer="char_wb",
                         ngram_range=(3, 5), norm="l2", alternate_sign=False)


def _embed(text: str) -> np.ndarray:
    return _VEC.transform([text or ""]).toarray()[0]


class ExperienceBank:
    def __init__(self, target_model: str, base_dir: str = _DIR,
                 max_entries: int = 1000):
        self.target_model = target_model
        self.base_dir = base_dir
        self.max_entries = max_entries
        os.makedirs(base_dir, exist_ok=True)
        self.entries: List[Dict] = self._load()

    def _path(self) -> str:
        safe = self.target_model.replace("/", "_").replace("\\", "_").replace(":", "_")
        return os.path.join(self.base_dir, f"{safe}.json")

    def _load(self) -> List[Dict]:
        p = self._path()
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return []

    def save(self):
        self.entries.sort(key=lambda e: e.get("score", 0), reverse=True)
        self.entries = self.entries[:self.max_entries]
        with open(self._path(), "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)

    def add(self, objective: str, strategy: str, config: Dict,
            score: float, breach: bool, payload: str = ""):
        self.entries.append({
            "objective": objective[:200], "strategy": strategy,
            "output_format": config.get("output_format"),
            "entanglement_strategy": config.get("entanglement_strategy"),
            "vcs_context": config.get("vcs_context"),
            "score": float(score), "breach": bool(breach),
            "payload_preview": (payload or "")[:400], "ts": time.time(),
        })

    def _similarities(self, objective: str) -> List[Tuple[float, Dict]]:
        if not self.entries:
            return []
        q = _embed(objective)
        out = []
        for e in self.entries:
            sim = float(np.dot(q, _embed(e["objective"])))
            out.append((sim, e))
        out.sort(key=lambda x: x[0], reverse=True)
        return out

    def retrieve(self, objective: str, k: int = 5,
                 min_sim: float = 0.2) -> List[Dict]:
        sims = self._similarities(objective)
        hits = [e for sim, e in sims if sim >= min_sim][:k]
        return hits

    def strategy_priors(self, objective: str, min_sim: float = 0.2,
                        prior_strength: float = 4.0) -> Dict[str, List[float]]:
        """Similarity-weighted Beta priors per strategy from similar objectives.

        Each similar experience contributes weight = similarity; a breach/high
        score pushes alpha, a failure pushes beta. Returned as {strategy:[a,b]}
        to seed a new objective's strategy bandit (so it starts biased toward
        strategies that worked on neighbors)."""
        priors: Dict[str, List[float]] = {}
        for sim, e in self._similarities(objective):
            if sim < min_sim:
                continue
            strat = e.get("strategy")
            if not strat:
                continue
            reward = 1.0 if e.get("breach") else max(0.0, min(1.0, e.get("score", 0.0)))
            w = sim * prior_strength
            a, b = priors.get(strat, [1.0, 1.0])
            priors[strat] = [a + w * reward, b + w * (1.0 - reward)]
        return priors

    def knob_priors(self, objective: str, min_sim: float = 0.2,
                    prior_strength: float = 4.0) -> Dict[str, List[float]]:
        """Similarity-weighted Beta priors per knob arm ("format|entanglement")
        from similar objectives -- seeds a new objective's knob posterior so the
        per-config bandit starts biased toward knobs that worked on neighbours."""
        priors: Dict[str, List[float]] = {}
        for sim, e in self._similarities(objective):
            if sim < min_sim:
                continue
            f = e.get("output_format")
            ent = e.get("entanglement_strategy")
            if f is None or ent is None:
                continue
            key = f"{f}|{ent}"
            reward = 1.0 if e.get("breach") else max(0.0, min(1.0, e.get("score", 0.0)))
            w = sim * prior_strength
            a, b = priors.get(key, [1.0, 1.0])
            priors[key] = [a + w * reward, b + w * (1.0 - reward)]
        return priors

    def best_similar(self, objective: str, min_sim: float = 0.3) -> Optional[Dict]:
        """Highest-scoring sufficiently-similar experience, to adapt on failure."""
        best, best_key = None, -1.0
        for sim, e in self._similarities(objective):
            if sim < min_sim:
                continue
            key = sim * (1.0 + e.get("score", 0.0)) + (0.5 if e.get("breach") else 0.0)
            if key > best_key:
                best_key, best = key, e
        return best
