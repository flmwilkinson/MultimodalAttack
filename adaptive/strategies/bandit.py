# -*- coding: utf-8 -*-
"""
Bandit strategy -- Thompson-sampling optimizer over the knob space.

A non-LLM optimizer: each attempt, Thompson-sample an (output_format,
entanglement) arm, run COMET, and update the arm's Beta posterior with the
StrongREJECT score. The posterior persists per target model, so the bandit
gets sharper every run. Complements the LLM-driven strategies -- use it to
find the best delivery knobs cheaply, then let PAIR/Tree refine the content.
"""

from __future__ import annotations
from typing import List, TYPE_CHECKING

from adaptive.strategies.base import BaseStrategy

if TYPE_CHECKING:
    from adaptive.orchestrator import AttemptRecord


class BanditStrategy(BaseStrategy):
    name = "bandit"
    description = "Thompson-sampling bandit over format x entanglement arms (persisted per target)"

    def run(self) -> List["AttemptRecord"]:
        cfg = self.config
        from adaptive.bandit import StructuredThompsonBandit

        bandit = StructuredThompsonBandit(cfg.target_model, seed=cfg.seed)
        print(f"\n{'='*70}")
        print(f"  BANDIT Knob Optimizer (structured/correlated Thompson sampling)")
        print(f"  Objective: {cfg.objective[:80]}...")
        print(f"  Target: {cfg.target_model}  |  Budget: {cfg.max_attempts} attempts")
        print(f"  {bandit.summary()}")
        print(f"{'='*70}\n")

        # A single scene is held fixed so the arms isolate format x entanglement.
        vcs = {"scene": "educational research laboratory", "target_group": "researchers"}

        for i in range(1, cfg.max_attempts + 1):
            fmt, ent = bandit.select()
            params = {
                "query_rewrite": cfg.objective,
                "entity_hints": None,
                "output_format": fmt,
                "vcs_context": vcs,
                "entanglement_strategy": ent,
            }
            prev = self.prev_score()
            record = self.run_attempt(i, params)
            self.orch.attempts.append(record)

            reward = record.score if record.score is not None else 0.0
            bandit.update(fmt, ent, reward)
            print(record.summary_line(prev))

            if record.is_breach:
                print(f"\n  >>> BREACH on attempt #{i}  (score={record.score:.2f})")
                break

        bandit.save()
        print(f"\n  {bandit.summary()}")
        return self.orch.attempts
