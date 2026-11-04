# -*- coding: utf-8 -*-
"""
Per-objective optimize strategy -- the unified learning controller in action.

Drives the persistent per-objective optimizer:
  (C) a strategy-portfolio bandit picks which attack strategy to run next for
      THIS objective (Thompson sampling, reward = verified breach / judge score),
  (D) every attempt feeds the experience bank (via the orchestrator hook), and
      priors transfer from similar objectives, and
  (E) multi-dimensional judge feedback steers the within-strategy refinement.

Budget is spent in chunks (chunked round-robin): run the bandit-chosen strategy
for `cfg.chunk` attempts, update the arm posterior from that chunk, re-select.
Each run on the same objective resumes a sharper posterior than the last.
"""

from __future__ import annotations
from typing import List, TYPE_CHECKING

from adaptive.strategies.base import BaseStrategy

if TYPE_CHECKING:
    from adaptive.orchestrator import AttemptRecord


class OptimizeStrategy(BaseStrategy):
    name = "optimize"
    description = "Per-objective strategy-portfolio bandit + experience bank + judge-feedback (unified)"

    def run(self) -> List["AttemptRecord"]:
        cfg = self.config
        opt = self.orch.objective_optimizer
        if opt is None:
            print("  [optimize requires per-objective learning; enabling a transient optimizer]")
            from adaptive.objective_optimizer import ObjectiveOptimizer
            opt = ObjectiveOptimizer(cfg.objective, cfg.target_model,
                                     score_threshold=cfg.score_threshold, seed=cfg.seed)
            self.orch.objective_optimizer = opt

        budget = cfg.max_attempts
        chunk = max(1, cfg.chunk)
        print(f"\n{'='*70}")
        print(f"  OPTIMIZE -- per-objective portfolio controller")
        print(f"  Objective: {cfg.objective[:80]}...")
        print(f"  Target: {cfg.target_model}  |  Budget: {budget}  |  Chunk: {chunk}")
        print(f"  Portfolio arms: {opt.portfolio}")
        print(f"  Resumed state: {opt.summary_line()}")
        print(f"{'='*70}\n")

        from adaptive.objective_optimizer import FULL_RUN_ARMS, CHUNKED_ARMS

        spent = 0
        while spent < budget:
            remaining = budget - spent
            # If little budget remains, only let chunked arms play so we don't
            # kick off a big structured run we can't afford.
            eligible = None if remaining >= 4 else CHUNKED_ARMS
            strat = opt.select_strategy(eligible=eligible)

            if strat in FULL_RUN_ARMS:
                # Full-run arm: one complete structured attack (sized by its own
                # config). It ignores the budget cap; we count what it produced.
                print(f"\n  [portfolio] -> {strat} (full run, spent {spent}/{budget})")
                new = self.orch.run_chunk(remaining, strategy_name=strat)
            else:
                this = min(chunk, remaining)
                print(f"\n  [portfolio] -> {strat} (+{this}, spent {spent}/{budget})")
                new = self.orch.run_chunk(this, strategy_name=strat)
            spent += len(new) if new else max(1, chunk)

            # Chunk reward = best outcome in the chunk (verified breach dominates).
            reward = 0.0
            breached = False
            for a in new:
                verified = a.is_breach and (a.breach_verified or not cfg.verify_breaches)
                reward = max(reward, 1.0 if verified else (a.score or 0.0))
                breached = breached or verified
            opt.update_strategy(strat, reward)
            print(f"  [portfolio] {strat} chunk reward={reward:.2f}  "
                  f"arm_means=[{', '.join(f'{s}={m:.2f}' for s, m in opt.strategy_means())}]")

            if breached:
                print(f"\n  >>> BREACH under strategy '{strat}'")
                break

        print(f"\n  Final per-objective state: {opt.summary_line()}")
        return self.orch.attempts
