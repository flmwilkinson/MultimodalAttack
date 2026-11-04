# -*- coding: utf-8 -*-
"""
Bayesian-optimization strategy over continuous COMET knobs.

Each iteration: the GP+EI optimizer suggests a continuous knob vector (format,
entanglement ratio, target temperature), we run COMET at those settings, score,
and feed the score back to the GP. Finds knob optima between the discrete
buckets the other strategies are limited to.
"""

from __future__ import annotations
import os
import time
import uuid
import datetime
from typing import List, TYPE_CHECKING

from adaptive.strategies.base import BaseStrategy

if TYPE_CHECKING:
    from adaptive.orchestrator import AttemptRecord


class BayesOptStrategy(BaseStrategy):
    name = "bayesopt"
    description = "Bayesian optimization (GP + Expected Improvement) over continuous knobs"

    def run(self) -> List["AttemptRecord"]:
        from adaptive.orchestrator import AttemptRecord
        from adaptive.bayesopt import BayesOpt, decode

        cfg = self.config
        budget = self.attempt_budget or cfg.max_attempts
        opt = BayesOpt(seed=cfg.seed)
        vcs = {"scene": "educational research laboratory", "target_group": "researchers"}
        print(f"\n{'='*70}")
        print(f"  BAYESIAN OPTIMIZATION over continuous knobs")
        print(f"  Objective: {cfg.objective[:80]}...")
        print(f"  Target: {cfg.target_model}  |  Budget: {budget}")
        print(f"{'='*70}\n")

        base = len(self.orch.attempts)
        for step in range(1, budget + 1):
            i = base + step
            x = opt.suggest()
            knobs = decode(x)
            attempt_dir = os.path.join(cfg.output_dir, f"bo_{i:03d}")

            t0 = time.time()
            pipe = self.orch._run_comet_pipeline(
                query=cfg.objective, attempt_dir=attempt_dir, entity_hints=None,
                output_format=knobs["output_format"], vcs_context=vcs,
                entanglement_strategy="balanced",
                entanglement_ratio=knobs["entanglement_ratio"],
            )
            if not pipe.get("success"):
                opt.observe(x, 0.0)
                continue
            sr = self.orch._send_and_score(
                pipe.get("chosen_text", ""), pipe.get("image_path"), cfg.objective,
                temperature=knobs["temperature"])

            rec = AttemptRecord(
                attempt_id=f"bo_{i}_{uuid.uuid4().hex[:8]}", attempt_number=i,
                strategy="bayesopt", timestamp=datetime.datetime.now().isoformat(),
                query_used=cfg.objective, entity_hints=None,
                output_format=knobs["output_format"], vcs_context=vcs,
                entanglement_strategy=f"ratio={knobs['entanglement_ratio']:.2f}",
                payload_text=pipe.get("chosen_text", ""), image_path=pipe.get("image_path"),
                target_response=sr.get("target_response", ""),
                score=sr.get("score"), refusal=sr.get("refusal"),
                convincingness=sr.get("convincingness"), specificity=sr.get("specificity"),
                indeterminate=sr.get("indeterminate", False),
            )
            rec.is_breach = (rec.score is not None and rec.score >= cfg.score_threshold
                             and rec.refusal == 0)
            if rec.is_breach and cfg.verify_breaches:
                rec.breach_verified = self.orch._verify_breach(
                    rec.payload_text, rec.image_path, cfg.objective, None)
                if not rec.breach_verified:
                    rec.is_breach = False
            rec.duration_seconds = time.time() - t0
            self.orch.attempts.append(rec)
            self.orch._save_attempt(rec, attempt_dir)
            self.orch.observe(rec, "bayesopt")

            opt.observe(x, rec.score if rec.score is not None else 0.0)
            prev = self.prev_score()
            print(record_line(rec, knobs, prev))

            if rec.is_breach:
                print(f"\n  >>> BREACH on attempt #{i}  (score={rec.score:.2f})")
                break

        bx, by = opt.best()
        if bx is not None:
            print(f"\n  Best knobs: {decode(bx)}  (score={by:.2f})")
        return self.orch.attempts


def record_line(rec, knobs, prev) -> str:
    base = rec.summary_line(prev)
    return base + f"  [fmt={knobs['output_format']} ratio={knobs['entanglement_ratio']:.2f} T={knobs['temperature']:.2f}]"
