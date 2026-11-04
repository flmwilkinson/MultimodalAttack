# -*- coding: utf-8 -*-
"""
PAIR strategy — iterative single-turn refinement.

Based on Chao et al. 2023, adapted for multimodal: the refinement LLM
adjusts COMET pipeline knobs (entity selection, entanglement ratio,
output format) rather than just rewriting text.
"""

from __future__ import annotations
import json
from typing import List, Dict, TYPE_CHECKING

from adaptive.strategies.base import BaseStrategy
from adaptive.prompts import PAIR_SYSTEM_PROMPT, PAIR_USER_TEMPLATE

if TYPE_CHECKING:
    from adaptive.orchestrator import AttemptRecord


class PairStrategy(BaseStrategy):
    name = "pair"
    description = "PAIR-style iterative single-turn refinement (Chao et al. 2023)"

    def run(self) -> List["AttemptRecord"]:
        cfg = self.config
        budget = self.attempt_budget or cfg.max_attempts
        system_prompt = self.system_prompt_override or PAIR_SYSTEM_PROMPT
        print(f"\n{'='*70}")
        print(f"  PAIR Adaptive Loop")
        print(f"  Objective: {cfg.objective[:80]}...")
        print(f"  Target: {cfg.target_model}  |  Budget: {budget} attempts")
        print(f"{'='*70}\n")

        # Warm-start from cross-objective memory if available, else defaults.
        params = self.orch.warm_start_params() or {
            "query_rewrite": cfg.objective,
            "entity_hints": None,
            "output_format": 5,
            "vcs_context": {"scene": "educational laboratory", "target_group": "researchers"},
            "entanglement_strategy": "balanced",
        }
        # If the run specifies the objective's true target entity, force COMET
        # into specified-entity mode on it so the riddle is built to uniquely
        # identify that compound (rather than letting COMET hide generic decoys
        # the target resolves to the wrong thing).
        seed_hints = getattr(cfg, "seed_entity_hints", None)
        if seed_hints and not params.get("entity_hints"):
            params["entity_hints"] = list(seed_hints)
            print(f"  [Specified-entity mode: {seed_hints}]")
        if self.orch.warm_start_configs:
            print(f"  [Warm-started from memory: fmt={params['output_format']} "
                  f"entangle={params['entanglement_strategy']}]")

        # Offset numbering by attempts already run (so Meta's repeated PAIR
        # rounds get continuous, non-colliding attempt numbers / output dirs).
        base = len(self.orch.attempts)
        for i in range(1, budget + 1):
            n = base + i
            prev = self.prev_score()
            record = self.run_attempt(n, params)
            self.orch.attempts.append(record)
            print(record.summary_line(prev))

            if record.is_breach:
                print(f"\n  >>> BREACH on attempt #{n}  (score={record.score:.2f})")
                break

            if i < budget:
                memory = self.learning_memory()
                user_prompt = PAIR_USER_TEMPLATE.format(
                    objective=cfg.objective,
                    attempt_number=n,
                    query_used=params["query_rewrite"],
                    entity_hints=json.dumps(params["entity_hints"]),
                    output_format=params["output_format"],
                    vcs_context=json.dumps(params["vcs_context"]),
                    entanglement_strategy=params["entanglement_strategy"],
                    target_response=record.target_response[:500],
                    score=f"{record.score:.2f}" if record.score is not None else "n/a",
                    refusal=record.refusal,
                    convincingness=f"{record.convincingness:.2f}" if record.convincingness is not None else "n/a",
                    specificity=f"{record.specificity:.2f}" if record.specificity is not None else "n/a",
                    learning_memory=memory,
                    next_attempt=n + 1,
                )
                refinement = self.call_refiner(system_prompt, user_prompt)

                best = self.best_attempt()
                baseline = {
                    "query_rewrite": best.query_used if best else params["query_rewrite"],
                    "entity_hints": best.entity_hints if best else params["entity_hints"],
                    "output_format": best.output_format if best else params["output_format"],
                    "vcs_context": best.vcs_context if best else params["vcs_context"],
                    "entanglement_strategy": best.entanglement_strategy if best else params["entanglement_strategy"],
                }

                params = {
                    "query_rewrite": refinement.get("query_rewrite", baseline["query_rewrite"]),
                    "entity_hints": refinement.get("entity_hints", baseline["entity_hints"]),
                    "output_format": refinement.get("output_format", baseline["output_format"]),
                    "vcs_context": refinement.get("vcs_context", baseline["vcs_context"]),
                    "entanglement_strategy": refinement.get("entanglement_strategy", baseline["entanglement_strategy"]),
                }
                # Keep the seeded target entity sticky: if the refiner nulled it,
                # restore it so every attempt keeps probing the real compound.
                if seed_hints and not params.get("entity_hints"):
                    params["entity_hints"] = list(seed_hints)
                record.refinement_reasoning = refinement.get("analysis", "")
                confidence = refinement.get("confidence", "?")
                print(f"         Refiner: {refinement.get('analysis', '')[:90]} (conf={confidence})")

        return self.orch.attempts
