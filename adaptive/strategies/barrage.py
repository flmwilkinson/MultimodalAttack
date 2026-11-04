# -*- coding: utf-8 -*-
"""
Barrage strategy — tournament selection with mutation.

Generates N variant configurations, evaluates them all, keeps the top-K
survivors, mutates them to fill the next generation, repeats.
"""

from __future__ import annotations
import json
from typing import List, Dict, Tuple, TYPE_CHECKING

from adaptive.strategies.base import BaseStrategy
from adaptive.prompts import BARRAGE_MUTATE_SYSTEM_PROMPT, BARRAGE_MUTATE_USER_TEMPLATE

if TYPE_CHECKING:
    from adaptive.orchestrator import AttemptRecord


class BarrageStrategy(BaseStrategy):
    name = "barrage"
    description = "Tournament selection with mutation"

    def run(self) -> List["AttemptRecord"]:
        cfg = self.config
        print(f"\n{'='*70}")
        print(f"  BARRAGE Tournament Selection")
        print(f"  Objective: {cfg.objective[:80]}...")
        print(f"  Target: {cfg.target_model}")
        print(f"  Pool: {cfg.pool_size} | Top-K: {cfg.top_k} | Generations: {cfg.generations}")
        print(f"{'='*70}\n")

        seeds = self._seed_population()
        attempt_counter = 0

        for gen in range(1, cfg.generations + 1):
            print(f"\n  --- Generation {gen}/{cfg.generations} ({len(seeds)} candidates) ---")
            gen_records: List[Tuple["AttemptRecord", Dict]] = []

            for params in seeds:
                if self.is_tried(params):
                    params = self._diversify(params)
                attempt_counter += 1
                prev = self.prev_score()
                record = self.run_attempt(attempt_counter, params)
                self.orch.attempts.append(record)
                gen_records.append((record, params))
                print(record.summary_line(prev))

                if record.is_breach:
                    print(f"\n  >>> BREACH on attempt #{attempt_counter}  (score={record.score:.2f})")
                    return self.orch.attempts

            gen_records.sort(key=lambda x: x[0].score or -1, reverse=True)
            survivors = gen_records[:cfg.top_k]
            print(f"  Survivors: {[f'#{r.attempt_number} (score={r.score:.2f})' for r, _ in survivors]}")

            if gen < cfg.generations:
                seeds = self._mutate_survivors(survivors, gen_records[cfg.top_k:])

        return self.orch.attempts

    def _diversify(self, params: Dict) -> Dict:
        """Nudge an already-tried config onto an untried format/entanglement combo."""
        formats = [4, 5, 6, 7, 8]
        entangles = ["balanced", "shift_to_image", "shift_to_text", "split_entities"]
        for ent in entangles:
            for fmt in formats:
                cand = dict(params)
                cand["output_format"] = fmt
                cand["entanglement_strategy"] = ent
                if not self.is_tried(cand):
                    return cand
        return params  # exhausted -- let it repeat rather than loop forever

    def _seed_population(self) -> List[Dict]:
        format_options = [4, 5, 6, 7, 8]
        entangle_options = ["balanced", "shift_to_image", "shift_to_text", "split_entities"]
        seeds = []
        for i in range(self.config.pool_size):
            seeds.append({
                "query_rewrite": self.config.objective,
                "entity_hints": None,
                "output_format": format_options[i % len(format_options)],
                "vcs_context": {"scene": "educational laboratory", "target_group": "researchers"},
                "entanglement_strategy": entangle_options[i % len(entangle_options)],
            })
        return seeds

    def _mutate_survivors(self, survivors: List[Tuple], eliminated: List[Tuple]) -> List[Dict]:
        cfg = self.config
        eliminated_lines = []
        for er, ep in eliminated:
            eliminated_lines.append(
                f"  score={er.score:.2f}, fmt={ep['output_format']}, "
                f"entangle={ep['entanglement_strategy']}, "
                f"query_start={ep['query_rewrite'][:60]}..."
            )
        eliminated_summary = "\n".join(eliminated_lines) if eliminated_lines else "(none)"

        seeds = []
        variants_each = max(1, cfg.pool_size // cfg.top_k)
        for record, params in survivors:
            mutate_prompt = BARRAGE_MUTATE_USER_TEMPLATE.format(
                objective=cfg.objective,
                base_score=record.score or 0,
                query_used=params["query_rewrite"],
                entity_hints=json.dumps(params["entity_hints"]),
                output_format=params["output_format"],
                vcs_context=json.dumps(params["vcs_context"]),
                entanglement_strategy=params["entanglement_strategy"],
                target_response=record.target_response[:400],
                num_variants=variants_each,
                eliminated_summary=eliminated_summary,
            )
            system = BARRAGE_MUTATE_SYSTEM_PROMPT.format(num_variants=variants_each)
            variants = self.call_refiner(system, mutate_prompt)
            if isinstance(variants, dict):
                variants = [variants]
            if not isinstance(variants, list):
                variants = [self.default_refinement()]
            for v in variants[:variants_each]:
                seeds.append({
                    "query_rewrite": v.get("query_rewrite", params["query_rewrite"]),
                    "entity_hints": v.get("entity_hints"),
                    "output_format": v.get("output_format", params["output_format"]),
                    "vcs_context": v.get("vcs_context", params["vcs_context"]),
                    "entanglement_strategy": v.get("entanglement_strategy", params["entanglement_strategy"]),
                })
        return seeds
