# -*- coding: utf-8 -*-
"""
Adaptive Multimodal Attack Orchestrator
=======================================

Self-contained adaptive loop for cross-modal adversarial testing.
Combines COMET's entanglement payload generation with iterative refinement
strategies drawn from PAIR, TAP, Crescendo, MAPA, TreeTeaming, and MAMJ
research (2023-2026).

This module is the multimodal-native alternative to PyRIT's text-only
adaptive orchestrators. It generates, sends, scores, and refines
image+text payloads in a closed loop.

Strategies:
    pair      -- Single-turn refinement (Chao et al. 2023)
    crescendo -- Multi-turn escalation (Russinovich et al. 2024)
    barrage   -- Tournament selection with mutation
    mapa      -- Intra-turn text/image alternation (Choi et al. 2026)
    tree      -- Hierarchical strategy tree (Li et al. CVPR 2026)
    meta      -- Meta-adaptive strategy prompt refinement (Cui et al. EMNLP 2026)

All transcripts are emitted in the schema consumed by robustness_eval.py,
so adaptive results flow into the same measurement harness as everything else.
"""

from .orchestrator import AdaptiveOrchestrator, AttemptRecord, AdaptiveConfig
from .memory_store import MemoryStore

__all__ = ["AdaptiveOrchestrator", "AttemptRecord", "AdaptiveConfig", "MemoryStore"]
