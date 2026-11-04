"""
defenses/ — defensive tooling for auditing a VLM against the COMET attack class.

This package is deliberately *defensive*: it does not generate, strengthen, or
obfuscate attacks. It provides a pluggable guard interface and a test harness that
measure how well a guardrail stops the attacks that already exist, and at what cost
to benign traffic.

See literature/02-defenses.md for the defense families these reference guards
instantiate, and literature/03-evaluation-and-judges.md for the measurement caveats
the harness is built to respect.
"""

from .guards import Guard, GuardDecision, NullGuard, KeywordGuard, LLMClassifierGuard, build_guard

__all__ = [
    "Guard",
    "GuardDecision",
    "NullGuard",
    "KeywordGuard",
    "LLMClassifierGuard",
    "build_guard",
]
