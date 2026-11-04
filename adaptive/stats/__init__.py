# -*- coding: utf-8 -*-
"""
Statistics layer for the adaptive orchestrator.

Turns the attack loop from "run attempts until budget runs out" into a
measured optimization: honest ASR estimation with confidence intervals,
best-of-N scaling-law forecasting, a learned breach-probability surrogate,
sequential testing for early config abandonment, and a suite-level budget
allocator that spends attempts where the marginal breach probability is
highest.
"""

from .estimators import (
    wilson_interval, clopper_pearson_interval, pass_at_k,
    fit_power_law, PowerLawFit,
)
from .sprt import SPRT
from .surrogate import BreachSurrogate
from .allocator import SuiteAllocator
from .calibration import ScoreCalibrator

__all__ = [
    "wilson_interval", "clopper_pearson_interval", "pass_at_k",
    "fit_power_law", "PowerLawFit",
    "SPRT", "BreachSurrogate", "SuiteAllocator", "ScoreCalibrator",
]
