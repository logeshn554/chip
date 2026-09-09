"""
Synthesis Evaluator — quality scoring for synthesis results.

Evaluates designs on:
- Area (cell count vs. reference)
- Timing (critical path vs. target)
- Resource utilization (LUTs, FFs, BRAMs)
"""

from __future__ import annotations

import logging
import math
from typing import Any

from agent.schemas import SynthesisResult, SynthesisScore

logger = logging.getLogger(__name__)


class SynthesisEvaluator:
    """Evaluates synthesis quality of hardware designs.

    Normalizes raw synthesis metrics (cell count, timing, etc.)
    into scores in the [0, 1] range using configurable references.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.area_max = config.get("area_max", 100000)
        self.timing_ns_max = config.get("timing_ns_max", 10.0)
        self.target_clock_ns = config.get("target_clock_ns", 5.0)

    def evaluate(self, synth_result: SynthesisResult) -> SynthesisScore:
        """Compute synthesis quality score.

        Args:
            synth_result: Raw synthesis result from Yosys

        Returns:
            SynthesisScore with area, timing, and power scores
        """
        score = SynthesisScore(synthesizable=synth_result.success)

        if not synth_result.success:
            logger.info("Synthesis failed — score = 0.0")
            return score

        # Area score: fewer cells = higher score
        # Uses exponential decay so very large designs still get some credit
        if synth_result.cell_count > 0:
            ratio = synth_result.cell_count / self.area_max
            score.area_score = max(0.0, math.exp(-ratio))
        else:
            score.area_score = 1.0

        # Timing score: faster critical path = higher score
        if synth_result.critical_path_ns > 0:
            if synth_result.critical_path_ns <= self.target_clock_ns:
                score.timing_score = 1.0
            else:
                # Linear decay above target
                slack = synth_result.critical_path_ns - self.target_clock_ns
                max_slack = self.timing_ns_max - self.target_clock_ns
                score.timing_score = max(0.0, 1.0 - (slack / max_slack))
        else:
            score.timing_score = 0.5  # Unknown timing

        # Power score: estimated from cell count (rough proxy)
        # Real power analysis would need a proper power tool
        power_proxy = synth_result.cell_count / self.area_max
        score.power_score = max(0.0, 1.0 - power_proxy)

        logger.info(
            f"Synthesis score: {score.score:.3f} "
            f"(area={score.area_score:.3f}, "
            f"timing={score.timing_score:.3f}, "
            f"power={score.power_score:.3f})"
        )

        return score
