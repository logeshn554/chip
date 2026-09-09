"""
Reward Engine — computes the reward signal for the agent.

Combines functional correctness and synthesis quality scores
into a single reward value using configurable weights.
Supports two phases:
  Phase 1: Correctness only (compile + tests)
  Phase 2: Correctness + synthesis (area + timing + power)
"""

from __future__ import annotations

import logging
from typing import Any

from agent.schemas import EvaluationResult, FunctionalScore, SynthesisScore

logger = logging.getLogger(__name__)


class RewardEngine:
    """Computes the reward signal for the self-evolving agent.

    The reward function evolves with the agent:
    - Phase 1: R = R_compile + R_functional
    - Phase 2: R = w1*R_correctness + w2*R_area + w3*R_timing + w4*R_power

    This lets the agent move from "correct hardware" to "better hardware".
    """

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}

        # Reward weights
        weights = config.get("reward_weights", {})
        self.w_correctness = weights.get("correctness", 0.5)
        self.w_area = weights.get("area", 0.2)
        self.w_timing = weights.get("timing", 0.2)
        self.w_power = weights.get("power", 0.1)

        # Phase (1 = correctness only, 2 = + synthesis)
        self.phase = config.get("reward_phase", 1)

        logger.info(f"RewardEngine: phase={self.phase}, weights=({self.w_correctness}, {self.w_area}, {self.w_timing}, {self.w_power})")

    def compute(
        self,
        functional: FunctionalScore,
        synthesis: SynthesisScore | None = None,
    ) -> EvaluationResult:
        """Compute the reward for a design evaluation.

        Args:
            functional: Functional correctness score
            synthesis: Optional synthesis quality score (Phase 2+)

        Returns:
            EvaluationResult with reward and breakdown
        """
        breakdown = {}

        if self.phase == 1:
            # Phase 1: Correctness only
            reward = functional.score
            breakdown = {
                "compile": 0.3 if functional.compile_pass else 0.0,
                "lint": 0.1 if functional.lint_pass else 0.0,
                "tests": 0.6 * functional.test_pass_rate,
                "total": reward,
            }

        else:
            # Phase 2: Correctness + Synthesis
            r_correctness = functional.score

            if synthesis and synthesis.synthesizable:
                r_area = synthesis.area_score
                r_timing = synthesis.timing_score
                r_power = synthesis.power_score
            else:
                r_area = 0.0
                r_timing = 0.0
                r_power = 0.0

            reward = (
                self.w_correctness * r_correctness
                + self.w_area * r_area
                + self.w_timing * r_timing
                + self.w_power * r_power
            )

            breakdown = {
                "correctness": self.w_correctness * r_correctness,
                "area": self.w_area * r_area,
                "timing": self.w_timing * r_timing,
                "power": self.w_power * r_power,
                "total": reward,
            }

        # Apply reward shaping
        reward = self._shape_reward(reward, functional, synthesis)

        result = EvaluationResult(
            functional=functional,
            synthesis=synthesis or SynthesisScore(),
            reward=reward,
            reward_breakdown=breakdown,
        )

        logger.info(f"Reward: {reward:.3f} (breakdown: {breakdown})")
        return result

    def _shape_reward(
        self,
        base_reward: float,
        functional: FunctionalScore,
        synthesis: SynthesisScore | None,
    ) -> float:
        """Apply reward shaping for better learning signals.

        - Partial credit for partial compilation
        - Bonus for improving test pass rate
        - Penalty for excessive warnings
        """
        reward = base_reward

        # Bonus: if all tests pass, small bonus
        if functional.test_pass_rate >= 1.0 and functional.compile_pass:
            reward = min(1.0, reward + 0.05)

        # Penalty: excessive lint warnings
        if functional.lint_warnings > 10:
            reward *= 0.95

        # Ensure reward stays in [0, 1]
        reward = max(0.0, min(1.0, reward))

        return reward

    def set_phase(self, phase: int) -> None:
        """Switch the reward phase.

        Phase 1: Correctness only
        Phase 2: Correctness + synthesis
        """
        self.phase = phase
        logger.info(f"Reward phase set to {phase}")
