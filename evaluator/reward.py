"""
Reward Engine — Grounded hardware evaluation and scoring.

Supports:
1. Version 1 Grounded Formula (milestone specification):
   - compile_success = 1
   - all_functional_tests_pass = 5
   - synthesis_success = 1
   - lint_clean = 1
   Total initial reward: R = compile + functional + synthesis + lint (Max = 8.0)

2. Multi-Objective / Phased scoring:
   - Phase 1: Correctness only (compile + tests)
   - Phase 2: Correctness + synthesis (area + timing + power)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class GroundedRewardResult:
    """Detailed breakdown of tool-grounded reward calculation."""
    compile_score: float  # 0 or 1
    functional_score: float  # 0 to 5
    synthesis_score: float  # 0 or 1
    lint_score: float  # 0 or 1
    total_reward: float  # R = compile + functional + synthesis + lint
    normalized_reward: float  # 0.0 to 1.0
    is_valid_hardware: bool
    breakdown: dict[str, Any] = field(default_factory=dict)


class RewardEngine:
    """Computes grounded reward strictly derived from EDA tool outputs."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        mode: str = "v1",
        weights: Optional[dict[str, float]] = None,
    ):
        if isinstance(config, dict):
            self.phase = config.get("reward_phase", 1)
            raw_weights = config.get("reward_weights", {})
        else:
            self.phase = 1
            raw_weights = {}

        self.mode = mode
        self.weights = weights or {
            "correctness": raw_weights.get("correctness", 0.5),
            "area": raw_weights.get("area", 0.2),
            "timing": raw_weights.get("timing", 0.2),
            "power": raw_weights.get("power", 0.1),
            "resource_efficiency": raw_weights.get("resource_efficiency", 0.05),
        }
        self.w_correctness = self.weights.get("correctness", 0.5)
        self.w_area = self.weights.get("area", 0.2)
        self.w_timing = self.weights.get("timing", 0.2)
        self.w_power = self.weights.get("power", 0.1)

    def set_phase(self, phase: int) -> None:
        """Set the reward phase (1: correctness only, 2: correctness + synthesis)."""
        self.phase = phase

    def compute_v1_reward(
        self,
        compile_success: bool,
        all_functional_tests_pass: bool,
        synthesis_success: bool,
        lint_clean: bool,
    ) -> GroundedRewardResult:
        """Version 1 grounded reward: R = compile + functional + synthesis + lint."""
        c = 1.0 if compile_success else 0.0
        f = 5.0 if all_functional_tests_pass else 0.0
        s = 1.0 if (synthesis_success and all_functional_tests_pass) else (0.5 if synthesis_success else 0.0)
        l = 1.0 if lint_clean else 0.0

        is_valid = bool(compile_success and all_functional_tests_pass)
        if not is_valid:
            f = 0.0
            s = min(s, 0.5)

        total = c + f + s + l
        normalized = total / 8.0

        breakdown = {
            "compile_success": c,
            "all_functional_tests_pass": f,
            "synthesis_success": s,
            "lint_clean": l,
            "compile": c,
            "functional": f,
            "synthesis": s,
            "lint": l,
            "formula": "R = compile(1) + functional(5) + synthesis(1) + lint(1)",
        }

        return GroundedRewardResult(
            compile_score=c,
            functional_score=f,
            synthesis_score=s,
            lint_score=l,
            total_reward=round(total, 4),
            normalized_reward=round(normalized, 4),
            is_valid_hardware=is_valid,
            breakdown=breakdown,
        )

    def compute(self, functional: Any, synthesis: Any = None) -> Any:
        """Compute evaluation score for Phase 1 / Phase 2 dataclass inputs."""
        from agent.schemas import EvaluationResult, FunctionalScore, SynthesisScore

        compile_pass = getattr(functional, "compile_pass", False) if functional else False
        lint_pass = getattr(functional, "lint_pass", False) if functional else False
        test_pass_rate = getattr(functional, "test_pass_rate", 0.0) if functional else 0.0
        tests_passed = getattr(functional, "tests_passed", 0) if functional else 0
        tests_total = getattr(functional, "tests_total", 0) if functional else 0

        # If empty functional score
        if not compile_pass and tests_total == 0 and test_pass_rate == 0.0:
            return EvaluationResult(
                reward=0.0,
                functional=functional or FunctionalScore(),
                synthesis=synthesis or SynthesisScore(),
                reward_breakdown={"compile": 0.0, "functional": 0.0},
            )

        # Correctness score [0, 1]
        correctness = 0.0
        if compile_pass:
            correctness += 0.3
        if lint_pass:
            correctness += 0.1
        correctness += 0.6 * test_pass_rate

        breakdown: dict[str, float] = {
            "compile": 0.3 if compile_pass else 0.0,
            "lint": 0.1 if lint_pass else 0.0,
            "tests": 0.6 * test_pass_rate,
            "correctness": correctness,
        }

        if self.phase == 1 or synthesis is None or not getattr(synthesis, "synthesizable", False):
            reward = correctness
        else:
            area_sc = getattr(synthesis, "area_score", 0.5)
            timing_sc = getattr(synthesis, "timing_score", 0.5)
            power_sc = getattr(synthesis, "power_score", 0.5)

            breakdown["area"] = self.w_area * area_sc
            breakdown["timing"] = self.w_timing * timing_sc
            breakdown["power"] = self.w_power * power_sc

            reward = (
                self.w_correctness * correctness
                + self.w_area * area_sc
                + self.w_timing * timing_sc
                + self.w_power * power_sc
            )

        reward = max(0.0, min(1.0, reward))

        return EvaluationResult(
            reward=round(reward, 4),
            functional=functional or FunctionalScore(),
            synthesis=synthesis or SynthesisScore(),
            reward_breakdown=breakdown,
        )
