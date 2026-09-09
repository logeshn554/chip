"""
Reward Engine — Grounded hardware evaluation and scoring.

Supports:
1. Version 1 Grounded Formula (milestone specification):
   - compile_success = 1
   - all_functional_tests_pass = 5
   - synthesis_success = 1
   - lint_clean = 1
   Total initial reward: R = compile + functional + synthesis + lint (Max = 8.0)

2. Modular / Multi-Objective Normalized Scoring:
   - R = w_correctness * R_correctness + w_formal * R_formal + w_area * R_area + w_timing * R_timing + w_power * R_power
   - Strict Hard Gates:
     * If compilation fails: R = 0.0 (quality = 0.0)
     * If mandatory functional tests fail: design cannot be considered successful (R_area = 0, R_timing = 0)
     * If formal verification fails: design fails safety gate
     * If metrics unavailable: never invent fake numbers
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class GroundedRewardResult:
    """Detailed breakdown of tool-grounded reward calculation."""
    compile_score: float       # 0 or 1
    functional_score: float    # 0 to 5
    synthesis_score: float     # 0 or 1
    lint_score: float          # 0 or 1
    formal_score: float = 0.0  # 0 or 1
    total_reward: float = 0.0  # R = compile + functional + synthesis + lint (+ formal)
    normalized_reward: float = 0.0  # 0.0 to 1.0
    is_valid_hardware: bool = False
    breakdown: dict[str, Any] = field(default_factory=dict)


class RewardEngine:
    """Computes grounded rewards strictly derived from EDA and verification tool outputs."""

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
            "correctness": raw_weights.get("correctness", 0.45),
            "formal": raw_weights.get("formal", 0.15),
            "area": raw_weights.get("area", 0.20),
            "timing": raw_weights.get("timing", 0.15),
            "power": raw_weights.get("power", 0.05),
        }

        # Normalize weights so sum == 1.0
        total_w = sum(self.weights.values())
        if total_w > 0:
            self.weights = {k: v / total_w for k, v in self.weights.items()}

        self.w_correctness = self.weights.get("correctness", 0.45)
        self.w_formal = self.weights.get("formal", 0.15)
        self.w_area = self.weights.get("area", 0.20)
        self.w_timing = self.weights.get("timing", 0.15)
        self.w_power = self.weights.get("power", 0.05)

    def set_phase(self, phase: int) -> None:
        """Set the reward phase (1: correctness only, 2: correctness + synthesis)."""
        self.phase = phase

    def compute_v1_reward(
        self,
        compile_success: bool,
        all_functional_tests_pass: bool,
        synthesis_success: bool,
        lint_clean: bool,
        formal_pass: Optional[bool] = None,
    ) -> GroundedRewardResult:
        """Version 1 grounded reward: R = compile + functional + synthesis + lint.

        Strict Hard Gate:
        - If compile_success is False, functional and synthesis rewards are clamped to 0.
        - If all_functional_tests_pass is False, hardware is invalid (synthesis reward capped).
        """
        c = 1.0 if compile_success else 0.0
        f = 5.0 if (compile_success and all_functional_tests_pass) else 0.0
        s = 1.0 if (compile_success and all_functional_tests_pass and synthesis_success) else (0.5 if synthesis_success else 0.0)
        l = 1.0 if (compile_success and lint_clean) else 0.0
        formal_sc = 1.0 if formal_pass is True else (0.0 if formal_pass is False else 0.0)

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
            "formal_pass": formal_sc if formal_pass is not None else "SKIPPED",
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
            formal_score=formal_sc,
            total_reward=round(total, 4),
            normalized_reward=round(normalized, 4),
            is_valid_hardware=is_valid,
            breakdown=breakdown,
        )

    def compute_modular_reward(
        self,
        compile_success: bool,
        test_pass_rate: float,
        formal_status: str = "SKIPPED",  # "PASS", "FAIL", "SKIPPED"
        area: Optional[float] = None,
        area_target: float = 500.0,
        timing_ns: Optional[float] = None,
        timing_target: float = 5.0,
        power_uw: Optional[float] = None,
    ) -> GroundedRewardResult:
        """Modular multi-objective reward with normalized weights and strict hard gates.

        Formula:
        R = w_correctness * R_correctness + w_formal * R_formal + w_area * R_area + w_timing * R_timing + w_power * R_power
        """
        # Hard Gate 1: Compile check
        if not compile_success:
            return GroundedRewardResult(
                compile_score=0.0,
                functional_score=0.0,
                synthesis_score=0.0,
                lint_score=0.0,
                formal_score=0.0,
                total_reward=0.0,
                normalized_reward=0.0,
                is_valid_hardware=False,
                breakdown={"gate_failed": "Compilation failed. Design quality = 0.0"},
            )

        # Hard Gate 2: Mandatory functional pass
        if test_pass_rate < 1.0:
            r_corr = round(0.4 * test_pass_rate, 4)
            return GroundedRewardResult(
                compile_score=1.0,
                functional_score=r_corr * 5.0,
                synthesis_score=0.0,
                lint_score=1.0,
                formal_score=0.0,
                total_reward=r_corr,
                normalized_reward=r_corr,
                is_valid_hardware=False,
                breakdown={"gate_failed": "Mandatory functional tests failed. No synthesis/area reward credited."},
            )

        # Correctness is 1.0
        r_correctness = 1.0

        # Formal verification score
        if formal_status == "PASS":
            r_formal = 1.0
        elif formal_status == "FAIL":
            # Safety violation: fail hard gate
            return GroundedRewardResult(
                compile_score=1.0,
                functional_score=3.0,
                synthesis_score=0.0,
                lint_score=1.0,
                formal_score=0.0,
                total_reward=0.3,
                normalized_reward=0.3,
                is_valid_hardware=False,
                breakdown={"gate_failed": "Formal assertion violation detected by SymbiYosys."},
            )
        else:
            # SKIPPED or UNAVAILABLE: do not penalize, use neutral 1.0 for weight re-normalization
            r_formal = 1.0

        # Area score: smaller is better (bounded in [0, 1])
        if area is not None and area > 0:
            r_area = max(0.0, min(1.0, area_target / area))
        else:
            r_area = 1.0  # neutral if unavailable (do not invent fake area)

        # Timing score: lower critical path is better
        if timing_ns is not None and timing_ns > 0:
            r_timing = max(0.0, min(1.0, timing_target / timing_ns))
        else:
            r_timing = 1.0

        # Power score: do not invent fake numbers
        r_power = 1.0

        total = (
            self.w_correctness * r_correctness
            + self.w_formal * r_formal
            + self.w_area * r_area
            + self.w_timing * r_timing
            + self.w_power * r_power
        )

        breakdown = {
            "w_correctness": self.w_correctness,
            "w_formal": self.w_formal,
            "w_area": self.w_area,
            "w_timing": self.w_timing,
            "w_power": self.w_power,
            "r_correctness": r_correctness,
            "r_formal": r_formal,
            "r_area": r_area,
            "r_timing": r_timing,
            "area_metric": area,
            "timing_metric": timing_ns,
            "power_metric": power_uw,
        }

        return GroundedRewardResult(
            compile_score=1.0,
            functional_score=5.0,
            synthesis_score=1.0,
            lint_score=1.0,
            formal_score=r_formal,
            total_reward=round(total * 8.0, 4),
            normalized_reward=round(total, 4),
            is_valid_hardware=True,
            breakdown=breakdown,
        )

    def compute(self, functional: Any, synthesis: Any = None) -> Any:
        """Compute evaluation score for Phase 1 / Phase 2 dataclass inputs."""
        from agent.schemas import EvaluationResult, FunctionalScore, SynthesisScore

        compile_pass = getattr(functional, "compile_pass", False) if functional else False
        lint_pass = getattr(functional, "lint_pass", False) if functional else False
        test_pass_rate = getattr(functional, "test_pass_rate", 0.0) if functional else 0.0
        tests_total = getattr(functional, "tests_total", 0) if functional else 0

        # If empty functional score
        if not compile_pass and tests_total == 0 and test_pass_rate == 0.0:
            return EvaluationResult(
                reward=0.0,
                functional=functional or FunctionalScore(),
                synthesis=synthesis or SynthesisScore(),
                reward_breakdown={"compile": 0.0, "functional": 0.0},
            )

        # Hard Gate: If compilation failed, reward is 0.0
        if not compile_pass:
            return EvaluationResult(
                reward=0.0,
                functional=functional or FunctionalScore(),
                synthesis=synthesis or SynthesisScore(),
                reward_breakdown={"compile": 0.0, "reason": "Compilation failed"},
            )

        correctness = 0.3 * (1.0 if compile_pass else 0.0)
        correctness += 0.1 * (1.0 if lint_pass else 0.0)
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
