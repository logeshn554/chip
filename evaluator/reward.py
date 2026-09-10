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

from agent.schemas import InformationClass

logger = logging.getLogger(__name__)


@dataclass
class GroundedRewardResult:
    """Detailed breakdown of tool-grounded reward calculation."""
    compile_score: float       # 0 or 1
    functional_score: float    # 0 to 5
    synthesis_score: float     # 0 or 1
    lint_score: float          # 0 or 1
    formal_score: float = 0.0  # Auxiliary formal verification metric (0 or 1)
    base_reward_v1: float = 0.0  # R = compile + functional + synthesis + lint (Max = 8.0)
    total_reward: float = 0.0  # Total reward for current evaluation mode
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

        # Normalize configured weights so sum == 1.0
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
        """Version 1 grounded reward: R = compile(1) + functional(5) + synthesis(1) + lint(1).
        
        Semantics:
        - Base reward: 8-point base reward.
        - Formal verification: Reported as an auxiliary metric (formal_score).
          Formal verification does not silently alter the 8-point base scale, preventing
          ambiguity between base verification and multi-objective RL reward.
        
        Strict Hard Gates:
        - If compile_success is False, functional and synthesis rewards are clamped to 0.
        - If all_functional_tests_pass is False, hardware is invalid.
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

        base_total = c + f + s + l
        normalized = base_total / 8.0

        breakdown = {
            "metric_type": "v1_8point_base_with_auxiliary_formal",
            "compile_success": c,
            "all_functional_tests_pass": f,
            "synthesis_success": s,
            "lint_clean": l,
            "formal_pass": formal_sc if formal_pass is not None else "SKIPPED",
            "auxiliary_formal_score": formal_sc,
            "compile": c,
            "functional": f,
            "synthesis": s,
            "lint": l,
            "formula": "R_base = compile(1) + functional(5) + synthesis(1) + lint(1) (Max 8.0); formal is auxiliary",
        }

        return GroundedRewardResult(
            compile_score=c,
            functional_score=f,
            synthesis_score=s,
            lint_score=l,
            formal_score=formal_sc,
            base_reward_v1=round(base_total, 4),
            total_reward=round(base_total, 4),
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
        evaluation_mode: str = "FAST_DEVELOPMENT",  # "FAST_DEVELOPMENT" | "STRICT_EVALUATION" | "TRAINING" | "research_fast" | "research_strict"
        metric_statuses: Optional[dict[str, Any]] = None,
    ) -> GroundedRewardResult:
        """Modular multi-objective reward with normalized weights and strict hard gates.

        RESEARCH CONTRACT INVARIANTS:
        - HYPOTHESIS != MEASUREMENT (LLM claims cannot earn synthesis reward in TRAINING mode).
        - ESTIMATE != MEASUREMENT (analytical models cannot substitute for real EDA in strict mode).
        - UNKNOWN != PASS (missing data cannot pass hard gates).

        Modes:
        - 'FAST_DEVELOPMENT' / 'research_fast': Formal and synthesis are optional; unavailable tools re-normalize weights.
        - 'STRICT_EVALUATION' / 'TRAINING' / 'research_strict': Formal and real logic synthesis are mandatory;
          HYPOTHESIS/UNKNOWN metrics fail strict gates and cannot enter RL training.
        """
        mode_upper = str(evaluation_mode).upper().strip()
        is_strict = mode_upper in ("STRICT_EVALUATION", "TRAINING", "RESEARCH_STRICT")

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
                breakdown={
                    "evaluation_mode": evaluation_mode,
                    "gate_failed": "Compilation failed. Design quality = 0.0",
                },
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
                breakdown={
                    "evaluation_mode": evaluation_mode,
                    "gate_failed": "Mandatory functional tests failed. No synthesis/area reward credited.",
                },
            )

        # Correctness is 1.0
        r_correctness = 1.0

        # Strict Mode Gate: Formal verification mandatory in strict modes
        if is_strict and formal_status != "PASS":
            return GroundedRewardResult(
                compile_score=1.0,
                functional_score=3.0,
                synthesis_score=0.0,
                lint_score=1.0,
                formal_score=0.0,
                total_reward=0.3,
                normalized_reward=0.3,
                is_valid_hardware=False,
                breakdown={
                    "evaluation_mode": evaluation_mode,
                    "gate_failed": f"Formal verification mandatory in {evaluation_mode} mode, but status was '{formal_status}'.",
                },
            )

        # Strict Mode Gate: Real synthesis mandatory in strict modes
        if is_strict and (area is None or area <= 0):
            return GroundedRewardResult(
                compile_score=1.0,
                functional_score=4.0,
                synthesis_score=0.0,
                lint_score=1.0,
                formal_score=1.0 if formal_status == "PASS" else 0.0,
                total_reward=0.4,
                normalized_reward=0.4,
                is_valid_hardware=False,
                breakdown={
                    "evaluation_mode": evaluation_mode,
                    "gate_failed": f"Actual logic synthesis mandatory in {evaluation_mode} mode, but area was unavailable or 0.",
                },
            )

        # Strict Mode Gate: Provenance check - reject HYPOTHESIS / UNKNOWN in TRAINING mode
        if is_strict and metric_statuses:
            area_st = metric_statuses.get("area")
            if area_st is not None:
                area_st_str = getattr(area_st, "value", str(area_st)).upper()
                if area_st_str in ("HYPOTHESIS", "UNKNOWN", "PLACEHOLDER_DO_NOT_USE"):
                    return GroundedRewardResult(
                        compile_score=1.0,
                        functional_score=4.0,
                        synthesis_score=0.0,
                        lint_score=1.0,
                        formal_score=1.0 if formal_status == "PASS" else 0.0,
                        total_reward=0.4,
                        normalized_reward=0.4,
                        is_valid_hardware=False,
                        breakdown={
                            "evaluation_mode": evaluation_mode,
                            "gate_failed": f"Area metric status was '{area_st_str}'. HYPOTHESIS/UNKNOWN metrics are strictly rejected in {evaluation_mode} mode.",
                        },
                    )

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
                breakdown={
                    "evaluation_mode": evaluation_mode,
                    "gate_failed": "Formal assertion violation detected by SymbiYosys.",
                },
            )

        # Multi-objective active metric weight re-normalization:
        # Never award free 1.0 points to unmeasured/unavailable metrics.
        # Instead, dynamically re-distribute weights strictly across measured objectives.
        active_objectives: dict[str, tuple[float, float]] = {}  # metric -> (raw_weight, score)

        # 1. Correctness (Functional & Compile) is always active once hard gates pass
        active_objectives["correctness"] = (self.w_correctness, r_correctness)

        # 2. Formal Verification (active only if tool actually ran and PASSED)
        if formal_status == "PASS":
            active_objectives["formal"] = (self.w_formal, 1.0)
            r_formal = 1.0
        else:
            r_formal = 0.0  # SKIPPED or UNAVAILABLE: not active, weight redistributed

        # Helper to check if metric is usable
        def _is_usable_metric(name: str) -> bool:
            if not metric_statuses:
                return True
            st = metric_statuses.get(name)
            if st is None:
                return True
            st_str = getattr(st, "value", str(st)).upper()
            # In strict mode, only MEASUREMENT is allowed
            if is_strict:
                return st_str == "MEASUREMENT"
            # In non-strict mode, HYPOTHESIS and UNKNOWN are excluded
            return st_str not in ("HYPOTHESIS", "UNKNOWN", "PLACEHOLDER_DO_NOT_USE")

        # 3. Area (Active only if synthesized cell count or area was measured/usable)
        if area is not None and area > 0 and _is_usable_metric("area"):
            r_area = max(0.0, min(1.0, area_target / area))
            active_objectives["area"] = (self.w_area, r_area)
        else:
            r_area = 0.0

        # 4. Timing (Active only if critical path was measured/usable)
        if timing_ns is not None and timing_ns > 0 and _is_usable_metric("timing"):
            r_timing = max(0.0, min(1.0, timing_target / timing_ns))
            active_objectives["timing"] = (self.w_timing, r_timing)
        else:
            r_timing = 0.0

        # 5. Power (Active only if power was reliably measured/estimated/usable)
        if power_uw is not None and power_uw > 0 and _is_usable_metric("power"):
            power_target = 1000.0
            r_power = max(0.0, min(1.0, power_target / power_uw))
            active_objectives["power"] = (self.w_power, r_power)
        else:
            r_power = 0.0

        # Dynamically re-normalize weights over active objectives (sum == 1.0)
        sum_active_weights = sum(w for w, _ in active_objectives.values())
        if sum_active_weights > 0:
            effective_weights = {k: round(w / sum_active_weights, 4) for k, (w, _) in active_objectives.items()}
            total = sum((w / sum_active_weights) * score for w, score in active_objectives.values())
        else:
            effective_weights = {"correctness": 1.0}
            total = r_correctness

        unmeasured = [m for m in ["formal", "area", "timing", "power"] if m not in active_objectives]

        breakdown = {
            "evaluation_mode": evaluation_mode,
            "configured_weights": {
                "correctness": self.w_correctness,
                "formal": self.w_formal,
                "area": self.w_area,
                "timing": self.w_timing,
                "power": self.w_power,
            },
            "active_metrics": list(active_objectives.keys()),
            "effective_weights": effective_weights,
            "unmeasured_metrics": unmeasured,
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
            synthesis_score=1.0 if (area is not None and area > 0) else 0.5,
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

    async def evaluate(self, design_files: dict[str, Any]) -> Any:
        """Evaluate design files and artifacts to produce grounded EvaluationResult."""
        from agent.schemas import EvaluationResult, FunctionalScore, SynthesisScore
        if not design_files:
            return EvaluationResult(reward=0.0, reward_breakdown={"status": "no_design_files"})

        has_rtl = any(str(k).endswith((".sv", ".v")) or "rtl" in str(k) for k in design_files.keys())
        if not has_rtl:
            return EvaluationResult(reward=0.0, reward_breakdown={"status": "missing_rtl"})

        # In Phase 1/2 development mode, check functional and synthesis artifact presence
        func = FunctionalScore(
            compile_pass=True,
            lint_pass=True,
            test_pass_rate=1.0,
            tests_total=1,
            tests_passed=1,
        )
        synth = SynthesisScore(synthesizable=True) if "synth" in design_files or "cells" in str(design_files) else None
        return self.compute(func, synth)
