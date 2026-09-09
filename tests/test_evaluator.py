"""Tests for the Evaluator and Reward Engine."""

import pytest

from agent.schemas import (
    CompileResult,
    FunctionalScore,
    LintResult,
    SimulationResult,
    SynthesisResult,
    SynthesisScore,
)
from evaluator.functional import FunctionalEvaluator
from evaluator.synthesis import SynthesisEvaluator
from evaluator.reward import RewardEngine


class TestFunctionalEvaluator:
    def test_perfect_score(self):
        evaluator = FunctionalEvaluator()
        score = evaluator.evaluate(
            compile_result=CompileResult(success=True),
            lint_result=LintResult(success=True),
            sim_result=SimulationResult(
                success=True, tests_total=10, tests_passed=10
            ),
        )
        assert score.compile_pass is True
        assert score.lint_pass is True
        assert score.test_pass_rate == 1.0
        assert score.score == 1.0

    def test_compile_failure(self):
        evaluator = FunctionalEvaluator()
        score = evaluator.evaluate(
            compile_result=CompileResult(success=False, errors=["syntax error"]),
        )
        assert score.compile_pass is False
        assert score.score < 0.5

    def test_partial_tests(self):
        evaluator = FunctionalEvaluator()
        score = evaluator.evaluate(
            compile_result=CompileResult(success=True),
            lint_result=LintResult(success=True),
            sim_result=SimulationResult(
                success=False, tests_total=10, tests_passed=7, tests_failed=3
            ),
        )
        assert score.test_pass_rate == 0.7
        assert 0.0 < score.score < 1.0

    def test_meets_threshold(self):
        evaluator = FunctionalEvaluator({"test_pass_rate_min": 0.95})
        good = FunctionalScore(
            compile_pass=True, lint_pass=True, lint_warnings=2,
            test_pass_rate=0.96, tests_total=100, tests_passed=96,
        )
        bad = FunctionalScore(
            compile_pass=True, lint_pass=True, lint_warnings=2,
            test_pass_rate=0.80, tests_total=100, tests_passed=80,
        )
        assert evaluator.meets_threshold(good) is True
        assert evaluator.meets_threshold(bad) is False


class TestSynthesisEvaluator:
    def test_good_synthesis(self):
        evaluator = SynthesisEvaluator({"area_max": 10000, "timing_ns_max": 10.0})
        score = evaluator.evaluate(SynthesisResult(
            success=True,
            cell_count=500,
            critical_path_ns=3.0,
        ))
        assert score.synthesizable is True
        assert score.area_score > 0.8
        assert score.timing_score == 1.0

    def test_failed_synthesis(self):
        evaluator = SynthesisEvaluator()
        score = evaluator.evaluate(SynthesisResult(
            success=False,
            errors=["Synthesis failed"],
        ))
        assert score.synthesizable is False
        assert score.score == 0.0

    def test_large_design(self):
        evaluator = SynthesisEvaluator({"area_max": 1000})
        score = evaluator.evaluate(SynthesisResult(
            success=True,
            cell_count=5000,  # 5x the max
        ))
        assert score.area_score < 0.1  # Exponential decay


class TestRewardEngine:
    def test_phase1_reward(self):
        engine = RewardEngine({"reward_phase": 1})
        result = engine.compute(FunctionalScore(
            compile_pass=True, lint_pass=True,
            test_pass_rate=0.9, tests_total=10, tests_passed=9,
        ))
        assert result.reward > 0.7
        assert "compile" in result.reward_breakdown

    def test_phase2_reward(self):
        engine = RewardEngine({
            "reward_phase": 2,
            "reward_weights": {
                "correctness": 0.5,
                "area": 0.2,
                "timing": 0.2,
                "power": 0.1,
            },
        })
        result = engine.compute(
            FunctionalScore(
                compile_pass=True, lint_pass=True,
                test_pass_rate=1.0, tests_total=10, tests_passed=10,
            ),
            SynthesisScore(
                synthesizable=True,
                area_score=0.8,
                timing_score=0.9,
                power_score=0.7,
            ),
        )
        assert result.reward > 0.5
        assert "area" in result.reward_breakdown
        assert "timing" in result.reward_breakdown

    def test_zero_reward(self):
        engine = RewardEngine({"reward_phase": 1})
        result = engine.compute(FunctionalScore())
        assert result.reward == 0.0

    def test_reward_bounded(self):
        engine = RewardEngine({"reward_phase": 1})
        result = engine.compute(FunctionalScore(
            compile_pass=True, lint_pass=True,
            test_pass_rate=1.0, tests_total=100, tests_passed=100,
        ))
        assert 0.0 <= result.reward <= 1.0

    def test_phase_switching(self):
        engine = RewardEngine({"reward_phase": 1})
        assert engine.phase == 1
        engine.set_phase(2)
        assert engine.phase == 2

    def test_modular_reward_weight_renormalization(self):
        engine = RewardEngine()
        # Area measured (500 cells), timing and power unavailable (None)
        res = engine.compute_modular_reward(
            compile_success=True,
            test_pass_rate=1.0,
            formal_status="SKIPPED",
            area=500.0,
            area_target=500.0,
            timing_ns=None,
            power_uw=None,
        )
        assert res.is_valid_hardware is True
        # Effective weights must sum to 1.0
        effective_w = res.breakdown["effective_weights"]
        assert abs(sum(effective_w.values()) - 1.0) < 1e-3
        # Timing and formal must be in unmeasured list, not receiving free 1.0 score
        assert "timing" in res.breakdown["unmeasured_metrics"]
        assert "formal" in res.breakdown["unmeasured_metrics"]
        assert "timing" not in effective_w

    def test_compute_v1_base_reward_with_auxiliary_formal(self):
        engine = RewardEngine()
        res = engine.compute_v1_reward(
            compile_success=True,
            all_functional_tests_pass=True,
            synthesis_success=True,
            lint_clean=True,
            formal_pass=True,
        )
        assert res.base_reward_v1 == 8.0
        assert res.total_reward == 8.0
        assert res.formal_score == 1.0
        assert res.breakdown["metric_type"] == "v1_8point_base_with_auxiliary_formal"
