"""Tests for RL & GRPO Training Pipeline Hardening."""

import json
import os
import pytest
from agent.schemas import Episode, TrajectoryStep
from benchmarks.curriculum import BenchmarkCurriculum, BenchmarkTask
from evaluator.reward import RewardEngine
from learning.dataset import TrajectoryDatasetBuilder, normalize_rtl_for_dedup
from learning.environment import HardwareDesignEnv, QwenHardwareDesignPolicy
from learning.grpo import HardwareRewardEvaluator
from tools.yosys import YosysTool


class TestYosysFallbackHardening:
    @pytest.mark.asyncio
    async def test_yosys_allow_heuristic_fallback_false(self, tmp_path):
        tool = YosysTool(work_dir=str(tmp_path / "synth"))
        tool._has_binary = False  # Simulate real Yosys missing

        # With fallback disabled (training mode)
        res = await tool.synthesize("dummy.sv", top_module="dummy", allow_heuristic_fallback=False)
        assert res["metric_type"] == "none"
        assert res["status"] in ["unavailable", "failed"]

        # With fallback enabled (manual dev inspect)
        sample_code = "module dummy (input a, output y); assign y = ~a; endmodule"
        res_heur = tool._heuristic_lint_check(sample_code, top_module="dummy")
        assert res_heur["metric_type"] == "heuristic_lint_only"
        assert "heuristic_area_guess" in res_heur

    @pytest.mark.asyncio
    async def test_grpo_evaluator_never_uses_heuristic_area(self, tmp_path, monkeypatch):
        evaluator = HardwareRewardEvaluator(work_dir=str(tmp_path / "eval"))
        # Force Yosys synthesize to return an estimated result
        async def mock_synth(file_path, top_module="mac", allow_heuristic_fallback=True):
            return {
                "stage": "yosys",
                "status": "passed",
                "metric_type": "estimated",
                "cells": 999,
                "estimated_area": 3136.86,
            }
        monkeypatch.setattr(evaluator.yosys, "synthesize", mock_synth)

        # Mock compile and cocotb as passed
        async def mock_verilator(rtl_path, top_module="not_gate"):
            return {"status": "passed"}
        async def mock_cocotb(rtl_path, testbench_path=None, benchmark_task=None):
            return {"status": "passed", "tests_total": 2, "tests_passed": 2}
        async def mock_formal(rtl_path, top_module="not_gate", external_properties=None):
            return {"status": "SKIPPED"}

        monkeypatch.setattr(evaluator.verilator, "lint_and_compile", mock_verilator)
        monkeypatch.setattr(evaluator.cocotb, "run_tests", mock_cocotb)
        monkeypatch.setattr(evaluator.formal, "verify", mock_formal)

        code = "```systemverilog\nmodule not_gate(input a, output y); assign y = ~a; endmodule\n```"
        res = await evaluator.evaluate_completion_async(code, benchmark_task_id="L1_NOT_GATE")
        # Area MUST be None, not 999 or 3136.86!
        assert res["area"] is None
        # Heuristic area must NOT enter reward calculation
        assert "area" not in res["breakdown"]["active_metrics"]


class TestEvaluationModes:
    def test_research_fast_vs_strict_formal_gate(self):
        engine = RewardEngine()

        # Fast mode: formal skipped -> weights re-normalize, design is valid
        res_fast = engine.compute_modular_reward(
            compile_success=True,
            test_pass_rate=1.0,
            formal_status="SKIPPED",
            area=None,
            evaluation_mode="research_fast",
        )
        assert res_fast.is_valid_hardware is True
        assert res_fast.normalized_reward == 1.0  # 100% of active (correctness)
        assert "formal" not in res_fast.breakdown["active_metrics"]

        # Strict mode: formal skipped -> fails strict gate, design is invalid
        res_strict = engine.compute_modular_reward(
            compile_success=True,
            test_pass_rate=1.0,
            formal_status="SKIPPED",
            area=10.0,
            evaluation_mode="research_strict",
        )
        assert res_strict.is_valid_hardware is False
        assert res_strict.normalized_reward <= 0.4
        assert "Formal verification mandatory" in res_strict.breakdown["gate_failed"]

    def test_research_strict_synthesis_gate(self):
        engine = RewardEngine()
        # Strict mode: area None -> fails strict synthesis gate
        res_strict = engine.compute_modular_reward(
            compile_success=True,
            test_pass_rate=1.0,
            formal_status="PASS",
            area=None,
            evaluation_mode="research_strict",
        )
        assert res_strict.is_valid_hardware is False
        assert "Actual logic synthesis mandatory" in res_strict.breakdown["gate_failed"]


class TestExactBenchmarkRouting:
    def test_exact_task_required_rejects_fuzzy_inference(self):
        evaluator = HardwareRewardEvaluator(require_exact_task=True)

        # Non-exact prompt or missing task_id returns None
        task = evaluator.resolve_benchmark_task(
            task_id=None,
            prompt="Please design an inverter module not_gate for Level 1",
        )
        assert task is None

        # Exact task_id resolves successfully
        task_exact = evaluator.resolve_benchmark_task(task_id="L1_NOT_GATE")
        assert task_exact is not None
        assert task_exact.id == "L1_NOT_GATE"
        assert task_exact.target_cells is not None

    def test_task_specific_cell_target(self):
        curriculum = BenchmarkCurriculum()
        not_task = curriculum.get_task("L1_NOT_GATE")
        mac_task = curriculum.get_task("L3_MAC_8BIT_SIGNED")
        assert not_task.target_cells == 10.0
        assert mac_task.target_cells == 150.0
        assert not_task.target_cells != mac_task.target_cells


class TestObservationSpaceContractHardening:
    def test_env_observation_space_contains_obs(self, tmp_path):
        env = HardwareDesignEnv(
            work_dir=str(tmp_path / "env_test"),
            task="Design a gate with unusual non-printable \x00\x01\x1f and unicode chars \u2603",
        )
        obs, info = env.reset()
        assert env.observation_space.contains(obs)

        # Step with very long code and unicode
        long_rtl = "module test;\n" + ("// comment \u2764\n" * 500) + "endmodule\n"
        obs, rew, term, trunc, info = env.step({"action": "GENERATE_RTL", "params": {"code": long_rtl}})
        assert env.observation_space.contains(obs)
        assert len(obs["current_rtl"]) <= 4000

        # Check get_reward
        assert isinstance(env.get_reward(), float)


class TestAuthenticDatasetAndDeduplication:
    def test_normalize_rtl_for_dedup(self):
        rtl1 = """// Single line comment
        module not_gate (input a, output y);
            assign y = ~a; /* multi line */
        endmodule"""

        rtl2 = """module not_gate ( input a , output y ) ;
        assign y = ~a ;
        endmodule"""

        norm1 = normalize_rtl_for_dedup(rtl1)
        assert "//" not in norm1
        assert "/*" not in norm1

    def test_sft_dataset_has_no_synthetic_thinking(self):
        ep = Episode(
            episode_id="ep_1",
            task="Design an inverter",
            steps=[
                TrajectoryStep(
                    step_index=1,
                    action="GENERATE_RTL",
                    action_params={"code": "module not_gate; endmodule"},
                    observation="passed",
                    reward=0.8,
                    state_summary="Initial RTL generated",
                )
            ],
            final_reward=0.9,
            success=True,
        )

        builder = TrajectoryDatasetBuilder()
        sft_data = builder.build_sft_dataset([ep])
        assert len(sft_data) == 1
        resp = json.loads(sft_data[0]["response"])
        assert "thinking" not in resp or "Based on the hardware requirements" not in str(resp.get("thinking", ""))
        assert resp["action"] == "GENERATE_RTL"

    def test_grpo_prompt_dataset_has_mandatory_benchmark_task_id(self):
        builder = TrajectoryDatasetBuilder()
        dataset = builder.build_grpo_prompt_dataset(["L1_NOT_GATE", "L1_MUX2TO1"])
        assert len(dataset) == 2
        for row in dataset:
            assert "benchmark_task_id" in row
            assert row["benchmark_task_id"] in ["L1_NOT_GATE", "L1_MUX2TO1"]


class TestBaselineEvaluationHarness:
    @pytest.mark.asyncio
    async def test_baseline_harness_direct_rtl(self, tmp_path, monkeypatch):
        from benchmarks.evaluate_baselines import BaselineEvaluationHarness
        harness = BaselineEvaluationHarness(work_dir=str(tmp_path / "base_eval"))

        curriculum = BenchmarkCurriculum()
        task = curriculum.get_task("L1_NOT_GATE")

        # Mock evaluator response
        async def mock_eval(completion, benchmark_task_id=None, sub_work_dir=None):
            return {
                "reward": 0.85,
                "compile_pass": True,
                "pass_rate": 1.0,
                "formal_status": "PASS",
                "area": 8.0,
            }
        monkeypatch.setattr(harness.evaluator, "evaluate_completion_async", mock_eval)

        record = await harness.evaluate_direct_rtl(task, "module not_gate; assign y = ~a; endmodule")
        assert record.functional_pass is True
        assert record.formal_pass is True
        assert record.synthesis_pass is True
        assert record.cells == 8

        summary = harness.aggregate_summary("baseline_1_direct", [record])
        assert summary.functional_pass_rate == 1.0
        assert summary.formal_pass_rate == 1.0
        report = harness.generate_markdown_report([summary])
        assert "| **baseline_1_direct** |" in report
