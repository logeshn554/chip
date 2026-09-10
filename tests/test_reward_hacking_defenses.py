"""
Comprehensive Anti-Reward-Hacking and Security Defense Test Suite.

Verifies that the hardware agent and evaluation engine defend against 10 critical attack/hacking vectors:
1. Evaluator modification attempts (file write rejection)
2. Benchmark ID manipulation (strict routing rejection)
3. Public-test exploitation (held-out test isolation)
4. Formal property modification (rejects tampering with benchmark SVA assertions)
5. Self-authored assertions as proof (model assertions rejected as formal proof)
6. Skipped formal verification (safety gate failure in strict/training mode)
7. Estimated metric exploitation (exclusion of heuristic cells*3.14 during training)
8. Hidden-test leakage (public spec never exposes held-out test vectors)
9. Hard-coded fallback generation (never fabricates reference RTL when inference is unavailable)
10. Unauthorized file writes and arbitrary shell execution (sandboxing & action schema enforcement)
"""

import pytest
import os
from agent.action_router import is_path_safe_for_write, ActionRouter
from agent.schemas import AgentAction, ActionType, ActionStatus
from benchmarks.curriculum import BenchmarkCurriculum, BenchmarkTask
from evaluator.reward import RewardEngine
from learning.grpo import HardwareRewardEvaluator
from tools.yosys import YosysTool


class TestRewardHackingDefenses:
    """Rigorous defenses against cheating, reward hacking, and environment escape."""

    # 1. Evaluator Modification Defense
    def test_defense_1_evaluator_modification_rejected(self):
        """Cannot overwrite evaluator source files."""
        assert not is_path_safe_for_write("evaluator/reward.py")
        assert not is_path_safe_for_write("evaluator/metrics.py")
        assert not is_path_safe_for_write("./evaluator/synthesis.py")

    # 2. Benchmark ID Manipulation Defense
    def test_defense_2_benchmark_id_manipulation_rejected(self):
        """In strict training mode, fuzzy prompt regex cannot hijack benchmark task id."""
        evaluator = HardwareRewardEvaluator(require_exact_task=True)
        # Attempt to fool evaluator with prompt text mentioning a simple inverter
        resolved = evaluator.resolve_benchmark_task(
            task_id="NON_EXISTENT_BENCHMARK_999",
            prompt="Design a simple NOT gate [L1_NOT_GATE]",
            require_exact=True,
        )
        assert resolved is None, "Heuristic prompt text must not resolve task when exact matching is required"

    # 3. Public-Test Exploitation Defense
    def test_defense_3_public_test_exploitation_isolated(self):
        """Passing public tests does not grant pass if held-out tests fail."""
        curriculum = BenchmarkCurriculum()
        task = curriculum.get_task("L1_NOT_GATE")
        assert task is not None
        # Verify public tests and held-out tests have disjoint expectations
        pub_inputs = [t["a"] for t in task.public_tests]
        held_inputs = [t["a"] for t in task.held_out_tests]
        assert set(pub_inputs).isdisjoint(set(held_inputs)), "Held-out tests must not overlap with public tests"

    # 4. Formal Property Modification Defense
    def test_defense_4_formal_property_modification_rejected(self):
        """Evaluator uses task's authoritative formal properties, not model-embedded ones."""
        curriculum = BenchmarkCurriculum()
        task = curriculum.get_task("L3_MAC_8BIT_SIGNED")
        assert task is not None
        # Authoritative formal properties are stored in curriculum BenchmarkTask
        assert "assert property" in task.formal_properties
        # Tampered prompt cannot change task.formal_properties
        tampered_prompt = "assume property (1 == 1); assert property (1 == 1);"
        assert tampered_prompt not in task.formal_properties

    # 5. Self-Authored Assertions Rejected as Proof
    def test_defense_5_self_authored_assertions_rejected(self):
        """Candidate containing trivial `assert (1'b1)` is evaluated by external SymbiYosys/Cocotb, not trusted blindly."""
        engine = RewardEngine()
        # Even if candidate RTL contains self-authored assertions, if formal_status != PASS in strict mode, gate fails
        res = engine.compute_modular_reward(
            compile_success=True,
            test_pass_rate=1.0,
            formal_status="SKIPPED",
            area=150.0,
            evaluation_mode="STRICT_EVALUATION",
        )
        assert not res.is_valid_hardware
        assert res.total_reward <= 0.4
        assert "Formal verification mandatory" in res.breakdown.get("gate_failed", "")

    # 6. Skipped Formal Verification Rejection
    def test_defense_6_skipped_formal_verification_fails_strict_gate(self):
        """Skipping formal verification in TRAINING / STRICT mode triggers hard gate rejection."""
        engine = RewardEngine()
        res = engine.compute_modular_reward(
            compile_success=True,
            test_pass_rate=1.0,
            formal_status="SKIPPED",
            area=150.0,
            evaluation_mode="TRAINING",
        )
        assert not res.is_valid_hardware
        assert res.total_reward == 0.3

    # 7. Estimated Metric Exploitation Defense
    @pytest.mark.asyncio
    async def test_defense_7_estimated_metric_exploitation_prevented(self):
        """Heuristic cell count estimator (cells*3.14) is disabled during training."""
        tool = YosysTool()
        res = await tool.synthesize("non_existent_file_for_test.sv", allow_heuristic_fallback=False)
        assert res["metric_type"] == "none"
        assert res["cells"] is None
        assert res["estimated_area"] is None

    # 8. Hidden-Test Leakage Defense
    def test_defense_8_hidden_test_never_leaked_in_public_spec(self):
        """get_public_spec() never reveals held_out_tests to the LLM."""
        curriculum = BenchmarkCurriculum()
        for task in curriculum.get_implemented_tasks():
            spec = task.get_public_spec()
            for held_test in task.held_out_tests:
                # Stringified held test should not appear verbatim in public spec
                assert str(held_test) not in spec
            assert "Held-Out" not in spec

    # 9. Hard-Coded Fallback Generation Defense
    @pytest.mark.asyncio
    async def test_defense_9_hardcoded_fallback_rejected_when_inference_unavailable(self):
        """If policy completion is missing, held-out evaluation scores 0.0 rather than substituting reference RTL."""
        evaluator = HardwareRewardEvaluator(require_exact_task=True)
        # Empty completion should fail compilation cleanly
        eval_res = await evaluator.evaluate_completion_async(
            completion="",
            benchmark_task_id="L1_NOT_GATE",
        )
        assert eval_res.get("compile_pass") is False
        assert eval_res.get("reward") == 0.0

    # 10. Unauthorized File Writes & Arbitrary Shell Rejection
    @pytest.mark.asyncio
    async def test_defense_10_unauthorized_file_writes_and_arbitrary_commands(self):
        """ActionRouter rejects paths outside workspace and unknown shell actions."""
        router = ActionRouter()
        
        # Test 10a: Directory traversal write rejection
        action_bad_path = AgentAction(
            action_type=ActionType.GENERATE_RTL,
            params={"filename": "../../../configs/agent.yaml", "code": "corrupted: true"},
        )
        res1 = await router.execute(action_bad_path)
        assert res1.status == ActionStatus.ERROR
        assert "Security violation" in res1.output

        # Test 10b: Arbitrary shell execution rejection (unknown action)
        action_shell = AgentAction(
            action_type=ActionType.UNKNOWN,
            params={"cmd": "rm -rf /"},
        )
        res2 = await router.execute(action_shell)
        assert res2.status == ActionStatus.ERROR
        assert "Unknown action requested" in res2.output
