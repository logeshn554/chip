"""Tests for WebRL-Style Online Curriculum & Self-Evolution Architecture."""

import json
import os
import pytest

from agent.schemas import Episode, TrajectoryStep
from benchmarks.curriculum import BenchmarkCurriculum, BenchmarkTask
from learning.self_evolution import (
    TrajectoryReconstructor,
    HardwareTaskGenerator,
    SelfEvolutionController,
    GenerationRecord,
)
from memory.experience_store import ExperienceStore, Experience, categorize_error


class TestTrajectoryReconstructor:
    def test_discounted_returns_computation(self):
        reconstructor = TrajectoryReconstructor(gamma=0.9)

        ep = Episode(
            episode_id="ep_returns_test",
            task="Design a counter",
            steps=[
                TrajectoryStep(step_index=0, action="CREATE_RTL", reward=-0.01),
                TrajectoryStep(step_index=1, action="RUN_VERILATOR", observation="%Error: syntax error", reward=-0.2),
                TrajectoryStep(step_index=2, action="EDIT_RTL", reward=-0.01),
                TrajectoryStep(step_index=3, action="RUN_VERILATOR", observation="passed", reward=0.3),
                TrajectoryStep(step_index=4, action="COMPLETE", observation="passed", reward=1.0),
            ],
            final_reward=1.0,
            success=True,
        )

        reconstructed = reconstructor.reconstruct(ep)
        assert len(reconstructed.steps) == 5
        # G_4 = 1.0
        assert reconstructed.steps[4].discounted_return == 1.0
        # G_3 = 0.3 + 0.9 * 1.0 = 1.2
        assert abs(reconstructed.steps[3].discounted_return - 1.2) < 1e-3
        # Failure-repair detection: step 1 failed, step 3 passed
        assert len(reconstructed.failure_repair_pairs) >= 1
        assert reconstructed.is_success is True


class TestHardwareTaskGenerator:
    def test_generate_variants_across_levels(self):
        generator = HardwareTaskGenerator()

        # Level 1 MUX variant
        l1_task = generator.generate_level_variant(1, seed=42)
        assert l1_task.level == 1
        assert "MUX" in l1_task.id
        assert len(l1_task.public_tests) > 0
        assert len(l1_task.held_out_tests) > 0
        assert "assert property" in l1_task.formal_properties
        assert l1_task.target_cells is not None

        # Level 2 FIFO variant
        l2_task = generator.generate_level_variant(2, seed=42)
        assert l2_task.level == 2
        assert "FIFO" in l2_task.id
        assert l2_task.top_module == "fifo"

        # Level 3 MAC variant
        l3_task = generator.generate_level_variant(3, seed=42)
        assert l3_task.level == 3
        assert "MAC" in l3_task.id
        assert l3_task.top_module == "mac"


class TestExperienceStoreFailureTaxonomy:
    def test_error_categorization(self):
        assert categorize_error("%Error-WIDTH: Operator '+' expects 32 bits") == "SYNTAX_LINT"
        assert categorize_error("Cocotb test failed: assertion mismatch") == "FUNCTIONAL_ASSERT"
        assert categorize_error("Synthesis error: Delays (#10) cannot be synthesized") == "SYNTHESIS_ERROR"
        assert categorize_error("SymbiYosys formal verification failed with BMC violation") == "FORMAL_FAIL"

    def test_record_and_retrieve_episode_experience(self, tmp_path):
        store = ExperienceStore(persist_dir=str(tmp_path / "exp_db"))

        ep = Episode(
            episode_id="ep_fix_test",
            task="Design an 8-bit signed multiplier",
            steps=[
                TrajectoryStep(
                    step_index=0,
                    action="CREATE_RTL",
                    action_params={"code": "logic [7:0] a;"},
                    observation="%Error-WIDTH: unsigned bug",
                    reward=-0.1,
                ),
                TrajectoryStep(
                    step_index=1,
                    action="EDIT_RTL",
                    action_params={"code": "logic signed [7:0] a;"},
                    observation="passed verilator compile",
                    reward=0.8,
                ),
            ],
            final_reward=0.8,
            success=True,
        )

        added_ids = store.record_episode_experience(ep)
        assert len(added_ids) == 1

        # Query past experiences by error
        matches = store.retrieve_similar_failures("unsigned bug width mismatch")
        assert len(matches) > 0
        assert any("signed" in m.get("correction", "") for m in matches)


class TestSelfEvolutionController:
    @pytest.mark.asyncio
    async def test_curriculum_progression_and_promotion(self, tmp_path, monkeypatch):
        controller = SelfEvolutionController(
            start_level=1,
            max_level=3,
            promotion_threshold=0.75,
            work_dir=str(tmp_path / "self_evo"),
        )
        assert controller.current_level == 1

        # Mock online rollout
        async def mock_rollout(task, policy=None, max_steps=6):
            return Episode(
                episode_id=f"rollout_{task.id}",
                task=task.get_public_spec(),
                steps=[
                    TrajectoryStep(step_index=0, action="GENERATE_RTL", reward=0.5),
                    TrajectoryStep(step_index=1, action="COMPLETE", reward=0.9),
                ],
                final_reward=0.9,
                success=True,
            )

        # Mock held-out evaluator returning 100% pass rate
        async def mock_eval(completion, benchmark_task_id=None, sub_work_dir=None):
            return {
                "reward": 0.95,
                "compile_pass": True,
                "pass_rate": 1.0,
                "formal_status": "PASS",
                "area": 10.0,
            }

        monkeypatch.setattr(controller, "collect_online_rollout", mock_rollout)
        monkeypatch.setattr(controller.evaluator, "evaluate_completion_async", mock_eval)

        # Run generation 1: high pass rate should trigger promotion to Level 2!
        record = await controller.run_generation(include_variants=False)
        assert record.generation == 1
        assert record.promoted is True
        assert controller.current_level == 2
        assert len(controller.history) == 1
        assert record.held_out_pass_rate == 1.0
