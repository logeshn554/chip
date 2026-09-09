"""Tests for the 7-Level Hardware Curriculum Benchmarks."""

import pytest
from benchmarks.curriculum import BenchmarkCurriculum, get_benchmark, get_curriculum_level


class TestCurriculumBenchmarks:
    def test_curriculum_levels_present(self):
        curriculum = BenchmarkCurriculum()
        for level in range(1, 8):
            tasks = curriculum.get_level_tasks(level)
            assert len(tasks) >= 1, f"Level {level} should have at least 1 benchmark task"

    def test_level_1_tasks(self):
        tasks = get_curriculum_level(1)
        ids = [t.id for t in tasks]
        assert "L1_NOT_GATE" in ids
        assert "L1_AND_OR" in ids
        assert "L1_MUX2TO1" in ids
        assert "L1_DECODER2TO4" in ids
        assert "L1_COUNTER" in ids

    def test_level_3_mac_task(self):
        task = get_benchmark("L3_MAC_8BIT_SIGNED")
        assert task is not None
        assert task.level == 3
        assert task.top_module == "mac"
        assert "signed" in task.reference_rtl
        # Public spec should not expose held-out tests
        spec = task.get_public_spec()
        assert "L3_MAC_8BIT_SIGNED" in spec
        assert "-128" not in spec  # held out test should not be visible

    def test_level_7_npu_task(self):
        task = get_benchmark("L7_NPU_PE")
        assert task is not None
        assert task.level == 7
        assert "npu_pe" in task.top_module
        assert len(task.verification_criteria) >= 2
