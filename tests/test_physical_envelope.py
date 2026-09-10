"""Unit and integration tests for Physical Envelope Modeling and Constraint Evolution Loop."""

import pytest
from agent.schemas import (
    DevicePhysicalEnvelope,
    HardwareArchitectureCandidate,
    PhysicalProjectionMetrics,
    ConstraintCheckResult,
)
from evaluator.physical_envelope import PhysicalEnvelopeModel, PhysicalConstraintChecker
from agent.architecture_search import ArchitectureSearchEngine
from learning.self_evolution import PhysicalConstraintEvolutionLoop


class TestPhysicalEnvelopeModel:
    """Verify grounded physical modeling from digital architecture to packaging & thermal."""

    def test_default_projections(self):
        envelope = DevicePhysicalEnvelope(
            enclosure_length_mm=100.0,
            enclosure_width_mm=30.0,
            max_power_w=5.0,
            max_junction_temp_c=85.0,
            min_tokens_per_sec=15.0,
        )
        model = PhysicalEnvelopeModel(envelope)

        cand = HardwareArchitectureCandidate(
            architecture_id="cand_test_01",
            task_id="L3_MAC",
            parallelism=2,
            pipeline_depth=2,
            actual_synthesis_metrics={"cells": 400},
        )
        proj = model.project_candidate(cand, process_node_nm=28)

        # Die area should be positive and bounded
        assert proj.digital_die_area_mm2 > 0.0
        # Package area includes ASIC, DRAM, Storage, PMIC, USB-C
        assert proj.total_component_area_mm2 > 400.0
        # Total PCB area should be non-zero and bounded
        assert proj.total_pcb_area_mm2 > proj.total_component_area_mm2
        # Power should be positive
        assert 0.5 < proj.total_device_power_w < 10.0
        # Thermal junction temperature should exceed ambient
        assert proj.estimated_junction_temp_c > envelope.ambient_temp_c
        # Throughput should be calculated
        assert proj.achievable_tokens_per_sec > 0.0

    def test_thermal_throttling_activation(self):
        # Configure a high-resistance thermal envelope that triggers throttling
        envelope = DevicePhysicalEnvelope(
            ambient_temp_c=40.0,
            thermal_resistance_c_per_w=25.0,  # High thermal resistance
            max_junction_temp_c=70.0,         # Low ceiling
        )
        model = PhysicalEnvelopeModel(envelope)

        # High parallelism design drawing high power
        cand = HardwareArchitectureCandidate(
            architecture_id="cand_hot",
            task_id="L4_VECTOR",
            parallelism=8,
            pipeline_depth=3,
            actual_synthesis_metrics={"cells": 5000},
        )
        proj = model.project_candidate(cand)

        # Power should be high and junction temp should trigger throttling
        assert proj.estimated_junction_temp_c > 70.0
        assert proj.bottleneck == "thermal_throttled"


class TestPhysicalConstraintChecker:
    """Verify constraint checker flags violations and passes valid designs."""

    def test_flag_power_and_size_violations(self):
        # Strict envelope: tiny enclosure and tiny power budget
        envelope = DevicePhysicalEnvelope(
            enclosure_length_mm=25.0,
            enclosure_width_mm=10.0,
            max_power_w=1.0,  # Extremely strict 1W
            max_junction_temp_c=85.0,
            min_tokens_per_sec=5.0,
        )
        checker = PhysicalConstraintChecker(envelope)

        cand = HardwareArchitectureCandidate(
            architecture_id="cand_oversized",
            task_id="L7_NPU",
            parallelism=8,
            pipeline_depth=3,
            actual_synthesis_metrics={"cells": 8000},
        )
        result = checker.evaluate(cand)

        assert not result.passed
        assert not result.checks["power_fit"]
        assert not result.checks["size_fit"]
        assert len(result.violations) >= 2
        assert len(result.recommendations) >= 1

    def test_passing_candidate_design(self):
        # Realistic standard envelope: 100mm x 30mm, 5.0W, 85C, 15 tok/s
        envelope = DevicePhysicalEnvelope(
            enclosure_length_mm=100.0,
            enclosure_width_mm=30.0,
            max_power_w=5.0,
            max_junction_temp_c=85.0,
            min_tokens_per_sec=12.0,
        )
        checker = PhysicalConstraintChecker(envelope)

        # Efficient INT4 streaming candidate
        cand = HardwareArchitectureCandidate(
            architecture_id="cand_efficient_int4",
            task_id="L3_MAC",
            parallelism=2,
            pipeline_depth=2,
            memory_organization="streaming",
            arithmetic_strategy="quantized_int4_shared",
            actual_synthesis_metrics={"cells": 350},
        )
        result = checker.evaluate(cand)

        assert result.passed
        assert all(result.checks.values())
        assert len(result.violations) == 0


class TestConstraintGuidedMutations:
    """Verify architecture search mutates designs specifically in response to constraint diagnostics."""

    def test_mutation_response_to_power_violation(self):
        search_engine = ArchitectureSearchEngine()
        cand = HardwareArchitectureCandidate(
            architecture_id="c_parallel",
            task_id="L3_MAC",
            parallelism=4,
            pipeline_depth=2,
        )
        fake_result = ConstraintCheckResult(
            passed=False,
            checks={"power_fit": False, "size_fit": True, "throughput_fit": True},
            violations=["Power 6.2W exceeds 5.0W limit"],
        )
        child = search_engine.mutate_for_physical_constraints(cand, fake_result)

        # Should reduce parallelism to address power
        assert child.parallelism < cand.parallelism
        assert child.parent_architecture_id == cand.architecture_id

    def test_mutation_response_to_throughput_violation(self):
        search_engine = ArchitectureSearchEngine()
        cand = HardwareArchitectureCandidate(
            architecture_id="c_slow",
            task_id="L3_MAC",
            parallelism=1,
            pipeline_depth=1,
        )
        fake_result = ConstraintCheckResult(
            passed=False,
            checks={"power_fit": True, "size_fit": True, "throughput_fit": False},
            violations=["Throughput 7.2 tok/s below required 15.0 tok/s"],
        )
        child = search_engine.mutate_for_physical_constraints(cand, fake_result)

        # Should deepen pipeline to raise clock frequency and throughput
        assert child.pipeline_depth > cand.pipeline_depth
        assert child.parent_architecture_id == cand.architecture_id


class TestPhysicalConstraintEvolutionLoop:
    """Verify end-to-end multi-generation evolution loop toward physical constraint satisfaction."""

    def test_closed_loop_satisfaction(self):
        # Envelope where initial baseline fails throughput, but after 2-3 generations passes
        envelope = DevicePhysicalEnvelope(
            enclosure_length_mm=100.0,
            enclosure_width_mm=30.0,
            max_power_w=5.0,
            max_junction_temp_c=85.0,
            min_tokens_per_sec=10.0,
        )
        loop = PhysicalConstraintEvolutionLoop(
            envelope=envelope,
            task_id="L3_MAC_8BIT_SIGNED",
            max_generations=5,
            candidates_per_generation=3,
        )
        report = loop.run()

        assert report.total_generations >= 1
        assert len(report.generation_history) == report.total_generations
        assert report.passed_all_constraints is True
        assert report.winning_candidate is not None
        assert report.winning_candidate.architecture_id.startswith("arch_")
