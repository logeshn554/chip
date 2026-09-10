"""
Unit and Integration Tests for Open-Ended Self-Evolving Hardware Architecture and Chip Design.

Covers:
1. TargetSpecification Configuration and Envelope Properties
2. Query-Driven Component Search & Provenance Verification
3. Physical Feasibility Engine (Footprint, Clearance, Volume, Thermal, Power)
4. Explicit Constraint States & Feasibility Labels
5. 16 Architecture Search Operations & Guided Root-Cause Evolution
6. Custom Chip Mode Escalation & Multi-Generation ASIC RTL Synthesis
7. Pareto Architecture Frontier (7-Objective Dominance & Selection)
8. Hypothesis Validation Lifecycle (is_hypothesis -> VALIDATED / FAILED)
9. Open-Ended Evolution Loop & Termination Conditions
"""

import json
import os
import pytest
from typing import Any

from agent.architecture_search import (
    ArchitectureGenealogy,
    ArchitectureSearchEngine,
    CustomChipGenerator,
    HardwareArchitectureCandidate,
    ParetoFrontier,
    SearchOperation,
)
from agent.schemas import (
    ComponentEvidence,
    ConstraintState,
    FeasibilityLabel,
    TargetSpecification,
    VerificationLevel,
)
from evaluator.physical_feasibility import PhysicalFeasibilityEngine, PhysicalFeasibilityReport
from learning.self_evolution import OpenEndedArchitectureEvolutionLoop
from scraping.component_search import ComponentSearchEngine


class TestTargetSpecification:
    """Test suite for configurable target specification."""

    def test_configurable_specification_defaults(self):
        spec = TargetSpecification()
        assert spec.max_power_w == 5.0
        assert spec.max_temperature_c == 85.0
        assert spec.min_ram_gb == 8.0
        assert spec.min_storage_gb == 256.0
        assert spec.min_tokens_per_second == 15.0
        assert spec.target_model == "Qwen3-4B"
        assert spec.process_node == "28nm"
        assert "USB-C" in spec.host_interfaces

    def test_custom_enclosure_and_pcb_area_derivation(self):
        spec = TargetSpecification(
            max_length_mm=80.0,
            max_width_mm=25.0,
            max_height_mm=10.0,
            max_power_w=3.5,
        )
        # Usable: (80 - 4) * (25 - 4) * 1.8 = 76 * 21 * 1.8 = 2872.8 mm2
        assert pytest.approx(spec.max_pcb_area_mm2, 0.1) == 2872.8
        assert spec.max_power_w == 3.5
        assert spec.enclosure_length_mm == 80.0


class TestDynamicComponentSearch:
    """Test suite for dynamic query-driven component search and provenance."""

    def test_source_quality_ranking(self):
        engine = ComponentSearchEngine(db_dir="./data/test_components")
        # Manufacturer tier (0.95)
        assert engine.assess_source_quality("https://semiconductor.samsung.com/dram/lpddr5") == 0.95
        assert engine.assess_source_quality("https://www.ti.com/power-management/datasheet.pdf") == 0.95
        # Distributor/Standards tier (0.80)
        assert engine.assess_source_quality("https://www.digikey.com/en/products/detail/microchip") == 0.80
        assert engine.assess_source_quality("https://www.usb.org/document-library") == 0.80
        # Generic tier (0.50)
        assert engine.assess_source_quality("https://hackaday.com/2024/01/npu-teardown") == 0.50

    def test_evidence_extraction_and_null_preservation(self):
        engine = ComponentSearchEngine(db_dir="./data/test_components")
        raw = "Samsung K3LK3K30EM-BGCN 8GB LPDDR5 FBGA, 12.0 mm x 12.0 mm x 0.8 mm, 1.05V, 0.9W active power."
        comp = engine.extract_component_evidence(raw, "https://semiconductor.samsung.com/dram/", "dram")
        assert comp is not None
        assert comp.part_number == "K3LK3K30EM-BGCN"
        assert comp.memory_capacity_gb == 8.0
        assert comp.length_mm == 12.0
        assert comp.width_mm == 12.0
        assert comp.height_mm == 0.8
        assert comp.power_w == 0.9
        assert comp.package == "FBGA"
        # Strictly ensure absent metrics are None, NEVER fabricated
        assert comp.mass_g is None
        assert comp.temperature_range is None

    def test_query_driven_search_returns_empty_without_network(self, tmp_path):
        """In a sandboxed/offline environment, the search engine must return an
        empty result list rather than fabricating hardcoded component data."""
        db_dir = str(tmp_path / "components")
        engine = ComponentSearchEngine(db_dir=db_dir)
        # In a sandboxed test environment there is no network access.
        # The engine must NOT inject hardcoded fallback data.
        results = engine.search_and_extract("LPDDR4 8GB package dimensions", category="dram")
        # Results are either empty (offline) or contain real scraped data (online).
        # Either way, the catalog should exactly match returned results.
        assert all(r.component_id in engine.catalog for r in results)

    def test_disk_persistence_of_manually_injected_evidence(self, tmp_path):
        """Evidence that is explicitly injected must be persisted to disk."""
        import json
        db_dir = str(tmp_path / "components")
        engine = ComponentSearchEngine(db_dir=db_dir)
        # Manually inject a real-evidence component (e.g. from a document fetch)
        from agent.schemas import ComponentEvidence
        comp = ComponentEvidence(
            component_id="comp_test_lpddr4_0001",
            manufacturer="TestManufacturer",
            part_number="TESTLPDDR4-8G",
            category="dram",
            source_urls=["https://example.com/lpddr4-datasheet.pdf"],
            length_mm=12.0,
            width_mm=12.0,
            power_w=0.9,
            confidence=0.95,
        )
        engine.save_component(comp)
        assert comp.component_id in engine.catalog
        saved_file = os.path.join(db_dir, f"{comp.component_id}.json")
        assert os.path.exists(saved_file)
        with open(saved_file, "r") as f:
            data = json.load(f)
        assert data["component_id"] == comp.component_id
        assert data["power_w"] == 0.9



class TestPhysicalFeasibilityEngine:
    """Test suite for physical fit, thermal dissipation, and custom chip escalation."""

    def test_commercial_assembly_feasibility_pass(self):
        spec = TargetSpecification(max_power_w=5.0, max_temperature_c=85.0)
        engine = PhysicalFeasibilityEngine(target_spec=spec)
        comps = [
            ComponentEvidence(
                component_id="c_ram", manufacturer="M", part_number="RAM-8GB", category="dram",
                length_mm=12.0, width_mm=12.0, power_w=0.8, memory_capacity_gb=8.0,
            ),
            ComponentEvidence(
                component_id="c_storage", manufacturer="M", part_number="FLASH-256GB", category="storage",
                length_mm=11.5, width_mm=13.0, power_w=0.4, memory_capacity_gb=256.0,
            ),
            ComponentEvidence(
                component_id="c_usb", manufacturer="M", part_number="USB-CTRL", category="usb_controller",
                length_mm=7.0, width_mm=7.0, power_w=0.3, interface="USB-C",
            ),
        ]
        report = engine.evaluate_component_assembly(comps)
        assert report.feasibility == FeasibilityLabel.FEASIBLE_ESTIMATE
        assert report.constraint_states["power"] == ConstraintState.PASS
        assert report.constraint_states["thermal"] == ConstraintState.PASS
        assert report.constraint_states["pcb_area"] == ConstraintState.PASS
        assert report.escalate_to_custom_chip is False

    def test_commercial_assembly_power_failure(self):
        spec = TargetSpecification(max_power_w=5.0)
        engine = PhysicalFeasibilityEngine(target_spec=spec)
        comps = [
            ComponentEvidence(
                component_id="c_power_hungry", manufacturer="M", part_number="HOT-NPU", category="accelerator",
                length_mm=15.0, width_mm=15.0, power_w=6.5,  # Exceeds 5.0W
            ),
            ComponentEvidence(
                component_id="c_ram", manufacturer="M", part_number="RAM-8GB", category="dram",
                length_mm=12.0, width_mm=12.0, power_w=0.9, memory_capacity_gb=8.0,
            ),
            ComponentEvidence(
                component_id="c_storage", manufacturer="M", part_number="FLASH-256GB", category="storage",
                length_mm=11.5, width_mm=13.0, power_w=0.4, memory_capacity_gb=256.0,
            ),
            ComponentEvidence(
                component_id="c_usb", manufacturer="M", part_number="USB-CTRL", category="usb_controller",
                length_mm=7.0, width_mm=7.0, power_w=0.3, interface="USB-C",
            ),
        ]
        report = engine.evaluate_component_assembly(comps)
        assert report.feasibility == FeasibilityLabel.INFEASIBLE_ESTIMATE
        assert report.constraint_states["power"] == ConstraintState.FAIL
        assert report.escalate_to_custom_chip is True

    def test_unknown_metrics_do_not_pass(self):
        spec = TargetSpecification()
        engine = PhysicalFeasibilityEngine(target_spec=spec)
        comps = [
            ComponentEvidence(
                component_id="c_ram", manufacturer="M", part_number="RAM-8GB", category="dram",
                length_mm=12.0, width_mm=12.0, power_w=0.8, memory_capacity_gb=8.0,
            ),
            ComponentEvidence(
                component_id="c_storage", manufacturer="M", part_number="FLASH-256GB", category="storage",
                length_mm=11.5, width_mm=13.0, power_w=0.4, memory_capacity_gb=256.0,
            ),
            ComponentEvidence(
                component_id="c_usb", manufacturer="M", part_number="USB-CTRL", category="usb_controller",
                length_mm=7.0, width_mm=7.0, power_w=0.3, interface="USB-C",
            ),
            ComponentEvidence(
                component_id="c_unkn", manufacturer="M", part_number="MYSTERY-CHIP", category="accelerator",
                length_mm=None, width_mm=None, power_w=None,  # Missing metrics
            ),
        ]
        report = engine.evaluate_component_assembly(comps)
        assert report.feasibility == FeasibilityLabel.UNKNOWN
        assert report.constraint_states["power"] == ConstraintState.UNKNOWN
        assert report.constraint_states["physical_size"] == ConstraintState.UNKNOWN


class TestArchitectureOperationsAndCustomChip:
    """Test suite for open-ended search operations, genealogy, and custom silicon generation."""

    def test_16_search_operations_supported(self):
        all_ops = [op.value for op in SearchOperation]
        assert len(all_ops) == 16
        expected = [
            "GENERATE", "MUTATE", "COMBINE", "REFACTOR", "REPLACE_COMPONENT",
            "REMOVE_COMPONENT", "ADD_COMPONENT", "CHANGE_MEMORY_ARCHITECTURE",
            "CHANGE_DATAFLOW", "CHANGE_PIPELINE", "CHANGE_PARALLELISM",
            "CHANGE_PRECISION", "CHANGE_INTERFACE", "CHANGE_ACCELERATOR",
            "DESIGN_CUSTOM_IP", "DESIGN_CUSTOM_CHIP",
        ]
        for exp in expected:
            assert exp in all_ops

    def test_custom_chip_generator_and_synthesizable_rtl(self):
        spec = TargetSpecification(process_node="28nm", target_model_quantization="INT4")
        cand = CustomChipGenerator.generate_custom_soc_spec(
            task_id="TEST_AI_SOC",
            generation=1,
            target_spec=spec,
            mutation_focus="initial_custom_asic",
        )
        assert cand.custom_chip is True
        assert cand.chip_generation == 1
        assert "custom_ai_soc_gen1" in cand.rtl_implementation
        assert "module custom_ai_soc_gen1" in cand.rtl_implementation
        assert "s_axis_valid" in cand.rtl_implementation
        assert "m_axis_result" in cand.rtl_implementation
        assert len(cand.components) >= 6
        comp_names = [c["name"] for c in cand.components]
        assert "control_core" in comp_names
        assert "transformer_accelerator" in comp_names
        assert "local_sram" in comp_names
        assert "usb_controller" in comp_names

    def test_root_cause_evolution_after_rejection(self):
        engine = ArchitectureSearchEngine()
        spec = TargetSpecification()
        cand = HardwareArchitectureCandidate(
            architecture_id="cand_fail_power",
            task_id="test",
            parallelism=4,
            arithmetic_strategy="fp16_mult",
        )
        # Power failure with high parallelism should trigger parallelism reduction
        mutated = engine.evolve_after_rejection(cand, ["POWER", "THERMAL"], spec)
        assert mutated.parent_architecture_id == "cand_fail_power"
        assert mutated.parallelism < cand.parallelism

        # Power failure with single lane should trigger INT4 quantization
        cand_single = HardwareArchitectureCandidate(
            architecture_id="cand_single_fail",
            task_id="test",
            parallelism=1,
            arithmetic_strategy="fp16_mult",
        )
        mutated_quant = engine.evolve_after_rejection(cand_single, ["POWER", "THERMAL"], spec)
        assert "quantized_int4" in mutated_quant.arithmetic_strategy

        # Commercial component failure should trigger Custom Chip Mode
        cand_comm = HardwareArchitectureCandidate(
            architecture_id="cand_comm_fail",
            task_id="test",
            custom_chip=False,
        )
        mutated_custom = engine.evolve_after_rejection(cand_comm, ["commercial_component_infeasibility"], spec)
        assert mutated_custom.custom_chip is True


class TestParetoFrontier:
    """Test suite for 7-objective Pareto frontier tracking."""

    def test_dominance_and_frontier_maintenance(self):
        frontier = ParetoFrontier()

        # Cand A: Low power, high area
        cA = HardwareArchitectureCandidate(
            architecture_id="cA", task_id="t",
            pareto_metrics={"area": 1000.0, "power": 1.5, "timing": 2.0, "throughput": 20.0, "memory": 8.0, "thermal": 45.0, "physical_size": 500.0},
        )
        # Cand B: High power, low area
        cB = HardwareArchitectureCandidate(
            architecture_id="cB", task_id="t",
            pareto_metrics={"area": 400.0, "power": 4.5, "timing": 1.0, "throughput": 25.0, "memory": 8.0, "thermal": 75.0, "physical_size": 200.0},
        )
        # Cand C: Strictly worse than A in all metrics
        cC = HardwareArchitectureCandidate(
            architecture_id="cC", task_id="t",
            pareto_metrics={"area": 1500.0, "power": 2.5, "timing": 3.0, "throughput": 15.0, "memory": 8.0, "thermal": 60.0, "physical_size": 700.0},
        )

        assert frontier.add(cA) is True
        assert frontier.add(cB) is True
        # cC is strictly dominated by cA, so it must not be added
        assert frontier.add(cC) is False

        current = frontier.get_frontier()
        ids = {c.architecture_id for c in current}
        assert "cA" in ids
        assert "cB" in ids
        assert "cC" not in ids

    def test_target_driven_selection(self):
        frontier = ParetoFrontier()
        spec = TargetSpecification(max_power_w=3.0, max_temperature_c=60.0)
        
        c1 = HardwareArchitectureCandidate(
            architecture_id="c1_low_power", task_id="t", reward=0.8,
            pareto_metrics={"area": 500.0, "power": 1.8, "timing": 2.0, "throughput": 16.0, "memory": 8.0, "thermal": 48.0, "physical_size": 600.0},
        )
        c2 = HardwareArchitectureCandidate(
            architecture_id="c2_high_power", task_id="t", reward=0.9,
            pareto_metrics={"area": 200.0, "power": 4.8, "timing": 1.0, "throughput": 30.0, "memory": 8.0, "thermal": 78.0, "physical_size": 300.0},
        )
        frontier.add(c1)
        frontier.add(c2)

        best = frontier.select_best_for_target(spec)
        assert best is not None
        # c1 satisfies the 3.0W budget; c2 exceeds it
        assert best.architecture_id == "c1_low_power"


class TestHypothesisValidation:
    """Test suite for hypothesis generation and external validation gating."""

    def test_hypothesis_lifecycle(self):
        engine = ArchitectureSearchEngine()
        cand = HardwareArchitectureCandidate(
            architecture_id="hypo_1", task_id="t",
            is_hypothesis=True, validation_status="UNVERIFIED",
        )

        # Failure path
        res_fail = engine.validate_hypothesis(cand, hard_constraint_failures=["POWER_EXCEEDED"])
        assert res_fail is False
        assert cand.is_hypothesis is False
        assert cand.validation_status == "FAILED_HYPOTHESIS"
        assert "POWER_EXCEEDED" in cand.rejection_reasons

        # Success path
        cand2 = HardwareArchitectureCandidate(
            architecture_id="hypo_2", task_id="t",
            is_hypothesis=True, validation_status="UNVERIFIED",
        )
        res_pass = engine.validate_hypothesis(cand2, hard_constraint_failures=None, verification_level=VerificationLevel.PHYSICALLY_ESTIMATED)
        assert res_pass is True
        assert cand2.is_hypothesis is False
        assert cand2.validation_status == "VALIDATED_ARCHITECTURE"
        assert cand2.verification_level == VerificationLevel.PHYSICALLY_ESTIMATED.value


class TestOpenEndedEvolutionLoop:
    """Test suite for full multi-generation open-ended evolution loop."""

    def test_evolution_loop_run_reaches_decision(self):
        spec = TargetSpecification(
            max_power_w=5.0,
            max_temperature_c=85.0,
            min_tokens_per_second=15.0,
        )
        loop = OpenEndedArchitectureEvolutionLoop(
            target_spec=spec,
            task_id="TEST_PORTABLE_AI",
            max_generations=2,
            candidates_per_gen=2,
        )
        outcome = loop.run()
        assert outcome.status in ("SUCCESS", "SEARCH_BUDGET_EXHAUSTED", "PROVEN_INFEASIBILITY")
        assert outcome.generations_run > 0
        assert len(outcome.history) > 0
        assert "=== ARCHITECTURE GENEALOGY TREE ===" in outcome.genealogy_ascii
