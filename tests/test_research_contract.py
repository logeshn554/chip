"""
Test Suite for Task 1 — The Real Research Contract.

Verifies the system-wide truthfulness invariants:
    ESTIMATED != MEASURED
    UNKNOWN != PASS
    HYPOTHESIS != VERIFIED
    MOCK != REAL

Every hardware candidate must enforce these three distinct information classes:
    1. HYPOTHESIS: LLM-generated claim, not verified by tools
    2. ESTIMATE: Analytical model with stated assumptions
    3. MEASUREMENT: Grounded EDA tool output (Yosys, Verilator, etc.)
"""

import pytest

from agent.schemas import (
    ConstraintState,
    FeasibilityLabel,
    HardwareArchitectureCandidate,
    InformationClass,
    MetricSet,
    MetricValue,
    TargetSpecification,
    VerificationLevel,
)
from agent.architecture_search import (
    ArchitectureSearchEngine,
    CustomChipGenerator,
    ParetoFrontier,
    SearchOperation,
)
from evaluator.physical_feasibility import PhysicalFeasibilityEngine
from evaluator.reward import RewardEngine


class TestInformationClassAndMetricValue:
    """Verify InformationClass enum and MetricValue provenance dataclass."""

    def test_information_class_enum(self):
        assert InformationClass.HYPOTHESIS == "HYPOTHESIS"
        assert InformationClass.ESTIMATE == "ESTIMATE"
        assert InformationClass.MEASUREMENT == "MEASUREMENT"
        assert InformationClass.UNKNOWN == "UNKNOWN"

    def test_metric_value_defaults(self):
        mv = MetricValue()
        assert mv.value is None
        assert mv.unit == ""
        assert mv.status == InformationClass.UNKNOWN
        assert mv.source == "unknown"
        assert mv.confidence == 0.0
        assert not mv.is_measured
        assert not mv.is_usable_for_ranking
        assert not mv.is_usable_for_training
        assert not mv.is_known

    def test_metric_value_measured(self):
        mv = MetricValue(
            value=1500.0,
            unit="cells",
            status=InformationClass.MEASUREMENT,
            source="yosys_synthesis",
            method="actual_cell_count",
            confidence=1.0,
            tool_version="yosys-0.40",
            evidence="Cell count: 1500 from synthesized gate netlist",
        )
        assert mv.is_measured
        assert mv.is_usable_for_ranking
        assert mv.is_usable_for_training
        assert mv.is_known
        assert mv.value == 1500.0

    def test_metric_value_hypothesis_cannot_rank_or_train(self):
        mv = MetricValue(
            value=500.0,
            unit="cells",
            status=InformationClass.HYPOTHESIS,
            source="llm_proposal",
            confidence=0.1,
        )
        assert not mv.is_measured
        assert not mv.is_usable_for_ranking
        assert not mv.is_usable_for_training
        assert mv.is_known

    def test_metric_value_estimate_can_rank_not_train(self):
        mv = MetricValue(
            value=2.5,
            unit="W",
            status=InformationClass.ESTIMATE,
            source="analytical_scaling_model",
            confidence=0.5,
        )
        assert not mv.is_measured
        assert mv.is_usable_for_ranking
        assert not mv.is_usable_for_training
        assert mv.is_known


class TestMetricSet:
    """Verify MetricSet typed container for the 7 hardware objectives."""

    def test_metric_set_defaults_are_unknown(self):
        ms = MetricSet()
        assert not ms.all_measured()
        assert not ms.all_rankable()
        assert not ms.has_any_known()
        for attr in ("area", "power", "timing", "throughput", "memory", "thermal", "physical_size"):
            mv = getattr(ms, attr)
            assert mv.status == InformationClass.UNKNOWN
            assert mv.value is None

    def test_metric_set_all_measured(self):
        ms = MetricSet()
        for attr in ("area", "power", "timing", "throughput", "memory", "thermal", "physical_size"):
            ms.set_metric(attr, 10.0, status=InformationClass.MEASUREMENT, source="eda_tool")
        assert ms.all_measured()
        assert ms.all_rankable()
        assert ms.has_any_known()

    def test_metric_set_estimates_are_rankable_not_all_measured(self):
        ms = MetricSet()
        for attr in ("area", "power", "timing", "throughput", "memory", "thermal", "physical_size"):
            ms.set_metric(attr, 10.0, status=InformationClass.ESTIMATE, source="analytical_model")
        assert not ms.all_measured()
        assert ms.all_rankable()

    def test_metric_set_hypothesis_is_not_rankable(self):
        ms = MetricSet()
        for attr in ("area", "power", "timing", "throughput", "memory", "thermal", "physical_size"):
            ms.set_metric(attr, 10.0, status=InformationClass.HYPOTHESIS, source="llm_prompt")
        assert not ms.all_measured()
        assert not ms.all_rankable()
        assert ms.has_any_known()


class TestParetoFrontierContract:
    """Verify Pareto frontier enforces provenance truthfulness invariants."""

    @pytest.fixture
    def frontier(self):
        return ParetoFrontier()

    def _make_measured_candidate(self, arch_id: str, area: float, power: float) -> HardwareArchitectureCandidate:
        c = HardwareArchitectureCandidate(architecture_id=arch_id, task_id="t1")
        c.metrics = MetricSet(
            area=MetricValue(value=area, unit="cells", status=InformationClass.MEASUREMENT),
            power=MetricValue(value=power, unit="W", status=InformationClass.MEASUREMENT),
            timing=MetricValue(value=2.0, unit="cycles", status=InformationClass.MEASUREMENT),
            throughput=MetricValue(value=20.0, unit="tok/s", status=InformationClass.MEASUREMENT),
            memory=MetricValue(value=8.0, unit="GB", status=InformationClass.MEASUREMENT),
            thermal=MetricValue(value=50.0, unit="°C", status=InformationClass.MEASUREMENT),
            physical_size=MetricValue(value=500.0, unit="mm2", status=InformationClass.MEASUREMENT),
        )
        return c

    def _make_estimated_candidate(self, arch_id: str, area: float, power: float) -> HardwareArchitectureCandidate:
        c = HardwareArchitectureCandidate(architecture_id=arch_id, task_id="t1")
        c.metrics = MetricSet(
            area=MetricValue(value=area, unit="cells", status=InformationClass.ESTIMATE),
            power=MetricValue(value=power, unit="W", status=InformationClass.ESTIMATE),
            timing=MetricValue(value=2.0, unit="cycles", status=InformationClass.ESTIMATE),
            throughput=MetricValue(value=20.0, unit="tok/s", status=InformationClass.ESTIMATE),
            memory=MetricValue(value=8.0, unit="GB", status=InformationClass.ESTIMATE),
            thermal=MetricValue(value=50.0, unit="°C", status=InformationClass.ESTIMATE),
            physical_size=MetricValue(value=500.0, unit="mm2", status=InformationClass.ESTIMATE),
        )
        return c

    def _make_hypothesis_candidate(self, arch_id: str, area: float, power: float) -> HardwareArchitectureCandidate:
        c = HardwareArchitectureCandidate(architecture_id=arch_id, task_id="t1")
        c.metrics = MetricSet(
            area=MetricValue(value=area, unit="cells", status=InformationClass.HYPOTHESIS),
            power=MetricValue(value=power, unit="W", status=InformationClass.HYPOTHESIS),
            timing=MetricValue(value=2.0, unit="cycles", status=InformationClass.HYPOTHESIS),
            throughput=MetricValue(value=20.0, unit="tok/s", status=InformationClass.HYPOTHESIS),
            memory=MetricValue(value=8.0, unit="GB", status=InformationClass.HYPOTHESIS),
            thermal=MetricValue(value=50.0, unit="°C", status=InformationClass.HYPOTHESIS),
            physical_size=MetricValue(value=500.0, unit="mm2", status=InformationClass.HYPOTHESIS),
        )
        return c

    def test_measured_vs_measured_dominance(self, frontier):
        c_good = self._make_measured_candidate("c_good", area=500.0, power=1.0)
        c_bad = self._make_measured_candidate("c_bad", area=1500.0, power=3.0)

        objs_good = frontier.extract_metric_objects(c_good)
        objs_bad = frontier.extract_metric_objects(c_bad)

        assert frontier.dominates(objs_good, objs_bad) is True
        assert frontier.dominates(objs_bad, objs_good) is False

        assert frontier.add(c_good) is True
        assert frontier.add(c_bad) is False  # Dominated by c_good

    def test_estimated_cannot_dominate_measured(self, frontier):
        """INVARIANT: An analytical ESTIMATE cannot displace a MEASURED candidate."""
        c_measured = self._make_measured_candidate("c_meas", area=1000.0, power=2.0)
        # c_est claims better numbers, but is only an ESTIMATE
        c_estimated = self._make_estimated_candidate("c_est", area=500.0, power=1.0)

        objs_meas = frontier.extract_metric_objects(c_measured)
        objs_est = frontier.extract_metric_objects(c_estimated)

        # ESTIMATE cannot dominate MEASURED
        assert frontier.dominates(objs_est, objs_meas) is False

    def test_measured_can_dominate_estimated(self, frontier):
        """A real MEASUREMENT strictly dominates an ESTIMATE when numbers are better."""
        c_measured = self._make_measured_candidate("c_meas", area=500.0, power=1.0)
        c_estimated = self._make_estimated_candidate("c_est", area=1000.0, power=2.0)

        objs_meas = frontier.extract_metric_objects(c_measured)
        objs_est = frontier.extract_metric_objects(c_estimated)

        assert frontier.dominates(objs_meas, objs_est) is True

    def test_hypothesis_cannot_enter_pareto_frontier(self, frontier):
        """INVARIANT: HYPOTHESIS metrics cannot enter Pareto frontier; tracked as UNRANKED."""
        c_hyp = self._make_hypothesis_candidate("c_hyp", area=100.0, power=0.5)

        # Attempt to add to frontier
        result = frontier.add(c_hyp)
        assert result is False
        assert c_hyp in frontier.get_unranked()
        assert c_hyp not in frontier.get_frontier()

    def test_unknown_cannot_dominate_anything(self, frontier):
        """INVARIANT: Candidates with UNKNOWN metrics cannot establish dominance."""
        c_meas = self._make_measured_candidate("c_meas", area=1000.0, power=2.0)
        c_empty = HardwareArchitectureCandidate(architecture_id="c_empty", task_id="t1")

        objs_meas = frontier.extract_metric_objects(c_meas)
        objs_empty = frontier.extract_metric_objects(c_empty)

        assert frontier.dominates(objs_empty, objs_meas) is False
        assert frontier.dominates(objs_empty, objs_empty) is False
        assert frontier.add(c_empty) is False
        assert c_empty in frontier.get_unranked()


class TestArchitectureSearchResearchContract:
    """Verify architecture search engine respects the research contract."""

    def test_propose_candidates_hypotheses_and_zero_reward(self):
        engine = ArchitectureSearchEngine()
        spec = TargetSpecification()
        candidates = engine.propose_candidates("t1", "Design AI hardware", n=4, target_spec=spec)

        # Seed hypotheses must have reward == 0.0 (no pre-assigned rewards)
        for c in candidates:
            assert c.reward == 0.0, f"Candidate {c.architecture_id} has pre-assigned reward {c.reward}"

        # Commercial/systolic/simd hypotheses must have HYPOTHESIS status
        c_hypotheses = [c for c in candidates if not c.custom_chip]
        for c in c_hypotheses:
            assert c.is_hypothesis is True
            assert c.validation_status == "UNVERIFIED"
            assert c.verification_status == "unverified"
            for attr in ("area", "power", "timing", "throughput", "memory", "thermal", "physical_size"):
                mv = getattr(c.metrics, attr)
                assert mv.status == InformationClass.HYPOTHESIS, f"Metric {attr} had status {mv.status}"

    def test_apply_operation_clears_synthesis_and_tags_hypothesis(self):
        engine = ArchitectureSearchEngine()
        spec = TargetSpecification()

        # Create parent with pseudo synthesis metrics
        parent = HardwareArchitectureCandidate(
            architecture_id="parent",
            task_id="t1",
            actual_synthesis_metrics={"cells": 1000, "area_um2": 50000.0},
            reward=0.85,
            pareto_metrics={"area": 1000.0, "power": 2.0, "timing": 2.0, "throughput": 15.0, "memory": 8.0, "thermal": 55.0, "physical_size": 600.0},
        )

        child = engine.apply_operation(parent, SearchOperation.CHANGE_PRECISION.value, target_spec=spec)

        # Synthesis metrics MUST be cleared
        assert child.actual_synthesis_metrics == {}
        # Reward MUST be reset to 0.0
        assert child.reward == 0.0
        # Must be tagged as hypothesis
        assert child.is_hypothesis is True
        assert child.validation_status == "UNVERIFIED"
        assert child.verification_status == "unverified"

        # Child metrics must carry HYPOTHESIS status
        for attr in ("area", "power", "timing", "throughput", "memory", "thermal", "physical_size"):
            mv = getattr(child.metrics, attr)
            assert mv.status == InformationClass.HYPOTHESIS


class TestRewardEngineResearchContract:
    """Verify RewardEngine rejects HYPOTHESIS metrics in TRAINING mode."""

    def test_strict_training_mode_rejects_hypothesis_area(self):
        engine = RewardEngine()
        res = engine.compute_modular_reward(
            compile_success=True,
            test_pass_rate=1.0,
            formal_status="PASS",
            area=300.0,
            timing_ns=2.0,
            power_uw=500.0,
            evaluation_mode="TRAINING",
            metric_statuses={"area": InformationClass.HYPOTHESIS},
        )
        assert res.is_valid_hardware is False
        assert res.synthesis_score == 0.0
        assert "Area metric status was 'HYPOTHESIS'" in res.breakdown["gate_failed"]

    def test_strict_training_mode_accepts_measured_area(self):
        engine = RewardEngine()
        res = engine.compute_modular_reward(
            compile_success=True,
            test_pass_rate=1.0,
            formal_status="PASS",
            area=300.0,
            timing_ns=2.0,
            power_uw=500.0,
            evaluation_mode="TRAINING",
            metric_statuses={"area": InformationClass.MEASUREMENT},
        )
        assert res.is_valid_hardware is True
        assert res.synthesis_score == 1.0


class TestPhysicalFeasibilityResearchContract:
    """Verify PhysicalFeasibilityEngine labels HYPOTHESIS data as FEASIBLE_HYPOTHESIS."""

    def test_hypothesis_data_produces_feasible_hypothesis_label(self):
        engine = PhysicalFeasibilityEngine(TargetSpecification(max_power_w=10.0, max_length_mm=100.0, max_width_mm=50.0))

        # Candidate with HYPOTHESIS metrics that pass constraints numerically
        cand = HardwareArchitectureCandidate(
            architecture_id="cand_hyp",
            task_id="t1",
            is_hypothesis=True,
            metrics=MetricSet(
                power=MetricValue(value=2.0, unit="W", status=InformationClass.HYPOTHESIS),
                physical_size=MetricValue(value=1000.0, unit="mm2", status=InformationClass.HYPOTHESIS),
                thermal=MetricValue(value=45.0, unit="°C", status=InformationClass.HYPOTHESIS),
                throughput=MetricValue(value=20.0, unit="tok/s", status=InformationClass.HYPOTHESIS),
                memory=MetricValue(value=8.0, unit="GB", status=InformationClass.HYPOTHESIS),
            ),
        )

        report = engine.evaluate_candidate(cand)
        assert report.feasibility == FeasibilityLabel.FEASIBLE_HYPOTHESIS
        assert "HYPOTHESIS" in report.recommendation

    def test_estimated_data_produces_feasible_estimate_label(self):
        engine = PhysicalFeasibilityEngine(TargetSpecification(max_power_w=10.0, max_length_mm=100.0, max_width_mm=50.0))

        # Candidate with ESTIMATE metrics that pass constraints
        cand = HardwareArchitectureCandidate(
            architecture_id="cand_est",
            task_id="t1",
            is_hypothesis=False,
            metrics=MetricSet(
                power=MetricValue(value=2.0, unit="W", status=InformationClass.ESTIMATE),
                physical_size=MetricValue(value=1000.0, unit="mm2", status=InformationClass.ESTIMATE),
                thermal=MetricValue(value=45.0, unit="°C", status=InformationClass.ESTIMATE),
                throughput=MetricValue(value=20.0, unit="tok/s", status=InformationClass.ESTIMATE),
                memory=MetricValue(value=8.0, unit="GB", status=InformationClass.ESTIMATE),
            ),
        )

        report = engine.evaluate_candidate(cand)
        assert report.feasibility == FeasibilityLabel.FEASIBLE_ESTIMATE
