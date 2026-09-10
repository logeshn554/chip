"""
Unit tests for Architecture Search, Mutations, Genealogy Tracking, and Action Routing.
"""

import os
import pytest
from agent.architecture_search import (
    ArchitectureSearchEngine,
    ArchitectureGenealogy,
    HardwareArchitectureCandidate,
    MUTATION_TYPES,
)
from agent.action_router import build_router
from agent.schemas import ActionType, ActionStatus, AgentAction


class TestArchitectureSearch:
    """Test suite for autonomous hardware architecture search."""

    def test_propose_initial_candidates(self):
        """Engine proposes N diverse initial candidates across archetypes."""
        engine = ArchitectureSearchEngine()
        cands = engine.propose_candidates(
            task_id="L3_MAC",
            task_description="8-bit signed MAC with accumulator",
            n=4,
        )
        assert len(cands) == 4
        # Verify archetype diversity
        datapaths = {c.datapath_structure for c in cands}
        assert len(datapaths) >= 3
        # Check attributes
        for c in cands:
            assert c.task_id == "L3_MAC"
            assert c.pipeline_depth >= 1
            assert c.generation == 0
            assert c.architecture_id in engine.genealogy.candidates

    def test_guided_mutations(self):
        """Guided mutations correctly modify architectural parameters."""
        engine = ArchitectureSearchEngine()
        parent = HardwareArchitectureCandidate(
            architecture_id="arch_base",
            task_id="L3_MAC",
            pipeline_depth=1,
            parallelism=1,
            datapath_structure="single_cycle",
            generation=0,
        )

        # Deeper pipeline mutation
        child_pipe = engine.mutate_candidate(parent, "deeper_pipeline")
        assert child_pipe.pipeline_depth == 2
        assert child_pipe.generation == 1
        assert child_pipe.parent_architecture_id == "arch_base"

        # More parallel lanes mutation
        child_lanes = engine.mutate_candidate(parent, "more_parallel_lanes")
        assert child_lanes.parallelism == 2
        assert child_lanes.generation == 1

        # Systolic organization mutation
        child_sys = engine.mutate_candidate(parent, "systolic_organization")
        assert child_sys.datapath_structure == "systolic_processing_elements"
        assert child_sys.buffering_strategy == "weight_stationary"

    def test_genealogy_tracking_and_lineage(self, tmp_path):
        """Genealogy accurately records parent-child edges and exports metadata."""
        genealogy = ArchitectureGenealogy()
        engine = ArchitectureSearchEngine(genealogy=genealogy)

        # Create root
        c0 = HardwareArchitectureCandidate(architecture_id="c0", task_id="L3_MAC", generation=0, reward=0.5)
        genealogy.register_candidate(c0)

        # Create child
        c1 = engine.mutate_candidate(c0, "deeper_pipeline")
        c1.architecture_id = "c1"
        c1.reward = 0.75
        genealogy.register_candidate(c1, parent_id="c0", mutation_type="deeper_pipeline")

        # Create grandchild
        c2 = engine.mutate_candidate(c1, "systolic_organization")
        c2.architecture_id = "c2"
        c2.reward = 0.90
        genealogy.register_candidate(c2, parent_id="c1", mutation_type="systolic_organization")

        # Verify lineage
        lineage = genealogy.get_lineage("c2")
        assert [c.architecture_id for c in lineage] == ["c0", "c1", "c2"]

        # Verify best candidate updated
        assert genealogy.best_by_task["L3_MAC"] == "c2"

        # Verify export and JSON persistence
        meta = genealogy.export_genealogy_metadata()
        assert meta["candidate_count"] == 3
        assert meta["edges_count"] == 2

        json_file = str(tmp_path / "genealogy.json")
        genealogy.save_to_file(json_file)
        assert os.path.exists(json_file)

    def test_candidate_ranking(self):
        """Engine ranks candidates based on reward, verification, and cell efficiency."""
        engine = ArchitectureSearchEngine()
        c_fail = HardwareArchitectureCandidate(
            architecture_id="fail", task_id="t1", reward=0.8, verification_status="failed"
        )
        c_pass_large = HardwareArchitectureCandidate(
            architecture_id="large", task_id="t1", reward=0.8, verification_status="passed",
            actual_synthesis_metrics={"cells": 300}
        )
        c_pass_small = HardwareArchitectureCandidate(
            architecture_id="small", task_id="t1", reward=0.8, verification_status="passed",
            actual_synthesis_metrics={"cells": 100}
        )

        ranked = engine.rank_candidates([c_fail, c_pass_large, c_pass_small])
        # c_pass_small should be first due to cell efficiency tiebreaker
        assert ranked[0].architecture_id == "small"
        assert ranked[1].architecture_id == "large"
        assert ranked[2].architecture_id == "fail"

    @pytest.mark.asyncio
    async def test_action_router_architecture_integration(self):
        """Router executes PROPOSE_ARCHITECTURE and COMPARE_ARCHITECTURES actions."""
        router = build_router()

        # Execute PROPOSE_ARCHITECTURE
        action_prop = AgentAction(
            action_type=ActionType.PROPOSE_ARCHITECTURE,
            params={"task_id": "L3_MAC", "n": 3},
        )
        res_prop = await router.execute(action_prop)
        assert res_prop.status == ActionStatus.SUCCESS
        assert res_prop.metrics["candidates_count"] == 3
        assert len(res_prop.artifacts) == 3

        # Execute COMPARE_ARCHITECTURES
        action_comp = AgentAction(
            action_type=ActionType.COMPARE_ARCHITECTURES,
            params={},
        )
        res_comp = await router.execute(action_comp)
        assert res_comp.status == ActionStatus.SUCCESS
        assert "Architecture Comparison" in res_comp.output
