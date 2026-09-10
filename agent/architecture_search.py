"""
Hardware Architecture Search & Genealogy Engine.

Enables the agent to reason at two levels:
- LEVEL A: Architecture selection (datapath, pipeline, parallelism, buffering, arithmetic)
- LEVEL B: RTL implementation (SystemVerilog generation and verification)

Supports:
- Proposing N diverse architecture candidates per task
- Guided, experience-aware architectural mutations
- Architecture ranking and empirical selection
- Genealogy tracking (parent-child lineages, mutation types, performance evolution)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import random
from typing import Any, Optional

from agent.schemas import HardwareArchitectureCandidate

logger = logging.getLogger(__name__)


# Standard architectural mutation dimensions
MUTATION_TYPES = [
    "wider_datapath",
    "narrower_datapath",
    "deeper_pipeline",
    "shallower_pipeline",
    "more_parallel_lanes",
    "less_parallelism",
    "shared_multiplier",
    "duplicated_multiplier",
    "systolic_organization",
    "simd_organization",
    "local_sram",
    "register_buffering",
    "streaming_interface",
    "staged_memory_access",
]


@dataclass
class GenealogyEdge:
    """Directed edge in the architecture evolution graph."""
    parent_id: str
    child_id: str
    mutation_type: str
    rationale: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ArchitectureGenealogy:
    """Tracks the genealogical tree and evolution history of hardware architectures."""

    def __init__(self):
        self.candidates: dict[str, HardwareArchitectureCandidate] = {}
        self.edges: list[GenealogyEdge] = []
        self.best_by_task: dict[str, str] = {}  # task_id -> best_architecture_id

    def register_candidate(
        self,
        candidate: HardwareArchitectureCandidate,
        parent_id: Optional[str] = None,
        mutation_type: str = "initial",
        rationale: str = "",
    ) -> None:
        """Add candidate to genealogy and track relationship to parent."""
        self.candidates[candidate.architecture_id] = candidate
        if parent_id and parent_id in self.candidates:
            edge = GenealogyEdge(
                parent_id=parent_id,
                child_id=candidate.architecture_id,
                mutation_type=mutation_type,
                rationale=rationale,
            )
            self.edges.append(edge)

        # Update best candidate for task if reward is higher
        tid = candidate.task_id
        if tid not in self.best_by_task:
            self.best_by_task[tid] = candidate.architecture_id
        else:
            current_best = self.candidates[self.best_by_task[tid]]
            if candidate.reward > current_best.reward:
                self.best_by_task[tid] = candidate.architecture_id

    def get_lineage(self, architecture_id: str) -> list[HardwareArchitectureCandidate]:
        """Return full lineage from root ancestor down to this architecture."""
        lineage = []
        curr_id: Optional[str] = architecture_id
        visited = set()
        while curr_id and curr_id in self.candidates and curr_id not in visited:
            visited.add(curr_id)
            cand = self.candidates[curr_id]
            lineage.append(cand)
            curr_id = cand.parent_architecture_id
        return list(reversed(lineage))

    def export_genealogy_metadata(self) -> dict[str, Any]:
        """Export machine-readable genealogy metadata for experiment tracking."""
        return {
            "candidate_count": len(self.candidates),
            "edges_count": len(self.edges),
            "best_architectures": self.best_by_task,
            "candidates": {cid: asdict(c) for cid, c in self.candidates.items()},
            "edges": [asdict(e) for e in self.edges],
        }

    def to_dict(self) -> dict[str, Any]:
        """Alias for export_genealogy_metadata."""
        return self.export_genealogy_metadata()

    def save_to_file(self, filepath: str) -> None:
        """Persist genealogy to a JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.export_genealogy_metadata(), f, indent=2)


class ArchitectureSearchEngine:
    """Autonomous hardware architecture search and mutation engine."""

    def __init__(self, genealogy: Optional[ArchitectureGenealogy] = None):
        self.genealogy = genealogy or ArchitectureGenealogy()

    def propose_candidates(
        self,
        task_id: str,
        task_description: str = "",
        n: int = 4,
        parent_candidate: Optional[HardwareArchitectureCandidate] = None,
        past_experiences: Optional[list[dict[str, Any]]] = None,
    ) -> list[HardwareArchitectureCandidate]:
        """Generate N diverse or mutated architectural candidates for a task.

        If parent_candidate is provided, produces guided mutations.
        Otherwise, samples diverse baseline architectural archetypes.
        """
        candidates: list[HardwareArchitectureCandidate] = []
        seed_val = int(hashlib.sha256(f"{task_id}_{task_description}".encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed_val + (parent_candidate.generation if parent_candidate else 0))

        if parent_candidate is not None:
            # Guided mutation branch
            selected_mutations = rng.sample(MUTATION_TYPES, min(n, len(MUTATION_TYPES)))
            for idx, mut in enumerate(selected_mutations):
                child = self.mutate_candidate(parent_candidate, mut)
                self.genealogy.register_candidate(
                    child,
                    parent_id=parent_candidate.architecture_id,
                    mutation_type=mut,
                    rationale=f"Guided mutation {mut} to explore design trade-offs",
                )
                candidates.append(child)
            return candidates

        # Archetype 1: Minimal Combinational / Low-Latency
        c1_id = f"arch_{task_id}_comb_{seed_val % 10000:04d}"
        c1 = HardwareArchitectureCandidate(
            architecture_id=c1_id,
            task_id=task_id,
            datapath_structure="single_cycle_combinational",
            pipeline_depth=1,
            parallelism=1,
            memory_organization="registers",
            buffering_strategy="single_buffer",
            arithmetic_strategy="standard_signed",
            interface_strategy="direct_valid",
            estimated_resource_requirements={"target_cells": 120, "estimated_latency_cycles": 1},
            generation=0,
            metadata={"archetype": "minimal_area_low_latency"},
        )
        self.genealogy.register_candidate(c1, mutation_type="archetype_minimal")
        candidates.append(c1)

        # Archetype 2: Pipelined / High-Frequency
        c2_id = f"arch_{task_id}_pipe_{seed_val % 10000:04d}"
        c2 = HardwareArchitectureCandidate(
            architecture_id=c2_id,
            task_id=task_id,
            datapath_structure="pipelined_datapath",
            pipeline_depth=2,
            parallelism=1,
            memory_organization="registers",
            buffering_strategy="ping_pong",
            arithmetic_strategy="registered_stages",
            interface_strategy="valid_ready_handshake",
            estimated_resource_requirements={"target_cells": 160, "estimated_latency_cycles": 2},
            generation=0,
            metadata={"archetype": "pipelined_timing_optimized"},
        )
        self.genealogy.register_candidate(c2, mutation_type="archetype_pipelined")
        candidates.append(c2)

        # Archetype 3: Parallel SIMD / High-Throughput
        c3_id = f"arch_{task_id}_simd_{seed_val % 10000:04d}"
        c3 = HardwareArchitectureCandidate(
            architecture_id=c3_id,
            task_id=task_id,
            datapath_structure="parallel_lanes",
            pipeline_depth=1,
            parallelism=2,
            memory_organization="banked_registers",
            buffering_strategy="fifo_queue",
            arithmetic_strategy="vector_parallel",
            interface_strategy="streaming",
            estimated_resource_requirements={"target_cells": 240, "estimated_latency_cycles": 1},
            generation=0,
            metadata={"archetype": "parallel_throughput_optimized"},
        )
        self.genealogy.register_candidate(c3, mutation_type="archetype_simd")
        candidates.append(c3)

        # Archetype 4: Balanced Modular / Standard
        c4_id = f"arch_{task_id}_bal_{seed_val % 10000:04d}"
        c4 = HardwareArchitectureCandidate(
            architecture_id=c4_id,
            task_id=task_id,
            datapath_structure="balanced_modular",
            pipeline_depth=1,
            parallelism=1,
            memory_organization="registers",
            buffering_strategy="single_buffer",
            arithmetic_strategy="standard_signed",
            interface_strategy="valid_ready",
            estimated_resource_requirements={"target_cells": 140, "estimated_latency_cycles": 1},
            generation=0,
            metadata={"archetype": "balanced_standard"},
        )
        self.genealogy.register_candidate(c4, mutation_type="archetype_balanced")
        candidates.append(c4)

        return candidates[:n]

    def mutate_candidate(
        self,
        candidate: HardwareArchitectureCandidate,
        mutation_type: str,
    ) -> HardwareArchitectureCandidate:
        """Apply a specific guided mutation operator to produce a child candidate."""
        new_gen = candidate.generation + 1
        h = abs(hash(f"{candidate.architecture_id}_{mutation_type}_{new_gen}")) % 100000
        child_id = f"arch_mut_{candidate.task_id}_g{new_gen}_{h:05d}"

        # Copy baseline attributes
        datapath = candidate.datapath_structure
        pipe_depth = candidate.pipeline_depth
        parallelism = candidate.parallelism
        mem_org = candidate.memory_organization
        buffering = candidate.buffering_strategy
        arithmetic = candidate.arithmetic_strategy
        interface = candidate.interface_strategy
        est_res = dict(candidate.estimated_resource_requirements)

        # Apply mutation
        if mutation_type == "deeper_pipeline":
            pipe_depth = min(pipe_depth + 1, 4)
            arithmetic = "multi_stage_registered"
            est_res["target_cells"] = est_res.get("target_cells", 150) + 25
            est_res["estimated_latency_cycles"] = pipe_depth
        elif mutation_type == "shallower_pipeline":
            pipe_depth = max(pipe_depth - 1, 1)
            arithmetic = "standard_signed"
            est_res["target_cells"] = max(est_res.get("target_cells", 150) - 20, 50)
            est_res["estimated_latency_cycles"] = pipe_depth
        elif mutation_type == "more_parallel_lanes":
            parallelism = min(parallelism * 2, 8)
            datapath = "parallel_simd_lanes"
            est_res["target_cells"] = est_res.get("target_cells", 150) * 1.8
        elif mutation_type == "less_parallelism":
            parallelism = max(parallelism // 2, 1)
            if parallelism == 1:
                datapath = "single_lane"
            est_res["target_cells"] = max(est_res.get("target_cells", 150) * 0.6, 50)
        elif mutation_type == "systolic_organization":
            datapath = "systolic_processing_elements"
            pipe_depth = max(pipe_depth, 2)
            buffering = "weight_stationary"
            interface = "systolic_streaming"
        elif mutation_type == "simd_organization":
            datapath = "simd_vector"
            parallelism = max(parallelism, 2)
            interface = "vector_stream"
        elif mutation_type == "local_sram":
            mem_org = "embedded_sram"
            buffering = "double_buffering"
        elif mutation_type == "register_buffering":
            mem_org = "flip_flop_registers"
            buffering = "single_buffer"
        elif mutation_type == "streaming_interface":
            interface = "axis_streaming"
            buffering = "fifo_queue"
        elif mutation_type == "staged_memory_access":
            interface = "dma_burst_ready"
            buffering = "ring_buffer"
        elif mutation_type == "quantized_int4_arithmetic":
            arithmetic = "quantized_int4_shared"
            est_res["target_cells"] = max(est_res.get("target_cells", 150) * 0.55, 45)
        elif mutation_type == "shared_multiplier":
            arithmetic = "time_multiplexed_shared"
            est_res["target_cells"] = max(est_res.get("target_cells", 150) * 0.70, 50)
        elif mutation_type == "streaming_dataflow":
            mem_org = "streaming"
            buffering = "double_buffering"
            interface = "axis_streaming"

        child = HardwareArchitectureCandidate(
            architecture_id=child_id,
            task_id=candidate.task_id,
            parent_architecture_id=candidate.architecture_id,
            datapath_structure=datapath,
            pipeline_depth=pipe_depth,
            parallelism=parallelism,
            memory_organization=mem_org,
            buffering_strategy=buffering,
            arithmetic_strategy=arithmetic,
            interface_strategy=interface,
            estimated_resource_requirements=est_res,
            generation=new_gen,
            metadata={"mutation_applied": mutation_type, "parent": candidate.architecture_id},
        )
        return child

    def mutate_for_physical_constraints(
        self,
        candidate: HardwareArchitectureCandidate,
        constraint_result: Any,
    ) -> HardwareArchitectureCandidate:
        """Derive a targeted architectural mutation guided directly by physical constraint diagnostics."""
        checks = getattr(constraint_result, "checks", {})

        # 1. Size / PCB area or Power violations take highest priority (hard physical wall)
        if not checks.get("size_fit", True) or not checks.get("power_fit", True):
            if candidate.parallelism > 1:
                mutation = "less_parallelism"
            elif "quantized" not in candidate.arithmetic_strategy:
                mutation = "quantized_int4_arithmetic"
            elif "shared" not in candidate.arithmetic_strategy:
                mutation = "shared_multiplier"
            else:
                mutation = "shallower_pipeline"

        # 2. Thermal violation: switch to streaming dataflow or lower clock burden
        elif not checks.get("thermal_fit", True):
            if candidate.memory_organization != "streaming":
                mutation = "streaming_dataflow"
            else:
                mutation = "quantized_int4_arithmetic"

        # 3. Throughput violation: deepen pipeline or increase parallelism
        elif not checks.get("throughput_fit", True):
            if candidate.pipeline_depth < 3:
                mutation = "deeper_pipeline"
            elif candidate.parallelism < 4:
                mutation = "more_parallel_lanes"
            else:
                mutation = "systolic_organization"

        else:
            # All mandatory constraints pass -> explore PPA optimization (e.g. local SRAM or systolic)
            mutation = "local_sram"

        child = self.mutate_candidate(candidate, mutation)
        self.genealogy.register_candidate(child, parent_id=candidate.architecture_id, mutation_type=mutation)
        return child

    def rank_candidates(
        self,
        candidates: list[HardwareArchitectureCandidate],
    ) -> list[HardwareArchitectureCandidate]:
        """Rank candidates based on grounded reward, verification status, and efficiency."""
        def _score(c: HardwareArchitectureCandidate) -> float:
            score = c.reward
            if c.verification_status == "passed":
                score += 1.0
            elif c.verification_status == "failed":
                score -= 2.0
            # Cell efficiency tiebreaker
            actual_cells = c.actual_synthesis_metrics.get("cells")
            if actual_cells and actual_cells > 0:
                score += 10.0 / actual_cells
            return score

        return sorted(candidates, key=_score, reverse=True)
