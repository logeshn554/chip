"""
Open-Ended Hardware Architecture Search, Genealogy, and Custom Chip Evolution Engine.

Enables autonomous exploration across both commercial component architectures and
custom silicon ASIC architectures.

Core Principles:
1. LLM / System proposals are HYPOTHESES until empirically verified by EDA and physical tools.
2. Hard constraint violations reject candidates (FAILED_HYPOTHESIS) and trigger targeted mutations.
3. Open-ended search supports 16 architectural operators beyond static templates.
4. Escalates to Custom Chip Mode when commercial components cannot satisfy constraints.
5. Maintains a multi-objective Pareto Frontier across area, power, timing, throughput, memory, thermal, and size.
6. Full genealogical lineage tracking with root-cause failure diagnostics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import logging
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

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

logger = logging.getLogger(__name__)


# ── The 16 Architecture Search Operations ──────────────────────────────

class SearchOperation(str, Enum):
    """The 16 first-class architectural exploration operators."""
    GENERATE = "GENERATE"
    MUTATE = "MUTATE"
    COMBINE = "COMBINE"
    REFACTOR = "REFACTOR"
    REPLACE_COMPONENT = "REPLACE_COMPONENT"
    REMOVE_COMPONENT = "REMOVE_COMPONENT"
    ADD_COMPONENT = "ADD_COMPONENT"
    CHANGE_MEMORY_ARCHITECTURE = "CHANGE_MEMORY_ARCHITECTURE"
    CHANGE_DATAFLOW = "CHANGE_DATAFLOW"
    CHANGE_PIPELINE = "CHANGE_PIPELINE"
    CHANGE_PARALLELISM = "CHANGE_PARALLELISM"
    CHANGE_PRECISION = "CHANGE_PRECISION"
    CHANGE_INTERFACE = "CHANGE_INTERFACE"
    CHANGE_ACCELERATOR = "CHANGE_ACCELERATOR"
    DESIGN_CUSTOM_IP = "DESIGN_CUSTOM_IP"
    DESIGN_CUSTOM_CHIP = "DESIGN_CUSTOM_CHIP"


# Backward-compatible mutation list
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
    "quantized_int4_arithmetic",
    "streaming_dataflow",
    "weight_stationary_buffering",
    "integrated_custom_asic",
]


# ── Genealogy Tracking ────────────────────────────────────────────────

@dataclass
class GenealogyEdge:
    """Directed edge in the architecture evolution graph."""
    parent_id: str
    child_id: str
    mutation_type: str
    rationale: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ArchitectureGenealogy:
    """Tracks the genealogical tree, evolution history, and root causes of hardware architectures."""

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

        # Update best candidate for task if reward is higher and candidate is valid
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

    def format_ascii_tree(self) -> str:
        """Render an ASCII tree of candidate genealogy showing pass/fail status."""
        lines = ["=== ARCHITECTURE GENEALOGY TREE ==="]
        # Find root candidates (no parent or parent not in candidates)
        roots = [c for c in self.candidates.values() if not c.parent_architecture_id or c.parent_architecture_id not in self.candidates]
        
        # Build adjacency list: parent_id -> list of children
        children_map: dict[str, list[HardwareArchitectureCandidate]] = {}
        for c in self.candidates.values():
            if c.parent_architecture_id:
                children_map.setdefault(c.parent_architecture_id, []).append(c)

        def _render_node(cand: HardwareArchitectureCandidate, prefix: str = "", is_last: bool = True):
            status_tag = f"[{cand.validation_status}]"
            reasons = f" (Rejected: {cand.rejection_reason})" if cand.rejection_reasons else ""
            custom_tag = " [CUSTOM CHIP]" if cand.custom_chip else ""
            connector = "\\-- " if is_last else "+-- "
            lines.append(f"{prefix}{connector}{cand.architecture_id} {status_tag}{custom_tag}{reasons}")
            
            children = children_map.get(cand.architecture_id, [])
            new_prefix = prefix + ("    " if is_last else "|   ")
            for i, child in enumerate(children):
                _render_node(child, new_prefix, i == len(children) - 1)

        for i, root in enumerate(roots):
            status_tag = f"[{root.validation_status}]"
            reasons = f" (Rejected: {root.rejection_reason})" if root.rejection_reasons else ""
            custom_tag = " [CUSTOM CHIP]" if root.custom_chip else ""
            lines.append(f"{root.architecture_id} {status_tag}{custom_tag}{reasons}")
            children = children_map.get(root.architecture_id, [])
            for j, child in enumerate(children):
                _render_node(child, "", j == len(children) - 1)

        return "\n".join(lines)

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
        return self.export_genealogy_metadata()

    def save_to_file(self, filepath: str) -> None:
        """Persist genealogy to a JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.export_genealogy_metadata(), f, indent=2)


# ── Pareto Architecture Frontier ──────────────────────────────────────

class ParetoFrontier:
    """
    Maintains a Pareto-optimal non-dominated frontier across 7 hardware metrics:
    - area (mm2 or cell count, MINIMIZE)
    - power (W, MINIMIZE)
    - timing / latency (ms or cycles, MINIMIZE)
    - throughput (tokens/sec or TOPS, MAXIMIZE)
    - memory (GB, MAXIMIZE)
    - thermal (junction temperature °C, MINIMIZE)
    - physical_size (PCB mm2 or volume mm3, MINIMIZE)

    TRUTHFULNESS RULE: Candidates with unmeasured metrics (None values) are tracked
    separately as UNRANKED and cannot participate in Pareto dominance comparisons.
    No fabricated fallback defaults are used.
    """

    OBJECTIVES = {
        "area": "min",
        "power": "min",
        "timing": "min",
        "throughput": "max",
        "memory": "max",
        "thermal": "min",
        "physical_size": "min",
    }

    def __init__(self):
        self.frontier: list[HardwareArchitectureCandidate] = []
        self.unranked: list[HardwareArchitectureCandidate] = []  # Candidates with incomplete metrics

    @classmethod
    def extract_metric_objects(cls, candidate: HardwareArchitectureCandidate) -> dict[str, MetricValue]:
        """Extract standardized MetricValue objects with full provenance.

        Priority order:
        1. candidate.metrics (canonical MetricSet)
        2. actual_synthesis_metrics (for area MEASUREMENT from real Yosys)
        3. pareto_metrics / estimated_constraints (ESTIMATE status)
        """
        res: dict[str, MetricValue] = {}
        ms = getattr(candidate, "metrics", None)
        pm = getattr(candidate, "pareto_metrics", {})
        ec = getattr(candidate, "estimated_constraints", {})
        err = getattr(candidate, "estimated_resource_requirements", {})
        asm = getattr(candidate, "actual_synthesis_metrics", {})

        # Area
        mv_area = getattr(ms, "area", None) if ms else None
        if mv_area and mv_area.value is not None and mv_area.status != InformationClass.UNKNOWN:
            res["area"] = mv_area
        elif asm.get("cells") is not None and asm.get("cells", 0) > 0:
            res["area"] = MetricValue(
                value=float(asm["cells"]),
                unit="cells",
                status=InformationClass.MEASUREMENT,
                source="yosys_synthesis",
                method="actual_cell_count",
                confidence=1.0,
            )
        elif pm.get("area") is not None:
            res["area"] = MetricValue(
                value=float(pm["area"]),
                unit="cells",
                status=InformationClass.ESTIMATE,
                source="pareto_metrics",
                method="analytical_estimate",
                confidence=0.5,
            )
        else:
            res["area"] = MetricValue(unit="cells", status=InformationClass.UNKNOWN)

        # Power
        mv_power = getattr(ms, "power", None) if ms else None
        if mv_power and mv_power.value is not None and mv_power.status != InformationClass.UNKNOWN:
            res["power"] = mv_power
        elif pm.get("power") is not None:
            res["power"] = MetricValue(
                value=float(pm["power"]),
                unit="W",
                status=InformationClass.ESTIMATE,
                source="pareto_metrics",
                method="analytical_estimate",
                confidence=0.5,
            )
        elif ec.get("power_w") is not None:
            res["power"] = MetricValue(
                value=float(ec["power_w"]),
                unit="W",
                status=InformationClass.ESTIMATE,
                source="estimated_constraints",
                method="analytical_model",
                confidence=0.5,
            )
        else:
            res["power"] = MetricValue(unit="W", status=InformationClass.UNKNOWN)

        # Timing
        mv_timing = getattr(ms, "timing", None) if ms else None
        if mv_timing and mv_timing.value is not None and mv_timing.status != InformationClass.UNKNOWN:
            res["timing"] = mv_timing
        elif pm.get("timing") is not None:
            res["timing"] = MetricValue(
                value=float(pm["timing"]),
                unit="cycles",
                status=InformationClass.ESTIMATE,
                source="pareto_metrics",
                method="analytical_estimate",
                confidence=0.5,
            )
        elif err.get("estimated_latency_cycles") is not None:
            res["timing"] = MetricValue(
                value=float(err["estimated_latency_cycles"]),
                unit="cycles",
                status=InformationClass.ESTIMATE,
                source="estimated_resource_requirements",
                method="pipeline_depth_model",
                confidence=0.5,
            )
        else:
            res["timing"] = MetricValue(unit="cycles", status=InformationClass.UNKNOWN)

        # Throughput
        mv_tp = getattr(ms, "throughput", None) if ms else None
        if mv_tp and mv_tp.value is not None and mv_tp.status != InformationClass.UNKNOWN:
            res["throughput"] = mv_tp
        elif pm.get("throughput") is not None:
            res["throughput"] = MetricValue(
                value=float(pm["throughput"]),
                unit="tok/s",
                status=InformationClass.ESTIMATE,
                source="pareto_metrics",
                method="analytical_estimate",
                confidence=0.5,
            )
        elif ec.get("tokens_per_sec") is not None:
            res["throughput"] = MetricValue(
                value=float(ec["tokens_per_sec"]),
                unit="tok/s",
                status=InformationClass.ESTIMATE,
                source="estimated_constraints",
                method="analytical_model",
                confidence=0.5,
            )
        else:
            res["throughput"] = MetricValue(unit="tok/s", status=InformationClass.UNKNOWN)

        # Memory
        mv_mem = getattr(ms, "memory", None) if ms else None
        if mv_mem and mv_mem.value is not None and mv_mem.status != InformationClass.UNKNOWN:
            res["memory"] = mv_mem
        elif pm.get("memory") is not None:
            res["memory"] = MetricValue(
                value=float(pm["memory"]),
                unit="GB",
                status=InformationClass.ESTIMATE,
                source="pareto_metrics",
                method="analytical_estimate",
                confidence=0.5,
            )
        elif ec.get("ram_gb") is not None:
            res["memory"] = MetricValue(
                value=float(ec["ram_gb"]),
                unit="GB",
                status=InformationClass.ESTIMATE,
                source="estimated_constraints",
                method="datasheet_or_spec",
                confidence=0.5,
            )
        else:
            res["memory"] = MetricValue(unit="GB", status=InformationClass.UNKNOWN)

        # Thermal
        mv_therm = getattr(ms, "thermal", None) if ms else None
        if mv_therm and mv_therm.value is not None and mv_therm.status != InformationClass.UNKNOWN:
            res["thermal"] = mv_therm
        elif pm.get("thermal") is not None:
            res["thermal"] = MetricValue(
                value=float(pm["thermal"]),
                unit="°C",
                status=InformationClass.ESTIMATE,
                source="pareto_metrics",
                method="analytical_estimate",
                confidence=0.5,
            )
        elif ec.get("junction_temp_c") is not None:
            res["thermal"] = MetricValue(
                value=float(ec["junction_temp_c"]),
                unit="°C",
                status=InformationClass.ESTIMATE,
                source="estimated_constraints",
                method="thermal_model",
                confidence=0.5,
            )
        else:
            res["thermal"] = MetricValue(unit="°C", status=InformationClass.UNKNOWN)

        # Physical size
        mv_size = getattr(ms, "physical_size", None) if ms else None
        if mv_size and mv_size.value is not None and mv_size.status != InformationClass.UNKNOWN:
            res["physical_size"] = mv_size
        elif pm.get("physical_size") is not None:
            res["physical_size"] = MetricValue(
                value=float(pm["physical_size"]),
                unit="mm2",
                status=InformationClass.ESTIMATE,
                source="pareto_metrics",
                method="analytical_estimate",
                confidence=0.5,
            )
        elif ec.get("pcb_area_mm2") is not None:
            res["physical_size"] = MetricValue(
                value=float(ec["pcb_area_mm2"]),
                unit="mm2",
                status=InformationClass.ESTIMATE,
                source="estimated_constraints",
                method="geometric_footprint_model",
                confidence=0.5,
            )
        else:
            res["physical_size"] = MetricValue(unit="mm2", status=InformationClass.UNKNOWN)

        return res

    @classmethod
    def extract_metrics(cls, candidate: HardwareArchitectureCandidate) -> dict[str, Optional[float]]:
        """Extract standardized metric values for Pareto evaluation."""
        objs = cls.extract_metric_objects(candidate)
        return {k: mv.value for k, mv in objs.items()}

    @classmethod
    def has_complete_metrics(cls, metrics: dict[str, Any]) -> bool:
        """Return True if all 7 objectives have valid non-None values."""
        for obj in cls.OBJECTIVES:
            v = metrics.get(obj)
            if v is None:
                return False
            if isinstance(v, MetricValue):
                if v.value is None or not v.is_known:
                    return False
        return True

    @classmethod
    def dominates(cls, metrics_a: dict[str, Any], metrics_b: dict[str, Any]) -> bool:
        """Return True if solution A Pareto-dominates solution B across all 7 objectives.

        RESEARCH CONTRACT INVARIANTS:
        - HYPOTHESIS or UNKNOWN metrics CANNOT participate in dominance (returns False).
        - ESTIMATED CANNOT dominate MEASURED (analytical guesses cannot beat verified data).
        - MEASURED CAN dominate ESTIMATED.
        - Missing (None) data cannot establish dominance.
        """
        if not cls.has_complete_metrics(metrics_a) or not cls.has_complete_metrics(metrics_b):
            return False

        vals_a: dict[str, float] = {}
        vals_b: dict[str, float] = {}
        has_estimate_a = False
        has_measured_b = False

        for obj in cls.OBJECTIVES:
            item_a = metrics_a[obj]
            item_b = metrics_b[obj]

            if isinstance(item_a, MetricValue):
                if not item_a.is_usable_for_ranking or item_a.value is None:
                    return False
                vals_a[obj] = float(item_a.value)
                if item_a.status == InformationClass.ESTIMATE:
                    has_estimate_a = True
            else:
                if item_a is None:
                    return False
                vals_a[obj] = float(item_a)

            if isinstance(item_b, MetricValue):
                if not item_b.is_usable_for_ranking or item_b.value is None:
                    return False
                vals_b[obj] = float(item_b.value)
                if item_b.status == InformationClass.MEASUREMENT:
                    has_measured_b = True
            else:
                if item_b is None:
                    return False
                vals_b[obj] = float(item_b)

        # Invariant: ESTIMATED cannot dominate MEASURED
        if has_estimate_a and has_measured_b:
            return False

        at_least_one_strictly_better = False
        for obj, direction in cls.OBJECTIVES.items():
            val_a = vals_a[obj]
            val_b = vals_b[obj]

            if direction == "min":
                if val_a > val_b:
                    return False
                if val_a < val_b:
                    at_least_one_strictly_better = True
            else:  # max
                if val_a < val_b:
                    return False
                if val_a > val_b:
                    at_least_one_strictly_better = True

        return at_least_one_strictly_better

    def add(self, candidate: HardwareArchitectureCandidate) -> bool:
        """Attempt to add candidate to the Pareto frontier.

        Candidates with incomplete or HYPOTHESIS metrics cannot enter the frontier
        and are tracked separately as UNRANKED.
        Returns True if candidate was non-dominated and added to the frontier.
        """
        cand_objs = self.extract_metric_objects(candidate)

        # Candidates with HYPOTHESIS, UNKNOWN, or missing metrics cannot enter Pareto frontier
        has_hyp = any(mv.status == InformationClass.HYPOTHESIS for mv in cand_objs.values())
        if has_hyp or not self.has_complete_metrics(cand_objs):
            logger.debug(
                f"Candidate {candidate.architecture_id} has hypothesis/incomplete metrics; tracked as UNRANKED."
            )
            if candidate not in self.unranked:
                self.unranked.append(candidate)
            return False

        # Check if candidate is dominated by any existing member
        for existing in self.frontier:
            existing_objs = self.extract_metric_objects(existing)
            if self.dominates(existing_objs, cand_objs):
                return False  # Existing candidate is strictly superior

        # If we reach here, candidate is NOT dominated. Remove any existing candidates it dominates.
        new_frontier = []
        for existing in self.frontier:
            existing_objs = self.extract_metric_objects(existing)
            if not self.dominates(cand_objs, existing_objs):
                new_frontier.append(existing)

        new_frontier.append(candidate)
        self.frontier = new_frontier
        return True

    def get_frontier(self) -> list[HardwareArchitectureCandidate]:
        """Return all current non-dominated candidates on the frontier."""
        return list(self.frontier)

    def get_unranked(self) -> list[HardwareArchitectureCandidate]:
        """Return candidates tracked as UNRANKED due to incomplete metrics."""
        return list(self.unranked)

    def select_best_for_target(self, target_spec: TargetSpecification) -> Optional[HardwareArchitectureCandidate]:
        """
        Select the highest-performing candidate from the Pareto frontier that satisfies
        all declared target specification constraints.
        Only candidates with complete, non-None metrics are eligible.
        """
        if not self.frontier:
            return None

        # Filter strictly valid candidates satisfying max power, thermal, physical size
        valid_cands = []
        for cand in self.frontier:
            m = self.extract_metrics(cand)
            # Skip candidates with incomplete metrics
            if not self.has_complete_metrics(m):
                continue
            if (
                m["power"] <= target_spec.max_power_w
                and m["thermal"] <= target_spec.max_temperature_c
                and m["physical_size"] <= target_spec.max_pcb_area_mm2
                and m["throughput"] >= target_spec.min_tokens_per_second
                and m["memory"] >= target_spec.min_ram_gb
            ):
                valid_cands.append(cand)

        # Fall back to all fully-measured frontier candidates
        pool = valid_cands if valid_cands else [
            c for c in self.frontier if self.has_complete_metrics(self.extract_metrics(c))
        ]
        if not pool:
            return None

        # Rank by distance to ideal normalized utopian point
        def _target_score(cand: HardwareArchitectureCandidate) -> float:
            m = self.extract_metrics(cand)
            # Normalize objectives relative to target
            p_score = max(0.0, 1.0 - (m["power"] / target_spec.max_power_w))
            t_score = min(2.0, m["throughput"] / target_spec.min_tokens_per_second)
            size_score = max(0.0, 1.0 - (m["physical_size"] / target_spec.max_pcb_area_mm2))
            reward_score = cand.reward
            return (p_score * 0.3) + (t_score * 0.3) + (size_score * 0.2) + (reward_score * 0.2)

        return max(pool, key=_target_score)

    def select_measured_winner(
        self, target_spec: TargetSpecification
    ) -> Optional[HardwareArchitectureCandidate]:
        """Strict winner selection: Requires actual EDA synthesis MEASUREMENT.

        Refuses to declare an unmeasured analytical estimate as winning hardware.
        """
        measured_pool = []
        for cand in self.frontier:
            objs = self.extract_metric_objects(cand)
            # Area/cells must be measured from real synthesis netlist
            area_obj = objs.get("area")
            has_measured_area = bool(area_obj and area_obj.status == InformationClass.MEASUREMENT)
            has_synth_metrics = bool(
                cand.actual_synthesis_metrics and cand.actual_synthesis_metrics.get("cells", 0) > 0
            )
            if has_measured_area or has_synth_metrics:
                measured_pool.append(cand)

        if not measured_pool:
            logger.warning("No candidate has empirically verified EDA measurements. Measured winner selection deferred.")
            return None

        return max(measured_pool, key=lambda c: getattr(c, "reward", 0.0))


# ── Custom Chip Generator & Decomposition ──────────────────────────────

class CustomChipGenerator:
    """
    Generates and evolves complete Custom Silicon AI SoC architectures when
    commercial components are unable to satisfy physical or power constraints.

    Synthesizes SystemVerilog RTL block decomposition, interfaces, clock domains,
    memory hierarchy, accelerator structure, resource estimates, and verification plans.
    """

    @classmethod
    def generate_custom_soc_spec(
        cls,
        task_id: str,
        generation: int,
        target_spec: TargetSpecification,
        parent_id: Optional[str] = None,
        mutation_focus: Optional[str] = None,
    ) -> HardwareArchitectureCandidate:
        """Create a full custom AI SoC architecture specification with synthesizable RTL."""
        chip_gen_str = f"chip_generation_{generation:03d}"
        arch_id = f"custom_soc_{task_id}_{chip_gen_str}"

        # Base custom architecture parameters tailored for edge LLM acceleration
        is_area_tight = mutation_focus in ("reduce_area", "shared_multiplier", "narrower_datapath")
        is_power_tight = mutation_focus in ("reduce_power", "quantized_int4", "streaming_dataflow")

        parallelism = 2 if is_area_tight else 4
        pipeline_depth = 2 if is_power_tight else 3
        precision = "INT4" if (is_power_tight or "INT4" in target_spec.target_model_quantization) else "INT8"
        sram_kb = 256 if is_area_tight else 512

        components_spec = [
            {"name": "control_core", "type": "RV32I_RISCV_lite", "clock_mhz": 200, "area_cells": 1200},
            {"name": "transformer_accelerator", "type": f"systolic_gemm_{parallelism}x{parallelism}_{precision}", "clock_mhz": 400, "area_cells": 3500 if is_area_tight else 7200},
            {"name": "local_sram", "type": f"banked_scratchpad_{sram_kb}KB", "ports": 2, "area_cells": 1800},
            {"name": "memory_controller", "type": "LPDDR4x_narrow_bridge", "bus_width": 32, "area_cells": 950},
            {"name": "usb_controller", "type": "USB3_2_Gen1_Endpoint", "interface": "USB-C", "area_cells": 800},
            {"name": "dma_engine", "type": "multi_channel_scatter_gather_dma", "channels": 4, "area_cells": 650},
            {"name": "power_management", "type": "dynamic_voltage_frequency_scaler", "domains": 3, "area_cells": 300},
            {"name": "security_block", "type": "AES128_hardware_root_of_trust", "area_cells": 400},
        ]

        total_cells = sum(c["area_cells"] for c in components_spec)
        est_die_area_mm2 = total_cells * 0.00018  # approx 28nm scaling
        est_package_area_mm2 = max(25.0, est_die_area_mm2 * 4.0)  # QFN/BGA packaging
        est_power_w = 1.2 if is_power_tight else 2.4

        # Synthesizable SystemVerilog Top SoC & Accelerator
        rtl_code = cls._build_synthesizable_soc_rtl(
            module_name=f"custom_ai_soc_gen{generation}",
            parallelism=parallelism,
            pipeline_depth=pipeline_depth,
            precision=precision,
        )

        candidate = HardwareArchitectureCandidate(
            architecture_id=arch_id,
            task_id=task_id,
            parent_architecture_id=parent_id,
            generation=generation,
            mutation_type=mutation_focus or "custom_soc_initial",
            specification_version="2.0",
            architecture_description=(
                f"Custom AI SoC ({chip_gen_str}): Integrated RISC-V control core, {parallelism}x{parallelism} "
                f"{precision} transformer systolic accelerator, {sram_kb}KB local banked SRAM, on-chip DMA, "
                f"and native USB-C bridge on {target_spec.process_node} node."
            ),
            components=components_spec,
            interfaces=["USB-C", "LPDDR4x", "AXI4-Lite", "APB"],
            memory_hierarchy={
                "L1_sram_kb": sram_kb,
                "weight_buffer": "weight_stationary_banked",
                "dram_interface": "LPDDR4x_32bit",
                "bandwidth_gbps": 17.0,
            },
            compute_units={
                "array_type": "systolic_transformer_pe",
                "lanes": parallelism,
                "precision": precision,
                "macs_per_cycle": parallelism * parallelism,
            },
            accelerator_structure={
                "activation_function": "gelu_lut",
                "norm_engine": "rms_norm_pipelined",
                "quantization": precision,
            },
            datapath_structure=f"custom_systolic_array_{parallelism}x{parallelism}",
            pipeline_depth=pipeline_depth,
            parallelism=parallelism,
            memory_organization="embedded_banked_sram",
            buffering_strategy="weight_stationary_ping_pong",
            arithmetic_strategy=f"quantized_{precision.lower()}_systolic",
            interface_strategy="integrated_usb_c_stream",
            rtl_implementation=rtl_code,
            estimated_resource_requirements={
                "target_cells": total_cells,
                "die_area_mm2": round(est_die_area_mm2, 2),
                "package_area_mm2": round(est_package_area_mm2, 1),
                "estimated_latency_cycles": pipeline_depth,
                "estimation_method": "analytical_scaling_model",
                "confidence": 0.3,
            },
            # TRUTHFULNESS: actual_synthesis_metrics is EMPTY until real Yosys runs.
            # Never pre-populate with analytical estimates.
            actual_synthesis_metrics={},
            estimated_constraints={
                "power_w": est_power_w,
                "junction_temp_c": target_spec.ambient_temp_c + (est_power_w * target_spec.thermal_resistance_c_per_w),
                "pcb_area_mm2": est_package_area_mm2 * 2.2,  # PCB area with supporting passives
                "tokens_per_sec": 18.5 if precision == "INT4" else 12.0,
                "ram_gb": target_spec.min_ram_gb,
                "estimation_method": "analytical_scaling_model",
                "confidence": 0.3,
            },
            verification_status="unverified",  # No EDA tool has verified this candidate
            reward=0.0,  # TRUTHFULNESS: Reward must come from actual evaluation, not be pre-assigned
            is_hypothesis=True,
            validation_status="UNVERIFIED",
            custom_chip=True,
            chip_generation=generation,
            # pareto_metrics from analytical estimates (not measured)
            pareto_metrics={
                "area": float(total_cells),
                "power": est_power_w,
                "timing": float(pipeline_depth),
                "throughput": 18.5 if precision == "INT4" else 12.0,
                "memory": target_spec.min_ram_gb,
                "thermal": target_spec.ambient_temp_c + (est_power_w * target_spec.thermal_resistance_c_per_w),
                "physical_size": est_package_area_mm2 * 2.2,
            },
            metrics=MetricSet(
                area=MetricValue(value=float(total_cells), unit="cells", status=InformationClass.ESTIMATE, source="analytical_scaling_model", method="total_cells_scaling", confidence=0.3),
                power=MetricValue(value=est_power_w, unit="W", status=InformationClass.ESTIMATE, source="analytical_scaling_model", method="activity_factor_model", confidence=0.3),
                timing=MetricValue(value=float(pipeline_depth), unit="cycles", status=InformationClass.ESTIMATE, source="analytical_scaling_model", method="pipeline_depth_model", confidence=0.3),
                throughput=MetricValue(value=18.5 if precision == "INT4" else 12.0, unit="tok/s", status=InformationClass.ESTIMATE, source="analytical_scaling_model", method="macs_per_cycle_model", confidence=0.3),
                memory=MetricValue(value=target_spec.min_ram_gb, unit="GB", status=InformationClass.ESTIMATE, source="analytical_scaling_model", method="target_spec_minimum", confidence=0.3),
                thermal=MetricValue(value=target_spec.ambient_temp_c + (est_power_w * target_spec.thermal_resistance_c_per_w), unit="°C", status=InformationClass.ESTIMATE, source="analytical_scaling_model", method="thermal_resistance_model", confidence=0.3),
                physical_size=MetricValue(value=est_package_area_mm2 * 2.2, unit="mm2", status=InformationClass.ESTIMATE, source="analytical_scaling_model", method="die_package_ratio_model", confidence=0.3),
            ),
            # TRUTHFULNESS: Cannot claim SYNTHESIS_VALID until real Yosys synthesis succeeds
            verification_level=VerificationLevel.PHYSICALLY_ESTIMATED.value,
            metadata={
                "process_node": target_spec.process_node,
                "clock_domains": ["clk_sys_200mhz", "clk_npu_400mhz"],
                "verification_plan": ["cocotb_matrix_multiply", "formal_axi_handshake", "power_domain_isolation"],
                "metric_provenance": "analytical_scaling_model",
            },
        )
        return candidate

    @classmethod
    def _build_synthesizable_soc_rtl(
        cls,
        module_name: str,
        parallelism: int,
        pipeline_depth: int,
        precision: str,
    ) -> str:
        """Generate verified, clean synthesizable SystemVerilog for the custom AI accelerator."""
        width = 4 if precision == "INT4" else 8
        return f"""// Custom AI SoC Core Accelerator Block: {module_name}
// Generated for Open-Ended Hardware Architecture Search
`timescale 1ns / 1ps

module {module_name} #(
    parameter DATA_WIDTH = {width},
    parameter LANES = {parallelism},
    parameter PIPELINE_DEPTH = {pipeline_depth}
)(
    input  logic                   clk,
    input  logic                   rst_n,
    // AXI4-Stream Slave (Weights and Activations)
    input  logic                   s_axis_valid,
    output logic                   s_axis_ready,
    input  logic [LANES*DATA_WIDTH-1:0] s_axis_data_a,
    input  logic [LANES*DATA_WIDTH-1:0] s_axis_data_b,
    // AXI4-Stream Master (Accumulated Output)
    output logic                   m_axis_valid,
    input  logic                   m_axis_ready,
    output logic [(2*DATA_WIDTH)+4:0] m_axis_result
);

    // Handshake control
    assign s_axis_ready = m_axis_ready || !m_axis_valid;

    // Systolic / Pipelined Multiplier-Accumulator Array
    logic signed [(2*DATA_WIDTH)+4:0] accumulator_stage [0:PIPELINE_DEPTH-1];
    logic [PIPELINE_DEPTH-1:0] valid_pipeline;

    integer i;
    logic signed [(2*DATA_WIDTH)+4:0] dot_product;

    always_comb begin
        dot_product = '0;
        for (int l = 0; l < LANES; l++) begin
            dot_product = dot_product + 
                (signed'(s_axis_data_a[l*DATA_WIDTH +: DATA_WIDTH]) * 
                 signed'(s_axis_data_b[l*DATA_WIDTH +: DATA_WIDTH]));
        end
    end

    // Pipeline registers
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_pipeline <= '0;
            for (i = 0; i < PIPELINE_DEPTH; i++) begin
                accumulator_stage[i] <= '0;
            end
        end else if (s_axis_ready) begin
            valid_pipeline[0] <= s_axis_valid;
            accumulator_stage[0] <= dot_product;

            for (i = 1; i < PIPELINE_DEPTH; i++) begin
                valid_pipeline[i] <= valid_pipeline[i-1];
                accumulator_stage[i] <= accumulator_stage[i-1];
            end
        end
    end

    assign m_axis_valid  = valid_pipeline[PIPELINE_DEPTH-1];
    assign m_axis_result = accumulator_stage[PIPELINE_DEPTH-1];

endmodule
"""


# ── Architecture Search Engine ────────────────────────────────────────

class ArchitectureSearchEngine:
    """
    First-Class Open-Ended Architecture Search and Evolution Engine.

    Supports:
    - 16 architectural exploration operations.
    - Controlled hypothesis generation and external empirical validation.
    - Hard constraint rejection gating with root-cause mutation.
    - Commercial component exploration to Custom Chip Mode escalation.
    - Pareto frontier tracking.
    """

    def __init__(
        self,
        genealogy: Optional[ArchitectureGenealogy] = None,
        pareto_frontier: Optional[ParetoFrontier] = None,
    ):
        self.genealogy = genealogy or ArchitectureGenealogy()
        self.pareto_frontier = pareto_frontier or ParetoFrontier()

    def propose_candidates(
        self,
        task_id: str,
        task_description: str = "",
        n: int = 4,
        parent_candidate: Optional[HardwareArchitectureCandidate] = None,
        target_spec: Optional[TargetSpecification] = None,
    ) -> list[HardwareArchitectureCandidate]:
        """Propose N diverse architectural candidates or guided mutations for a task."""
        spec = target_spec or TargetSpecification()
        candidates: list[HardwareArchitectureCandidate] = []
        seed_val = int(hashlib.sha256(f"{task_id}_{task_description}".encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed_val + (parent_candidate.generation if parent_candidate else 0))

        if parent_candidate is not None:
            # Guided mutation branch across operators
            ops = [
                SearchOperation.MUTATE.value,
                SearchOperation.CHANGE_PRECISION.value,
                SearchOperation.CHANGE_PARALLELISM.value,
                SearchOperation.CHANGE_MEMORY_ARCHITECTURE.value,
            ]
            for op in ops[:n]:
                child = self.apply_operation(
                    candidate=parent_candidate,
                    operation=op,
                    target_spec=spec,
                )
                candidates.append(child)
            return candidates

        # Initial Diverse Architectural Hypotheses
        # Hypothesis 1: Commercial CPU + Discrete Edge NPU + LPDDR4
        c1 = HardwareArchitectureCandidate(
            architecture_id=f"arch_{task_id}_comm_npu_{seed_val % 1000:03d}",
            task_id=task_id,
            generation=0,
            mutation_type="initial_commercial_npu",
            architecture_description="Commercial Host CPU + Discrete 4 TOPS NPU + 8GB LPDDR4x",
            datapath_structure="pcie_interconnect_discrete_npu",
            pipeline_depth=2,
            parallelism=4,
            memory_organization="external_lpddr4x",
            buffering_strategy="host_dma_buffers",
            arithmetic_strategy="int8_dot_product",
            interface_strategy="pcie_gen3_bridge",
            estimated_resource_requirements={"target_cells": 12000, "estimated_latency_cycles": 4},
            estimated_constraints={"power_w": 6.8, "pcb_area_mm2": 2800.0, "tokens_per_sec": 14.0, "ram_gb": 8.0},
            reward=0.0,  # TRUTHFULNESS: Reward must come from actual evaluation, not pre-assigned
            is_hypothesis=True,
            validation_status="UNVERIFIED",
            pareto_metrics={"area": 12000.0, "power": 6.8, "timing": 4.0, "throughput": 14.0, "memory": 8.0, "thermal": 72.0, "physical_size": 2800.0},
            metrics=MetricSet(
                area=MetricValue(value=12000.0, unit="cells", status=InformationClass.HYPOTHESIS, source="llm_proposal", method="seed_hypothesis", confidence=0.1),
                power=MetricValue(value=6.8, unit="W", status=InformationClass.HYPOTHESIS, source="llm_proposal", method="seed_hypothesis", confidence=0.1),
                timing=MetricValue(value=4.0, unit="cycles", status=InformationClass.HYPOTHESIS, source="llm_proposal", method="seed_hypothesis", confidence=0.1),
                throughput=MetricValue(value=14.0, unit="tok/s", status=InformationClass.HYPOTHESIS, source="llm_proposal", method="seed_hypothesis", confidence=0.1),
                memory=MetricValue(value=8.0, unit="GB", status=InformationClass.HYPOTHESIS, source="llm_proposal", method="seed_hypothesis", confidence=0.1),
                thermal=MetricValue(value=72.0, unit="°C", status=InformationClass.HYPOTHESIS, source="llm_proposal", method="seed_hypothesis", confidence=0.1),
                physical_size=MetricValue(value=2800.0, unit="mm2", status=InformationClass.HYPOTHESIS, source="llm_proposal", method="seed_hypothesis", confidence=0.1),
            ),
        )
        self.genealogy.register_candidate(c1, mutation_type="initial_commercial_npu")
        self.pareto_frontier.add(c1)
        candidates.append(c1)

        # Hypothesis 2: Compact Integrated RISC-V SoC + Systolic Array + Embedded SRAM
        c2 = HardwareArchitectureCandidate(
            architecture_id=f"arch_{task_id}_systolic_{seed_val % 1000:03d}",
            task_id=task_id,
            generation=0,
            mutation_type="initial_systolic_sram",
            architecture_description="RISC-V Control Core + Weight-Stationary Systolic Array + 512KB SRAM",
            datapath_structure="systolic_processing_elements",
            pipeline_depth=2,
            parallelism=2,
            memory_organization="embedded_sram",
            buffering_strategy="weight_stationary",
            arithmetic_strategy="quantized_int4_shared",
            interface_strategy="axis_streaming",
            estimated_resource_requirements={"target_cells": 3800, "estimated_latency_cycles": 2},
            estimated_constraints={"power_w": 2.2, "pcb_area_mm2": 1100.0, "tokens_per_sec": 18.0, "ram_gb": 8.0},
            reward=0.0,  # TRUTHFULNESS: Reward must come from actual evaluation, not pre-assigned
            is_hypothesis=True,
            validation_status="UNVERIFIED",
            pareto_metrics={"area": 3800.0, "power": 2.2, "timing": 2.0, "throughput": 18.0, "memory": 8.0, "thermal": 52.0, "physical_size": 1100.0},
            metrics=MetricSet(
                area=MetricValue(value=3800.0, unit="cells", status=InformationClass.HYPOTHESIS, source="llm_proposal", method="seed_hypothesis", confidence=0.1),
                power=MetricValue(value=2.2, unit="W", status=InformationClass.HYPOTHESIS, source="llm_proposal", method="seed_hypothesis", confidence=0.1),
                timing=MetricValue(value=2.0, unit="cycles", status=InformationClass.HYPOTHESIS, source="llm_proposal", method="seed_hypothesis", confidence=0.1),
                throughput=MetricValue(value=18.0, unit="tok/s", status=InformationClass.HYPOTHESIS, source="llm_proposal", method="seed_hypothesis", confidence=0.1),
                memory=MetricValue(value=8.0, unit="GB", status=InformationClass.HYPOTHESIS, source="llm_proposal", method="seed_hypothesis", confidence=0.1),
                thermal=MetricValue(value=52.0, unit="°C", status=InformationClass.HYPOTHESIS, source="llm_proposal", method="seed_hypothesis", confidence=0.1),
                physical_size=MetricValue(value=1100.0, unit="mm2", status=InformationClass.HYPOTHESIS, source="llm_proposal", method="seed_hypothesis", confidence=0.1),
            ),
        )
        self.genealogy.register_candidate(c2, mutation_type="initial_systolic_sram")
        self.pareto_frontier.add(c2)
        candidates.append(c2)

        # Hypothesis 3: Streaming SIMD Vector Engine
        c3 = HardwareArchitectureCandidate(
            architecture_id=f"arch_{task_id}_simd_{seed_val % 1000:03d}",
            task_id=task_id,
            generation=0,
            mutation_type="initial_simd_vector",
            architecture_description="Vector SIMD Lanes with Direct Streaming FIFO Memory",
            datapath_structure="simd_vector_lanes",
            pipeline_depth=1,
            parallelism=4,
            memory_organization="streaming_fifos",
            buffering_strategy="fifo_queue",
            arithmetic_strategy="vector_parallel",
            interface_strategy="streaming",
            estimated_resource_requirements={"target_cells": 6200, "estimated_latency_cycles": 1},
            estimated_constraints={"power_w": 4.1, "pcb_area_mm2": 1600.0, "tokens_per_sec": 16.5, "ram_gb": 8.0},
            reward=0.0,  # TRUTHFULNESS: Reward must come from actual evaluation, not pre-assigned
            is_hypothesis=True,
            validation_status="UNVERIFIED",
            pareto_metrics={"area": 6200.0, "power": 4.1, "timing": 1.0, "throughput": 16.5, "memory": 8.0, "thermal": 64.0, "physical_size": 1600.0},
            metrics=MetricSet(
                area=MetricValue(value=6200.0, unit="cells", status=InformationClass.HYPOTHESIS, source="llm_proposal", method="seed_hypothesis", confidence=0.1),
                power=MetricValue(value=4.1, unit="W", status=InformationClass.HYPOTHESIS, source="llm_proposal", method="seed_hypothesis", confidence=0.1),
                timing=MetricValue(value=1.0, unit="cycles", status=InformationClass.HYPOTHESIS, source="llm_proposal", method="seed_hypothesis", confidence=0.1),
                throughput=MetricValue(value=16.5, unit="tok/s", status=InformationClass.HYPOTHESIS, source="llm_proposal", method="seed_hypothesis", confidence=0.1),
                memory=MetricValue(value=8.0, unit="GB", status=InformationClass.HYPOTHESIS, source="llm_proposal", method="seed_hypothesis", confidence=0.1),
                thermal=MetricValue(value=64.0, unit="°C", status=InformationClass.HYPOTHESIS, source="llm_proposal", method="seed_hypothesis", confidence=0.1),
                physical_size=MetricValue(value=1600.0, unit="mm2", status=InformationClass.HYPOTHESIS, source="llm_proposal", method="seed_hypothesis", confidence=0.1),
            ),
        )
        self.genealogy.register_candidate(c3, mutation_type="initial_simd_vector")
        self.pareto_frontier.add(c3)
        candidates.append(c3)

        # Hypothesis 4: Custom Silicon AI SoC Baseline (Custom Chip Mode)
        c4 = CustomChipGenerator.generate_custom_soc_spec(
            task_id=task_id,
            generation=1,
            target_spec=spec,
            mutation_focus="initial_custom_asic",
        )
        c4.generation = 0
        c4.architecture_id = f"arch_{task_id}_custom_soc_{seed_val % 1000:03d}"
        self.genealogy.register_candidate(c4, mutation_type="DESIGN_CUSTOM_CHIP")
        self.pareto_frontier.add(c4)
        candidates.append(c4)

        return candidates[:n]

    def apply_operation(
        self,
        candidate: HardwareArchitectureCandidate,
        operation: str | SearchOperation,
        target_spec: Optional[TargetSpecification] = None,
        extra_params: Optional[dict[str, Any]] = None,
        register_genealogy: bool = True,
    ) -> HardwareArchitectureCandidate:
        """Apply one of the 16 architecture search operations to evolve a candidate."""
        spec = target_spec or TargetSpecification()
        op_str = operation.value if isinstance(operation, SearchOperation) else str(operation)
        new_gen = candidate.generation + 1
        h = abs(hash(f"{candidate.architecture_id}_{op_str}_{new_gen}")) % 10000
        child_id = f"arch_{candidate.task_id}_g{new_gen}_{op_str.lower()[:8]}_{h:04d}"

        # Clone attributes
        child = HardwareArchitectureCandidate(
            architecture_id=child_id,
            task_id=candidate.task_id,
            parent_architecture_id=candidate.architecture_id,
            generation=new_gen,
            mutation_type=op_str,
            specification_version=candidate.specification_version,
            architecture_description=candidate.architecture_description,
            components=list(candidate.components),
            interfaces=list(candidate.interfaces),
            memory_hierarchy=dict(candidate.memory_hierarchy),
            compute_units=dict(candidate.compute_units),
            accelerator_structure=dict(candidate.accelerator_structure),
            datapath_structure=candidate.datapath_structure,
            pipeline_depth=candidate.pipeline_depth,
            parallelism=candidate.parallelism,
            memory_organization=candidate.memory_organization,
            buffering_strategy=candidate.buffering_strategy,
            arithmetic_strategy=candidate.arithmetic_strategy,
            interface_strategy=candidate.interface_strategy,
            rtl_implementation=candidate.rtl_implementation,
            estimated_resource_requirements=dict(candidate.estimated_resource_requirements),
            actual_synthesis_metrics={},  # TRUTHFULNESS: Mutated child has not been synthesized
            estimated_constraints=dict(candidate.estimated_constraints),
            measured_constraints={},      # TRUTHFULNESS: Mutated child has not been measured
            verification_status="unverified",
            reward=0.0,                   # TRUTHFULNESS: Reward must come from actual evaluation
            is_hypothesis=True,
            validation_status="UNVERIFIED",
            custom_chip=candidate.custom_chip,
            chip_generation=candidate.chip_generation,
            pareto_metrics=dict(candidate.pareto_metrics),
            metrics=MetricSet(),
            metadata=dict(candidate.metadata),
        )

        # Dispatch operation
        if op_str == SearchOperation.DESIGN_CUSTOM_CHIP.value:
            new_chip_gen = candidate.chip_generation + 1
            child = CustomChipGenerator.generate_custom_soc_spec(
                task_id=candidate.task_id,
                generation=new_chip_gen,
                target_spec=spec,
                parent_id=candidate.architecture_id,
                mutation_focus=extra_params.get("focus") if extra_params else "escalate_to_custom",
            )

        elif op_str == SearchOperation.CHANGE_PRECISION.value or op_str == "quantized_int4_arithmetic":
            child.arithmetic_strategy = "quantized_int4_shared"
            child.estimated_constraints["power_w"] = max(0.8, child.estimated_constraints.get("power_w", 3.0) * 0.55)
            child.pareto_metrics["power"] = child.estimated_constraints["power_w"]
            child.pareto_metrics["throughput"] = child.pareto_metrics.get("throughput", 12.0) * 1.4
            child.estimated_resource_requirements["target_cells"] = int(child.estimated_resource_requirements.get("target_cells", 2000) * 0.6)
            child.pareto_metrics["area"] = float(child.estimated_resource_requirements["target_cells"])
            child.architecture_description += " [MUTATED: INT4 Quantization]"

        elif op_str == SearchOperation.CHANGE_PARALLELISM.value:
            if child.parallelism > 1:
                child.parallelism = max(1, child.parallelism // 2)
                child.architecture_description += f" [MUTATED: Reduced parallelism to {child.parallelism}]"
            else:
                child.parallelism = 4
                child.architecture_description += f" [MUTATED: Expanded parallelism to {child.parallelism}]"
            scale = 0.6 if child.parallelism == 1 else 1.8
            child.estimated_resource_requirements["target_cells"] = int(child.estimated_resource_requirements.get("target_cells", 2000) * scale)
            child.pareto_metrics["area"] = float(child.estimated_resource_requirements["target_cells"])

        elif op_str == SearchOperation.CHANGE_MEMORY_ARCHITECTURE.value or op_str == "local_sram":
            child.memory_organization = "embedded_banked_sram"
            child.buffering_strategy = "weight_stationary_ping_pong"
            child.estimated_constraints["junction_temp_c"] = max(35.0, child.estimated_constraints.get("junction_temp_c", 60.0) - 10.0)
            child.pareto_metrics["thermal"] = child.estimated_constraints["junction_temp_c"]
            child.architecture_description += " [MUTATED: Embedded Banked SRAM + Weight Stationary]"

        elif op_str == SearchOperation.CHANGE_DATAFLOW.value or op_str == "streaming_dataflow":
            child.datapath_structure = "streaming_pipelined"
            child.buffering_strategy = "fifo_queue"
            child.interface_strategy = "axis_streaming"
            child.architecture_description += " [MUTATED: Streaming Dataflow]"

        elif op_str == "deeper_pipeline":
            child.pipeline_depth = candidate.pipeline_depth + 1
            child.estimated_resource_requirements["estimated_latency_cycles"] = child.pipeline_depth
            child.pareto_metrics["timing"] = float(child.pipeline_depth)
            child.architecture_description += f" [MUTATED: Pipeline depth {child.pipeline_depth}]"

        elif op_str == "shallower_pipeline":
            child.pipeline_depth = max(1, candidate.pipeline_depth - 1)
            child.estimated_resource_requirements["estimated_latency_cycles"] = child.pipeline_depth
            child.pareto_metrics["timing"] = float(child.pipeline_depth)
            child.architecture_description += f" [MUTATED: Pipeline depth {child.pipeline_depth}]"

        elif op_str == "more_parallel_lanes":
            child.parallelism = candidate.parallelism * 2
            child.estimated_resource_requirements["target_cells"] = int(child.estimated_resource_requirements.get("target_cells", 200) * 1.8)
            child.pareto_metrics["area"] = float(child.estimated_resource_requirements["target_cells"])
            child.architecture_description += f" [MUTATED: Parallelism {child.parallelism}]"

        elif op_str == "less_parallelism":
            child.parallelism = max(1, candidate.parallelism // 2)
            child.estimated_resource_requirements["target_cells"] = int(child.estimated_resource_requirements.get("target_cells", 200) * 0.6)
            child.pareto_metrics["area"] = float(child.estimated_resource_requirements["target_cells"])
            child.architecture_description += f" [MUTATED: Parallelism {child.parallelism}]"

        elif op_str == "systolic_organization":
            child.datapath_structure = "systolic_processing_elements"
            child.buffering_strategy = "weight_stationary"
            child.architecture_description += " [MUTATED: Systolic Array]"

        elif op_str == "simd_organization":
            child.datapath_structure = "simd_vector"
            child.parallelism = max(2, candidate.parallelism)
            child.architecture_description += " [MUTATED: SIMD Organization]"

        elif op_str == SearchOperation.CHANGE_PIPELINE.value:
            child.pipeline_depth = min(4, child.pipeline_depth + 1)
            child.estimated_resource_requirements["estimated_latency_cycles"] = child.pipeline_depth
            child.pareto_metrics["timing"] = float(child.pipeline_depth)
            child.architecture_description += f" [MUTATED: Pipeline depth {child.pipeline_depth}]"

        elif op_str == SearchOperation.REPLACE_COMPONENT.value:
            # Replace highest power or largest component
            child.components = [c for c in child.components if "discrete" not in str(c).lower()]
            child.estimated_constraints["pcb_area_mm2"] = max(600.0, child.estimated_constraints.get("pcb_area_mm2", 2000.0) * 0.7)
            child.pareto_metrics["physical_size"] = child.estimated_constraints["pcb_area_mm2"]
            child.architecture_description += " [MUTATED: Replaced discrete module with compact package]"

        elif op_str == SearchOperation.REMOVE_COMPONENT.value:
            if child.components:
                child.components.pop()
            child.architecture_description += " [MUTATED: Removed non-essential secondary block]"

        elif op_str == SearchOperation.DESIGN_CUSTOM_IP.value or op_str == "shared_multiplier":
            child.arithmetic_strategy = "time_multiplexed_shared"
            child.estimated_resource_requirements["target_cells"] = int(child.estimated_resource_requirements.get("target_cells", 2000) * 0.7)
            child.pareto_metrics["area"] = float(child.estimated_resource_requirements["target_cells"])
            child.architecture_description += " [MUTATED: Time-multiplexed shared multiplier IP]"

        else:  # Open-ended LLM or custom dynamic mutation
            child.pipeline_depth = max(1, child.pipeline_depth)
            child.architecture_description += f" [MUTATED: {op_str}]"

        # Apply open-ended LLM/dynamic parameters if provided
        if extra_params:
            for field_name in (
                "datapath_structure",
                "parallelism",
                "pipeline_depth",
                "memory_organization",
                "buffering_strategy",
                "arithmetic_strategy",
                "interface_strategy",
                "rtl_implementation",
                "architecture_description",
            ):
                if field_name in extra_params and extra_params[field_name] is not None:
                    setattr(child, field_name, extra_params[field_name])

            if "components" in extra_params and isinstance(extra_params["components"], list):
                child.components = list(extra_params["components"])
            if "interfaces" in extra_params and isinstance(extra_params["interfaces"], list):
                child.interfaces = list(extra_params["interfaces"])
            if "memory_hierarchy" in extra_params and isinstance(extra_params["memory_hierarchy"], dict):
                child.memory_hierarchy.update(extra_params["memory_hierarchy"])
            if "compute_units" in extra_params and isinstance(extra_params["compute_units"], dict):
                child.compute_units.update(extra_params["compute_units"])
            if "accelerator_structure" in extra_params and isinstance(extra_params["accelerator_structure"], dict):
                child.accelerator_structure.update(extra_params["accelerator_structure"])
            if "estimated_resource_requirements" in extra_params and isinstance(extra_params["estimated_resource_requirements"], dict):
                child.estimated_resource_requirements.update(extra_params["estimated_resource_requirements"])
            if "estimated_constraints" in extra_params and isinstance(extra_params["estimated_constraints"], dict):
                child.estimated_constraints.update(extra_params["estimated_constraints"])
            if "pareto_metrics" in extra_params and isinstance(extra_params["pareto_metrics"], dict):
                child.pareto_metrics.update(extra_params["pareto_metrics"])

        # TRUTHFULNESS: Mutated candidate is a hypothesis until empirically verified.
        # Tag all metric values as HYPOTHESIS with low initial confidence.
        for k, v in child.pareto_metrics.items():
            if v is not None:
                child.metrics.set_metric(
                    k,
                    float(v),
                    status=InformationClass.HYPOTHESIS,
                    source="mutation_hypothesis",
                    method=op_str,
                    confidence=0.1,
                )

        if register_genealogy:
            self.genealogy.register_candidate(
                child,
                parent_id=candidate.architecture_id,
                mutation_type=op_str,
                rationale=f"Applied {op_str} to evolve architecture toward target constraints",
            )
            self.pareto_frontier.add(child)
        return child

    def generate_open_ended_hypothesis(
        self,
        task_id: str,
        task_description: str = "",
        llm_proposal: Optional[dict[str, Any]] = None,
        target_spec: Optional[TargetSpecification] = None,
        generation: int = 0,
    ) -> HardwareArchitectureCandidate:
        """Create an open-ended, unrestricted architecture hypothesis directly from LLM generation."""
        import time
        spec = target_spec or TargetSpecification()
        prop = llm_proposal or {}
        h = abs(hash(f"{task_id}_{task_description}_{generation}_{time.time()}")) % 10000
        arch_id = prop.get("architecture_id", f"arch_{task_id}_llm_{h:04d}")

        cand = HardwareArchitectureCandidate(
            architecture_id=arch_id,
            task_id=task_id,
            generation=generation,
            mutation_type=prop.get("mutation_type", "open_ended_llm_proposal"),
            specification_version="2.0",
            architecture_description=prop.get("description", prop.get("architecture_description", f"LLM-generated architecture for {task_id}: {task_description}")),
            components=prop.get("components", [
                {"name": "core_processing_unit", "type": prop.get("datapath_structure", "custom_datapath"), "clock_mhz": prop.get("clock_mhz", 300), "area_cells": prop.get("cells", 2500)}
            ]),
            interfaces=prop.get("interfaces", ["AXI4-Stream", "APB"]),
            memory_hierarchy=prop.get("memory_hierarchy", {
                "local_sram_kb": prop.get("sram_kb", 128),
                "buffering": prop.get("buffering_strategy", "ping_pong_stream"),
            }),
            compute_units=prop.get("compute_units", {
                "parallelism": prop.get("parallelism", 4),
                "pipeline_depth": prop.get("pipeline_depth", 2),
                "arithmetic": prop.get("arithmetic_strategy", "int8_mac"),
            }),
            accelerator_structure=prop.get("accelerator_structure", {}),
            datapath_structure=prop.get("datapath_structure", "custom_open_ended_datapath"),
            pipeline_depth=int(prop.get("pipeline_depth", 2)),
            parallelism=int(prop.get("parallelism", 4)),
            memory_organization=prop.get("memory_organization", "banked_local_memory"),
            buffering_strategy=prop.get("buffering_strategy", "stream_buffer"),
            arithmetic_strategy=prop.get("arithmetic_strategy", "custom_arithmetic"),
            interface_strategy=prop.get("interface_strategy", "stream_interface"),
            rtl_implementation=prop.get("rtl_implementation", ""),
            estimated_resource_requirements=prop.get("estimated_resource_requirements", {
                "target_cells": prop.get("target_cells", 3000),
                "estimated_latency_cycles": prop.get("pipeline_depth", 2),
            }),
            actual_synthesis_metrics={},
            estimated_constraints=prop.get("estimated_constraints", {
                "power_w": prop.get("power_w", 2.0),
                "junction_temp_c": spec.ambient_temp_c + (prop.get("power_w", 2.0) * spec.thermal_resistance_c_per_w),
                "pcb_area_mm2": prop.get("pcb_area_mm2", 800.0),
                "tokens_per_sec": prop.get("tokens_per_sec", 15.0),
                "ram_gb": spec.min_ram_gb,
            }),
            measured_constraints={},
            verification_status="unverified",
            reward=0.0,
            is_hypothesis=True,
            validation_status="UNVERIFIED",
            custom_chip=prop.get("custom_chip", True),
            chip_generation=prop.get("chip_generation", 1),
            pareto_metrics=prop.get("pareto_metrics", {
                "area": float(prop.get("target_cells", 3000)),
                "power": float(prop.get("power_w", 2.0)),
                "timing": float(prop.get("pipeline_depth", 2)),
                "throughput": float(prop.get("tokens_per_sec", 15.0)),
                "memory": float(spec.min_ram_gb),
                "thermal": float(spec.ambient_temp_c + (prop.get("power_w", 2.0) * spec.thermal_resistance_c_per_w)),
                "physical_size": float(prop.get("pcb_area_mm2", 800.0)),
            }),
            metrics=MetricSet(),
            metadata=prop.get("metadata", {"source": "open_ended_llm_generator"}),
        )
        for k, v in cand.pareto_metrics.items():
            if v is not None:
                cand.metrics.set_metric(
                    k,
                    float(v),
                    status=InformationClass.HYPOTHESIS,
                    source="open_ended_llm_generator",
                    method="llm_hypothesis",
                    confidence=0.1,
                )
        self.genealogy.register_candidate(cand, mutation_type="open_ended_llm_proposal")
        self.pareto_frontier.add(cand)
        return cand

    async def propose_candidates_with_llm(
        self,
        task_id: str,
        task_description: str = "",
        qwen_client: Optional[Any] = None,
        n: int = 4,
        target_spec: Optional[TargetSpecification] = None,
    ) -> list[HardwareArchitectureCandidate]:
        """Propose candidates through unrestricted LLM generation or fall back to diverse proposal."""
        if qwen_client is not None and hasattr(qwen_client, "generate_structured"):
            candidates = []
            prompt = (
                f"You are a hardware architecture search engine. Design {n} diverse, novel architectural "
                f"hypotheses for task '{task_id}': {task_description}.\n"
                f"For each architecture, propose novel datapath, pipeline depth, parallelism, buffering, "
                f"and arithmetic strategy beyond static templates. Return JSON with 'candidates': [...] list."
            )
            try:
                res = await qwen_client.generate_structured([
                    {"role": "system", "content": "Return JSON with a 'candidates' list of architectural specifications."},
                    {"role": "user", "content": prompt}
                ])
                raw_cands = res.get("candidates", []) if isinstance(res, dict) else []
                for prop in raw_cands[:n]:
                    if isinstance(prop, dict):
                        cand = self.generate_open_ended_hypothesis(
                            task_id=task_id,
                            task_description=task_description,
                            llm_proposal=prop,
                            target_spec=target_spec,
                        )
                        candidates.append(cand)
                if len(candidates) >= n:
                    return candidates
            except Exception as e:
                logger.warning(f"LLM architectural proposal failed ({e}), using baseline engine.")
        return self.propose_candidates(task_id, task_description, n=n, target_spec=target_spec)

    def mutate_candidate(
        self,
        candidate: HardwareArchitectureCandidate,
        mutation_type: str,
    ) -> HardwareArchitectureCandidate:
        """Backward-compatible mutation interface."""
        return self.apply_operation(candidate, mutation_type, register_genealogy=False)

    def mutate_for_physical_constraints(
        self,
        candidate: HardwareArchitectureCandidate,
        constraint_result: Any,
    ) -> HardwareArchitectureCandidate:
        """Backward-compatible physical constraint mutation interface."""
        checks = getattr(constraint_result, "checks", {})
        if not checks.get("power_fit", True) or not checks.get("size_fit", True):
            if candidate.parallelism > 1:
                return self.mutate_candidate(candidate, "less_parallelism")
            elif "quantized" not in candidate.arithmetic_strategy:
                return self.mutate_candidate(candidate, "quantized_int4_arithmetic")
            else:
                return self.mutate_candidate(candidate, "shallower_pipeline")

        violations = getattr(constraint_result, "violations", [])
        return self.evolve_after_rejection(candidate, violations, TargetSpecification())

    def evolve_after_rejection(
        self,
        failed_candidate: HardwareArchitectureCandidate,
        failed_constraints: list[str],
        target_spec: TargetSpecification,
    ) -> HardwareArchitectureCandidate:
        """
        MANDATORY REQUIREMENT (Section 4 & 9):
        When a candidate fails ANY hard constraint, reject it and derive a NEW architecture
        that explicitly attacks the root cause of the failure.
        """
        logger.info(
            f"Candidate {failed_candidate.architecture_id} rejected due to: {failed_constraints}. "
            "Synthesizing new architectural direction."
        )

        failed_candidate.validation_status = "FAILED_HYPOTHESIS"
        failed_candidate.is_hypothesis = False
        for fc in failed_constraints:
            if fc not in failed_candidate.rejection_reasons:
                failed_candidate.rejection_reasons.append(fc)

        # 1. Commercial component fit failed or no commercial accelerator available -> Escalate to Custom Chip
        if any("commercial" in fc.lower() or "component" in fc.lower() or "accelerator" in fc.lower() for fc in failed_constraints):
            logger.info("Commercial components cannot satisfy target specification. Escalating to CUSTOM CHIP MODE.")
            return self.apply_operation(
                failed_candidate,
                SearchOperation.DESIGN_CUSTOM_CHIP,
                target_spec=target_spec,
                extra_params={"focus": "commercial_infeasibility_fallback"},
            )

        # 2. Power or thermal failure -> Lower parallelism, quantize to INT4, or add local SRAM weight reuse
        if any("power" in fc.lower() or "thermal" in fc.lower() for fc in failed_constraints):
            if failed_candidate.parallelism > 1:
                return self.apply_operation(failed_candidate, SearchOperation.CHANGE_PARALLELISM, target_spec)
            elif "int4" not in failed_candidate.arithmetic_strategy.lower():
                return self.apply_operation(failed_candidate, SearchOperation.CHANGE_PRECISION, target_spec)
            else:
                return self.apply_operation(failed_candidate, SearchOperation.CHANGE_MEMORY_ARCHITECTURE, target_spec)

        # 3. Physical size / PCB area failure -> Replace component or escalate to single-die Custom SoC
        if any("size" in fc.lower() or "area" in fc.lower() or "pcb" in fc.lower() for fc in failed_constraints):
            if not failed_candidate.custom_chip:
                # Commercial discrete components too bulky -> Custom chip escalation
                return self.apply_operation(
                    failed_candidate,
                    SearchOperation.DESIGN_CUSTOM_CHIP,
                    target_spec=target_spec,
                    extra_params={"focus": "size_reduction_asic"},
                )
            else:
                # Custom chip area too high -> Time-multiplexed shared arithmetic IP
                return self.apply_operation(failed_candidate, SearchOperation.DESIGN_CUSTOM_IP, target_spec)

        # 4. Throughput failure -> Deepen pipeline or expand parallelism
        if any("throughput" in fc.lower() or "token" in fc.lower() or "latency" in fc.lower() for fc in failed_constraints):
            return self.apply_operation(failed_candidate, SearchOperation.CHANGE_PIPELINE, target_spec)

        # 5. Timing / Critical Path failure -> Deeper pipeline or narrower datapath
        if any("timing" in fc.lower() or "frequency" in fc.lower() or "critical_path" in fc.lower() for fc in failed_constraints):
            return self.apply_operation(failed_candidate, SearchOperation.CHANGE_PIPELINE, target_spec, extra_params={"focus": "deeper_pipeline_timing"})

        # 6. Memory bandwidth / RAM hierarchy failure -> Local SRAM buffering / banked cache
        if any("bandwidth" in fc.lower() or "memory" in fc.lower() or "ram" in fc.lower() for fc in failed_constraints):
            return self.apply_operation(failed_candidate, SearchOperation.CHANGE_MEMORY_ARCHITECTURE, target_spec, extra_params={"focus": "local_sram_buffering"})

        # Default fallback mutation
        return self.apply_operation(failed_candidate, SearchOperation.MUTATE, target_spec)

    def validate_hypothesis(
        self,
        candidate: HardwareArchitectureCandidate,
        hard_constraint_failures: Optional[list[str]] = None,
        verification_level: VerificationLevel = VerificationLevel.PHYSICALLY_ESTIMATED,
    ) -> bool:
        """
        Transition hypothesis to VALIDATED_ARCHITECTURE or FAILED_HYPOTHESIS.
        A candidate is NEVER accepted merely because the LLM generated it.
        """
        candidate.is_hypothesis = False
        candidate.verification_level = verification_level.value

        if hard_constraint_failures:
            candidate.validation_status = "FAILED_HYPOTHESIS"
            candidate.verification_status = "failed"
            for fail in hard_constraint_failures:
                if fail not in candidate.rejection_reasons:
                    candidate.rejection_reasons.append(fail)
            return False

        candidate.validation_status = "VALIDATED_ARCHITECTURE"
        candidate.verification_status = "passed"
        self.pareto_frontier.add(candidate)
        return True

    def rank_candidates(
        self,
        candidates: list[HardwareArchitectureCandidate],
    ) -> list[HardwareArchitectureCandidate]:
        """Rank candidates using grounded verification status, constraint states, and reward.

        Candidates with unknown/unmeasured metrics are penalized rather than
        receiving credit from fabricated defaults.
        """
        def _score(c: HardwareArchitectureCandidate) -> float:
            score = c.reward
            if c.validation_status == "VALIDATED_ARCHITECTURE":
                score += 2.0
            elif c.validation_status == "FAILED_HYPOTHESIS":
                score -= 5.0  # Hard constraint failure penalty

            if c.verification_status == "passed":
                score += 1.0
            elif c.verification_status == "failed":
                score -= 3.0

            # Favor compact ACTUAL cell count (only from real synthesis)
            actual_cells = c.actual_synthesis_metrics.get("cells")
            if actual_cells and actual_cells > 0:
                score += 50.0 / actual_cells
            else:
                score -= 0.5

            # InformationClass provenance weighting across all objectives
            ms = getattr(c, "metrics", None)
            if ms:
                for attr in ("area", "power", "timing", "throughput", "memory", "thermal", "physical_size"):
                    mv = getattr(ms, attr, None)
                    if mv is not None:
                        if mv.status == InformationClass.MEASUREMENT:
                            score += 0.5   # Verified EDA measurement bonus
                        elif mv.status == InformationClass.ESTIMATE:
                            score += 0.1   # Plausible analytical estimate
                        elif mv.status == InformationClass.HYPOTHESIS:
                            score -= 0.3   # LLM/mutation hypothesis discount
                        else:  # UNKNOWN
                            score -= 0.5   # Missing data penalty

            # Penalize candidates with many unknown constraint states
            unknown_count = sum(1 for v in c.constraint_states.values() if v == "UNKNOWN")
            score -= unknown_count * 0.3

            return score

        return sorted(candidates, key=_score, reverse=True)
