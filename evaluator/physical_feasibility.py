"""Physical Feasibility Engine for Commercial Component Integration & PCB Packaging.

Evaluates whether real-world commercial hardware components (DRAM, storage,
accelerator, PMIC, USB-C controller) physically and electrically fit within
a declared TargetSpecification enclosure and power/thermal budget.

Labels outcomes:
- FEASIBLE_ESTIMATE: Plausible physical/electrical fit.
- INFEASIBLE_ESTIMATE: Proved violation of size, power, or thermal wall.
- UNKNOWN: Missing critical physical metrics (UNKNOWN != PASS).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from agent.schemas import (
    ComponentEvidence,
    ConstraintState,
    FeasibilityLabel,
    TargetSpecification,
)

logger = logging.getLogger(__name__)


@dataclass
class PhysicalFeasibilityReport:
    """Comprehensive physical feasibility assessment of a component assembly."""
    feasibility: FeasibilityLabel
    target_name: str
    component_count: int
    total_component_footprint_mm2: float
    estimated_pcb_area_mm2: float
    max_allowable_pcb_area_mm2: float
    total_estimated_power_w: float
    max_allowable_power_w: float
    estimated_junction_temp_c: float
    max_allowable_temp_c: float
    constraint_states: dict[str, ConstraintState] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    failure_categories: list[str] = field(default_factory=list)
    recommendation: str = ""
    escalate_to_custom_chip: bool = False
    # TRUTHFULNESS: Always declare the estimation method so downstream consumers
    # know this is an analytical model, not a physical measurement.
    estimation_method: str = "analytical_envelope_model_v1"


class PhysicalFeasibilityEngine:
    """Evaluates physical fit, power budgets, and thermal dissipation of real-world components."""

    def __init__(self, target_spec: Optional[TargetSpecification] = None):
        self.target_spec = target_spec or TargetSpecification()

    def evaluate_component_assembly(
        self,
        components: list[ComponentEvidence],
    ) -> PhysicalFeasibilityReport:
        """Evaluate a combination of real commercial components against target specification."""
        constraint_states: dict[str, ConstraintState] = {
            "functional": ConstraintState.NOT_APPLICABLE,
            "physical_size": ConstraintState.UNKNOWN,
            "pcb_area": ConstraintState.UNKNOWN,
            "power": ConstraintState.UNKNOWN,
            "thermal": ConstraintState.UNKNOWN,
            "ram_capacity": ConstraintState.UNKNOWN,
            "storage_capacity": ConstraintState.UNKNOWN,
            "interface_type": ConstraintState.UNKNOWN,
            "throughput": ConstraintState.UNKNOWN,
        }
        violations: list[str] = []
        failure_categories: list[str] = []

        if not components:
            return PhysicalFeasibilityReport(
                feasibility=FeasibilityLabel.UNKNOWN,
                target_name=self.target_spec.target_name,
                component_count=0,
                total_component_footprint_mm2=0.0,
                estimated_pcb_area_mm2=0.0,
                max_allowable_pcb_area_mm2=self.target_spec.max_pcb_area_mm2,
                total_estimated_power_w=0.0,
                max_allowable_power_w=self.target_spec.max_power_w,
                estimated_junction_temp_c=self.target_spec.ambient_temp_c,
                max_allowable_temp_c=self.target_spec.max_temperature_c,
                constraint_states=constraint_states,
                violations=["No components provided for physical feasibility evaluation."],
                failure_categories=["MISSING_COMPONENTS"],
                recommendation="Search for commercial components or escalate to custom chip design.",
                escalate_to_custom_chip=True,
            )

        # 1. Footprint & PCB Area Calculation
        total_footprint = 0.0
        missing_dimensions = False
        for c in components:
            area = c.footprint_area_mm2
            if area <= 0.0:
                missing_dimensions = True
            total_footprint += area

        # Fixed passive components & connector overhead
        # USB-C receptacle: ~76.5 mm²
        # Discrete passives / decoupling: ~60.0 mm²
        fixed_overhead = 136.5
        total_footprint += fixed_overhead

        # Double-sided PCB area requires 1.30x component footprint for trace escapes & vias
        estimated_pcb_area = total_footprint * 1.30
        max_pcb_area = self.target_spec.max_pcb_area_mm2

        if missing_dimensions:
            constraint_states["physical_size"] = ConstraintState.UNKNOWN
            constraint_states["pcb_area"] = ConstraintState.UNKNOWN
            violations.append("One or more components lack verifiable package dimensions.")
        elif estimated_pcb_area <= max_pcb_area:
            constraint_states["physical_size"] = ConstraintState.PASS
            constraint_states["pcb_area"] = ConstraintState.PASS
        else:
            constraint_states["physical_size"] = ConstraintState.FAIL
            constraint_states["pcb_area"] = ConstraintState.FAIL
            violations.append(
                f"Assembly estimated PCB area {estimated_pcb_area:.1f} mm² exceeds enclosure maximum {max_pcb_area:.1f} mm²."
            )
            failure_categories.append("PHYSICAL_SIZE")

        # 2. Power Consumption
        total_power = 0.0
        missing_power = False
        for c in components:
            if c.power_w is None:
                missing_power = True
            else:
                total_power += c.power_w

        # System discrete overhead & PMIC conversion loss (~12% loss)
        total_power += 0.20  # Standby passives & pullups
        total_power /= 0.88  # PMIC efficiency

        if missing_power:
            constraint_states["power"] = ConstraintState.UNKNOWN
            violations.append("One or more components lack verifiable active power ratings.")
        elif total_power <= self.target_spec.max_power_w:
            constraint_states["power"] = ConstraintState.PASS
        else:
            constraint_states["power"] = ConstraintState.FAIL
            violations.append(
                f"Total estimated assembly power {total_power:.2f} W exceeds max power budget {self.target_spec.max_power_w:.1f} W."
            )
            failure_categories.append("POWER")

        # 3. Thermal Dissipation
        theta_ja = self.target_spec.thermal_resistance_c_per_w
        ambient = self.target_spec.ambient_temp_c
        estimated_temp = ambient + (total_power * theta_ja)

        if missing_power:
            constraint_states["thermal"] = ConstraintState.UNKNOWN
        elif estimated_temp <= self.target_spec.max_temperature_c:
            constraint_states["thermal"] = ConstraintState.PASS
        else:
            constraint_states["thermal"] = ConstraintState.FAIL
            violations.append(
                f"Assembly junction temperature {estimated_temp:.1f} °C exceeds maximum thermal limit {self.target_spec.max_temperature_c:.1f} °C."
            )
            failure_categories.append("THERMAL")

        # 4. Memory & Storage Capacity Checks
        dram_caps = [c.memory_capacity_gb for c in components if c.category == "dram" and c.memory_capacity_gb is not None]
        total_ram = sum(dram_caps)
        if total_ram >= self.target_spec.min_ram_gb:
            constraint_states["ram_capacity"] = ConstraintState.PASS
        else:
            constraint_states["ram_capacity"] = ConstraintState.FAIL
            violations.append(
                f"Total RAM {total_ram:.1f} GB below required {self.target_spec.min_ram_gb:.1f} GB."
            )
            failure_categories.append("RAM_CAPACITY")

        storage_caps = [c.memory_capacity_gb for c in components if c.category == "storage" and c.memory_capacity_gb is not None]
        total_storage = sum(storage_caps)
        if total_storage >= self.target_spec.min_storage_gb:
            constraint_states["storage_capacity"] = ConstraintState.PASS
        else:
            constraint_states["storage_capacity"] = ConstraintState.FAIL
            violations.append(
                f"Total storage {total_storage:.1f} GB below required {self.target_spec.min_storage_gb:.1f} GB."
            )
            failure_categories.append("STORAGE_CAPACITY")

        # 5. Interface Check
        usb_controllers = [c for c in components if "usb" in c.interface.lower() or c.category == "usb_controller"]
        if usb_controllers:
            constraint_states["interface_type"] = ConstraintState.PASS
        else:
            constraint_states["interface_type"] = ConstraintState.FAIL
            violations.append("Missing dedicated USB-C / USB 3.2 interface controller.")
            failure_categories.append("INTERFACE")

        # 6. Compute Throughput Check
        npu_comps = [c for c in components if c.category == "accelerator" or c.compute_capability_tops is not None]
        if npu_comps:
            total_tops = sum(c.compute_capability_tops or 0.0 for c in npu_comps)
            if total_tops >= 1.0:
                constraint_states["throughput"] = ConstraintState.PASS
            elif any(c.compute_capability_tops is None for c in npu_comps):
                constraint_states["throughput"] = ConstraintState.UNKNOWN
            else:
                constraint_states["throughput"] = ConstraintState.FAIL
                violations.append(f"Compute {total_tops:.1f} TOPS below required limit.")
                failure_categories.append("COMPUTE")
        else:
            constraint_states["throughput"] = ConstraintState.NOT_APPLICABLE

        # 7. Overall Feasibility Labeling
        has_fail = any(v == ConstraintState.FAIL for v in constraint_states.values())
        has_unknown = any(v == ConstraintState.UNKNOWN for v in constraint_states.values())

        if has_fail:
            feasibility = FeasibilityLabel.INFEASIBLE_ESTIMATE
            escalate = True
            recommendation = (
                f"Commercial components failed constraints ({', '.join(failure_categories)}). "
                "Search for smaller/lower-power components or transition to CUSTOM CHIP MODE."
            )
        elif has_unknown:
            feasibility = FeasibilityLabel.UNKNOWN
            escalate = False
            recommendation = "Acquire verified manufacturer datasheets for unknown physical parameters."
        else:
            feasibility = FeasibilityLabel.FEASIBLE_ESTIMATE
            escalate = False
            recommendation = "Commercial component assembly satisfies physical and electrical envelope."

        return PhysicalFeasibilityReport(
            feasibility=feasibility,
            target_name=self.target_spec.target_name,
            component_count=len(components),
            total_component_footprint_mm2=round(total_footprint, 2),
            estimated_pcb_area_mm2=round(estimated_pcb_area, 2),
            max_allowable_pcb_area_mm2=round(max_pcb_area, 2),
            total_estimated_power_w=round(total_power, 2),
            max_allowable_power_w=round(self.target_spec.max_power_w, 2),
            estimated_junction_temp_c=round(estimated_temp, 1),
            max_allowable_temp_c=round(self.target_spec.max_temperature_c, 1),
            constraint_states=constraint_states,
            violations=violations,
            failure_categories=failure_categories,
            recommendation=recommendation,
            escalate_to_custom_chip=escalate,
        )

    def evaluate_candidate(self, candidate: Any) -> PhysicalFeasibilityReport:
        """Evaluate a HardwareArchitectureCandidate against physical and electrical envelope."""
        # 1. If real ComponentEvidence components are present, evaluate component assembly
        raw_components = getattr(candidate, "components", [])
        if raw_components and all(isinstance(c, ComponentEvidence) for c in raw_components):
            return self.evaluate_component_assembly(raw_components)

        # 2. Evaluate Custom Chip or Projected Architecture Candidate
        constraint_states: dict[str, ConstraintState] = {
            "power": ConstraintState.UNKNOWN,
            "pcb_area": ConstraintState.UNKNOWN,
            "thermal": ConstraintState.UNKNOWN,
            "ram_capacity": ConstraintState.UNKNOWN,
            "throughput": ConstraintState.UNKNOWN,
        }
        violations: list[str] = []
        failure_categories: list[str] = []

        pm = getattr(candidate, "pareto_metrics", {})
        ec = getattr(candidate, "estimated_constraints", {})

        # Power
        power_w = pm.get("power", ec.get("power_w"))
        if power_w is None:
            constraint_states["power"] = ConstraintState.UNKNOWN
        elif power_w <= self.target_spec.max_power_w:
            constraint_states["power"] = ConstraintState.PASS
        else:
            constraint_states["power"] = ConstraintState.FAIL
            violations.append(f"Power {power_w:.2f}W exceeds budget {self.target_spec.max_power_w:.2f}W.")
            failure_categories.append("POWER")

        # Physical PCB Area
        pcb_area = pm.get("physical_size", ec.get("pcb_area_mm2"))
        max_pcb = self.target_spec.max_pcb_area_mm2
        if pcb_area is None:
            constraint_states["pcb_area"] = ConstraintState.UNKNOWN
        elif pcb_area <= max_pcb:
            constraint_states["pcb_area"] = ConstraintState.PASS
        else:
            constraint_states["pcb_area"] = ConstraintState.FAIL
            violations.append(f"PCB footprint {pcb_area:.1f} mm2 exceeds maximum envelope {max_pcb:.1f} mm2.")
            failure_categories.append("PCB_AREA")

        # Thermal
        thermal_c = pm.get("thermal", ec.get("junction_temp_c"))
        if thermal_c is None and power_w is not None:
            thermal_c = self.target_spec.ambient_temp_c + (power_w * self.target_spec.thermal_resistance_c_per_w)

        if thermal_c is None:
            constraint_states["thermal"] = ConstraintState.UNKNOWN
        elif thermal_c <= self.target_spec.max_temperature_c:
            constraint_states["thermal"] = ConstraintState.PASS
        else:
            constraint_states["thermal"] = ConstraintState.FAIL
            violations.append(f"Junction temp {thermal_c:.1f}°C exceeds max limit {self.target_spec.max_temperature_c:.1f}°C.")
            failure_categories.append("THERMAL")

        # Throughput
        tok_s = pm.get("throughput", ec.get("tokens_per_sec"))
        if tok_s is None:
            constraint_states["throughput"] = ConstraintState.UNKNOWN
        elif tok_s >= self.target_spec.min_tokens_per_second:
            constraint_states["throughput"] = ConstraintState.PASS
        else:
            constraint_states["throughput"] = ConstraintState.FAIL
            violations.append(f"Throughput {tok_s:.1f} tok/s below minimum required {self.target_spec.min_tokens_per_second:.1f} tok/s.")
            failure_categories.append("THROUGHPUT")

        # RAM
        ram_gb = pm.get("memory", ec.get("ram_gb"))
        if ram_gb is None:
            constraint_states["ram_capacity"] = ConstraintState.UNKNOWN
        elif ram_gb >= self.target_spec.min_ram_gb:
            constraint_states["ram_capacity"] = ConstraintState.PASS
        else:
            constraint_states["ram_capacity"] = ConstraintState.FAIL
            violations.append(f"RAM capacity {ram_gb:.1f}GB below required {self.target_spec.min_ram_gb:.1f}GB.")
            failure_categories.append("RAM_CAPACITY")

        # Feasibility classification
        has_fail = any(v == ConstraintState.FAIL for v in constraint_states.values())
        has_unknown = any(v == ConstraintState.UNKNOWN for v in constraint_states.values())

        if has_fail:
            feasibility = FeasibilityLabel.INFEASIBLE_ESTIMATE
            rec = f"Candidate violated constraints: {', '.join(failure_categories)}. Architectural evolution required."
        elif has_unknown:
            feasibility = FeasibilityLabel.UNKNOWN
            rec = "Unknown physical constraint parameters."
        else:
            feasibility = FeasibilityLabel.FEASIBLE_ESTIMATE
            rec = "Candidate satisfies all declared physical, electrical, and performance envelope constraints."

        return PhysicalFeasibilityReport(
            feasibility=feasibility,
            target_name=self.target_spec.target_name,
            component_count=len(raw_components),
            total_component_footprint_mm2=round(float(pcb_area or 0.0) * 0.6, 2),
            estimated_pcb_area_mm2=round(float(pcb_area or 0.0), 2),
            max_allowable_pcb_area_mm2=round(max_pcb, 2),
            total_estimated_power_w=round(float(power_w or 0.0), 2),
            max_allowable_power_w=round(self.target_spec.max_power_w, 2),
            estimated_junction_temp_c=round(float(thermal_c or 0.0), 1),
            max_allowable_temp_c=round(self.target_spec.max_temperature_c, 1),
            constraint_states=constraint_states,
            violations=violations,
            failure_categories=failure_categories,
            recommendation=rec,
            escalate_to_custom_chip=not getattr(candidate, "custom_chip", False) and ("PCB_AREA" in failure_categories or "POWER" in failure_categories),
        )
