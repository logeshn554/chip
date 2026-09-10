"""Physical Constraint and System Integration Model.

Bridges synthesized digital RTL and microarchitecture candidates with physical
reality: die size, packaging, double-sided PCB area, DRAM/storage integration,
thermal dissipation (theta_ja), and end-to-end AI token throughput.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from agent.schemas import (
    ConstraintCheckResult,
    DevicePhysicalEnvelope,
    HardwareArchitectureCandidate,
    PhysicalProjectionMetrics,
)

logger = logging.getLogger(__name__)


class PhysicalEnvelopeModel:
    """Grounded physical projection engine for portable AI computer subsystems."""

    def __init__(self, envelope: Optional[DevicePhysicalEnvelope] = None):
        self.envelope = envelope or DevicePhysicalEnvelope()

    def project_candidate(
        self,
        candidate: HardwareArchitectureCandidate,
        process_node_nm: int = 28,
    ) -> PhysicalProjectionMetrics:
        """Compute grounded physical projections for an architecture candidate."""
        # 1. Gate count / cell count extraction
        metrics = candidate.actual_synthesis_metrics or candidate.estimated_resource_requirements
        cells = metrics.get("cells") or metrics.get("cell_count") or 1200
        if not isinstance(cells, (int, float)) or cells <= 0:
            cells = 1200

        # Technology scaling: approximate standard-cell logic density
        # 28nm: ~1.5 M gates/mm^2 | 14nm: ~3.5 M gates/mm^2 | 7nm: ~8.0 M gates/mm^2
        if process_node_nm <= 7:
            density_gates_per_mm2 = 8_000_000
        elif process_node_nm <= 14:
            density_gates_per_mm2 = 3_500_000
        else:
            density_gates_per_mm2 = 1_500_000

        # Parallelism scaling: candidate parallelism scales compute datapath area
        parallelism = max(1, candidate.parallelism)
        pipeline_depth = max(1, candidate.pipeline_depth)
        
        # Effective gate equivalent: cell count * parallelism factor * pipeline register overhead
        effective_gates = cells * parallelism * (1.0 + 0.08 * (pipeline_depth - 1))
        
        # Die area in mm^2 (including 25% pad-ring, clock tree, and power grid overhead)
        digital_die_area = (effective_gates / density_gates_per_mm2) * 1.25
        # Enforce minimum pad-limited die floorplan (e.g. 1.2 mm x 1.2 mm)
        digital_die_area = max(1.44, digital_die_area)

        # ASIC Package Area (Flip-Chip BGA with 0.5mm ball pitch)
        asic_package_area = max(49.0, digital_die_area * 2.5)  # e.g., 7mm x 7mm to 12mm x 12mm

        # Fixed subsystem packages
        # 8GB LPDDR4x/5 BGA package (12mm x 12mm)
        dram_package_area = 144.0
        # 256GB UFS / eMMC BGA153 package (11.5mm x 13mm)
        storage_package_area = 149.5
        # Multi-rail PMIC + inductors + bulk capacitors
        pmic_passives_area = 115.0
        # USB-C mid-mount receptacle footprint
        usbc_connector_area = 76.5

        # Component sum & double-sided routing area
        component_sum = (
            asic_package_area
            + dram_package_area
            + storage_package_area
            + pmic_passives_area
            + usbc_connector_area
        )
        # PCB layout requires ~1.30x component footprint for traces, testpads, and margins
        total_pcb_area = component_sum * 1.30

        # 2. Power Modeling
        # Clock frequency: deeper pipeline enables higher Fmax
        base_clock_mhz = 200.0 + min(600.0, (pipeline_depth - 1) * 75.0)
        
        # ASIC Dynamic power: C_eff * V^2 * f * alpha * parallelism
        # At 28nm (0.9V core): ~0.08 W per 100k gates @ 500MHz
        voltage = 0.9 if process_node_nm >= 28 else 0.75
        asic_dynamic_w = (effective_gates / 100_000.0) * 0.08 * (base_clock_mhz / 500.0) * (voltage / 0.9) ** 2
        # Arithmetic strategy modifier
        if "quantized" in candidate.arithmetic_strategy or "int4" in candidate.arithmetic_strategy:
            asic_dynamic_w *= 0.55  # 4-bit datapath reduces switching capacitance
        elif "shared" in candidate.arithmetic_strategy:
            asic_dynamic_w *= 0.75

        # ASIC Leakage: ~10% of dynamic at 28nm
        asic_leakage_w = 0.05 + 0.08 * (digital_die_area / 10.0)
        asic_power = asic_dynamic_w + asic_leakage_w

        # DRAM power: LPDDR4x @ 32-bit bus active read/write
        dram_power = 1.15  # W active during continuous inference
        if candidate.memory_organization == "streaming":
            dram_power *= 0.85  # Sequential bursts reduce command overhead

        # Storage & system overhead power
        storage_power = 0.25
        system_overhead_power = 0.15

        # PMIC 88% efficiency
        pmic_efficiency = 0.88
        total_device_power = (asic_power + dram_power + storage_power + system_overhead_power) / pmic_efficiency

        # 3. Thermal Dissipation Model
        # Tj = T_ambient + P_total * theta_ja
        theta_ja = self.envelope.thermal_resistance_c_per_w
        ambient_temp = self.envelope.ambient_temp_c
        estimated_junction_temp = ambient_temp + (total_device_power * theta_ja)

        # 4. Performance & Token Throughput Model
        # Parameterizable target model workload scaling:
        params_b = getattr(self.envelope, "target_model_params_b", 14.0)
        weight_bits = getattr(self.envelope, "weight_bits", 4)
        model_size_gb = (params_b * weight_bits) / 8.0
        
        # Effective memory bandwidth: 32-bit LPDDR4x-4266 -> ~34.1 GB/s peak, 75% efficiency -> ~25.6 GB/s
        effective_mem_bw_gbps = 25.6
        if candidate.memory_organization == "streaming":
            effective_mem_bw_gbps = 28.5  # Streaming prefetch improves bus utilization
        elif candidate.memory_organization == "local_sram":
            effective_mem_bw_gbps = 31.0  # On-chip KV caching reduces external DRAM re-reads

        # Memory-bound token generation rate (tokens/sec) = Bandwidth / Model Footprint
        memory_tokens_per_sec = effective_mem_bw_gbps / max(0.5, model_size_gb)

        # Compute throughput: Parallelism * MACs * Clock
        # An AI accelerator integrates dedicated processing arrays (e.g. 64-256 MACs per lane)
        if "systolic" in candidate.datapath_structure:
            mac_units = parallelism * 256  # 2D Systolic PE array
        elif "simd" in candidate.datapath_structure:
            mac_units = parallelism * 128  # SIMD vector processing units
        elif "quantized" in candidate.arithmetic_strategy or "int4" in candidate.arithmetic_strategy:
            mac_units = parallelism * 128  # Dense low-precision integer execution units
        else:
            mac_units = parallelism * 64

        compute_tops = (2.0 * mac_units * (base_clock_mhz * 1e6)) / 1e12
        # Dynamic operations per token: 2.0 * parameters (Giga-Ops)
        giga_ops_per_token = max(0.1, 2.0 * params_b)
        compute_tokens_per_sec = (compute_tops * 1000.0) / giga_ops_per_token

        # Bottleneck throughput
        achievable_tokens_per_sec = min(memory_tokens_per_sec, compute_tokens_per_sec)
        bottleneck = "memory_bandwidth" if memory_tokens_per_sec < compute_tokens_per_sec else "compute"

        # If junction temperature exceeds maximum allowed, thermal throttling degrades performance
        if estimated_junction_temp > self.envelope.max_junction_temp_c:
            throttle_factor = self.envelope.max_junction_temp_c / estimated_junction_temp
            achievable_tokens_per_sec *= throttle_factor
            bottleneck = "thermal_throttled"

        return PhysicalProjectionMetrics(
            digital_die_area_mm2=round(digital_die_area, 2),
            asic_package_area_mm2=round(asic_package_area, 2),
            dram_package_area_mm2=round(dram_package_area, 2),
            storage_package_area_mm2=round(storage_package_area, 2),
            pmic_passives_area_mm2=round(pmic_passives_area, 2),
            usbc_connector_area_mm2=round(usbc_connector_area, 2),
            total_component_area_mm2=round(component_sum, 2),
            total_pcb_area_mm2=round(total_pcb_area, 2),
            asic_power_w=round(asic_power, 2),
            dram_power_w=round(dram_power, 2),
            system_overhead_power_w=round(storage_power + system_overhead_power, 2),
            total_device_power_w=round(total_device_power, 2),
            estimated_junction_temp_c=round(estimated_junction_temp, 1),
            effective_memory_bandwidth_gbps=round(effective_mem_bw_gbps, 1),
            compute_tops=round(compute_tops, 2),
            achievable_tokens_per_sec=round(achievable_tokens_per_sec, 1),
            bottleneck=bottleneck,
        )


class PhysicalConstraintChecker:
    """Evaluates candidate architectures against physical, electrical, and thermal constraints."""

    def __init__(self, envelope: Optional[DevicePhysicalEnvelope] = None):
        self.envelope = envelope or DevicePhysicalEnvelope()
        self.model = PhysicalEnvelopeModel(self.envelope)

    def evaluate(
        self,
        candidate: HardwareArchitectureCandidate,
    ) -> ConstraintCheckResult:
        """Check all mandatory constraints for a candidate design."""
        projections = self.model.project_candidate(candidate)
        violations: list[str] = []
        recommendations: list[str] = []
        checks: dict[str, bool] = {}

        # 1. Physical size / PCB area fit
        max_pcb = self.envelope.max_pcb_area_mm2
        if projections.total_pcb_area_mm2 <= max_pcb:
            checks["size_fit"] = True
        else:
            checks["size_fit"] = False
            violations.append(
                f"Total PCB area {projections.total_pcb_area_mm2:.1f} mm² exceeds enclosure maximum {max_pcb:.1f} mm²."
            )
            recommendations.append(
                "Reduce datapath parallelism or switch from discrete registers to shared SRAM buffers to reduce package footprint."
            )

        # 2. Maximum Power
        if projections.total_device_power_w <= self.envelope.max_power_w:
            checks["power_fit"] = True
        else:
            checks["power_fit"] = False
            violations.append(
                f"Total device power {projections.total_device_power_w:.2f} W exceeds maximum budget {self.envelope.max_power_w:.1f} W."
            )
            recommendations.append(
                "Lower parallel compute lanes or apply quantized INT4 arithmetic sharing to reduce dynamic switching power."
            )

        # 3. Maximum Thermal / Junction Temperature
        if projections.estimated_junction_temp_c <= self.envelope.max_junction_temp_c:
            checks["thermal_fit"] = True
        else:
            checks["thermal_fit"] = False
            violations.append(
                f"Estimated junction temperature {projections.estimated_junction_temp_c:.1f} °C exceeds safe limit {self.envelope.max_junction_temp_c:.1f} °C."
            )
            recommendations.append(
                "Incorporate streaming dataflow and reduce peak burst clock frequency to limit passive thermal rise."
            )

        # 4. Minimum RAM Capacity
        # Architecture candidate must specify or support >= 8GB for LLM weights + KV cache
        checks["ram_fit"] = self.envelope.min_ram_gb >= 8.0
        if not checks["ram_fit"]:
            violations.append(f"RAM capacity below mandatory {self.envelope.min_ram_gb:.1f} GB.")
            recommendations.append("Configure dual LPDDR4x/5 channels for 8GB minimum capacity.")

        # 5. Minimum Storage Capacity
        checks["storage_fit"] = self.envelope.min_storage_gb >= 256.0
        if not checks["storage_fit"]:
            violations.append(f"Storage capacity below mandatory {self.envelope.min_storage_gb:.1f} GB.")
            recommendations.append("Integrate 256GB UFS BGA module for local model weight storage.")

        # 6. USB-C Interface Connectivity
        checks["interface_fit"] = "usb" in self.envelope.interface_type.lower()
        if not checks["interface_fit"]:
            violations.append(f"Interface type '{self.envelope.interface_type}' incompatible with USB-C standard.")
            recommendations.append("Bind interface strategy to USB 3.2 Gen 2 Type-C controller.")

        # 7. Token Throughput Target
        if projections.achievable_tokens_per_sec >= self.envelope.min_tokens_per_sec:
            checks["throughput_fit"] = True
        else:
            checks["throughput_fit"] = False
            violations.append(
                f"Achievable throughput {projections.achievable_tokens_per_sec:.1f} tok/s below required target {self.envelope.min_tokens_per_sec:.1f} tok/s."
            )
            if projections.bottleneck == "compute":
                recommendations.append(
                    "Deepen pipeline registers or increase parallel compute lanes to raise compute throughput."
                )
            elif projections.bottleneck == "memory_bandwidth":
                recommendations.append(
                    "Switch to local SRAM buffering or quantized INT4 weights to compress memory bandwidth requirement."
                )
            else:
                recommendations.append(
                    "Resolve thermal throttling to restore peak inference throughput."
                )

        all_passed = all(checks.values())
        return ConstraintCheckResult(
            passed=all_passed,
            checks=checks,
            violations=violations,
            recommendations=recommendations,
            projections=projections,
            envelope_name=self.envelope.target_name,
        )
