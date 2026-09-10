"""Demonstration script for Physical Envelope Constraint Satisfaction Evolution Loop.

Top-level research objective:
"Autonomously discover a hardware architecture that satisfies a configurable
physical envelope and AI-performance specification, using LLM-guided architecture
evolution and externally grounded EDA rewards."
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath("."))

from agent.schemas import DevicePhysicalEnvelope
from learning.self_evolution import PhysicalConstraintEvolutionLoop

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    print("=" * 80)
    print("  AUTONOMOUS PHYSICAL CONSTRAINT SATISFACTION & ARCHITECTURE EVOLUTION")
    print("  Target: Portable Independent AI Device Envelope")
    print("  Objective: Discover hardware architecture passing all physical/PPA constraints")
    print("=" * 80)

    # 1. Define Configurable Target Envelope
    envelope = DevicePhysicalEnvelope(
        target_name="Portable Independent AI Device (100x30x12mm)",
        enclosure_length_mm=100.0,
        enclosure_width_mm=30.0,
        enclosure_height_mm=12.0,
        max_power_w=5.0,              # USB-C bus powered limit
        max_junction_temp_c=85.0,     # Max junction temperature
        ambient_temp_c=30.0,          # Nominal ambient
        thermal_resistance_c_per_w=10.0,  # Passive aluminum cooling theta_ja
        min_ram_gb=8.0,               # 8GB LPDDR4x/5 minimum
        min_storage_gb=256.0,         # 256GB UFS storage
        interface_type="USB-C",
        target_model_name="Qwen3-4B-INT4",
        target_model_params_b=4.0,
        weight_bits=4,
        min_tokens_per_sec=12.0,      # Interactive decoding target
    )

    print(f"\n[TARGET SPECIFICATION]")
    print(f"  Physical Envelope: {envelope.enclosure_length_mm} x {envelope.enclosure_width_mm} x {envelope.enclosure_height_mm} mm")
    print(f"  Maximum Power:     {envelope.max_power_w:.1f} W")
    print(f"  Thermal Limit:     {envelope.max_junction_temp_c:.1f} °C (ambient {envelope.ambient_temp_c:.1f} °C, theta_ja={envelope.thermal_resistance_c_per_w}°C/W)")
    print(f"  Memory & Storage:  >= {envelope.min_ram_gb:.1f} GB RAM, >= {envelope.min_storage_gb:.1f} GB Storage")
    print(f"  Interface:         {envelope.interface_type} (USB 3.2 Gen 2)")
    print(f"  AI Workload:       {envelope.target_model_name} ({envelope.target_model_params_b:.1f}B params, {envelope.weight_bits}-bit)")
    print(f"  Target Speed:      >= {envelope.min_tokens_per_sec:.1f} tokens/sec")

    # 2. Run the Physical Constraint Evolution Loop
    loop = PhysicalConstraintEvolutionLoop(
        envelope=envelope,
        task_id="L3_MAC_8BIT_SIGNED",
        max_generations=5,
        candidates_per_generation=4,
    )

    print("\n>>> Launching Generational Constraint Satisfaction Loop...")
    report = loop.run()

    # 3. Present Generation History
    print("\n" + "=" * 80)
    print("  GENERATION-BY-GENERATION EVOLUTION PROGRESSION")
    print("=" * 80)
    for g in report.generation_history:
        status_marker = "[PASS ALL]" if g.passed else "[FAILED CONSTRAINTS]"
        print(f"\n* Generation {g.generation}: {status_marker} Candidate: {g.best_candidate_id}")
        print(f"    Mutation Applied:    {g.mutation_applied}")
        print(f"    Power:               {g.total_power_w:.2f} W (Limit: {envelope.max_power_w:.1f} W)")
        print(f"    Junction Temp:       {g.junction_temp_c:.1f} °C (Limit: {envelope.max_junction_temp_c:.1f} °C)")
        print(f"    PCB Area:            {g.total_pcb_area_mm2:.1f} mm² (Max: {envelope.max_pcb_area_mm2:.1f} mm²)")
        print(f"    AI Throughput:       {g.tokens_per_sec:.1f} tok/s (Target: {envelope.min_tokens_per_sec:.1f} tok/s)")
        print(f"    Checks Breakdown:    {g.checks}")
        if g.violations:
            print(f"    Active Violations:   {g.violations}")
            print(f"    Agent Diagnosis/Fix: {g.recommendations[0] if g.recommendations else 'N/A'}")

    # 4. Final Verdict & Export
    print("\n" + "=" * 80)
    print("  FINAL EVOLUTION SUMMARY")
    print("=" * 80)
    print(f"  Target Achieved:       {report.passed_all_constraints}")
    print(f"  Total Generations:     {report.total_generations}")
    if report.winning_candidate:
        win = report.winning_candidate
        print(f"  Winning Architecture:  {win.architecture_id}")
        print(f"    Datapath Structure:  {win.datapath_structure}")
        print(f"    Pipeline Depth:      {win.pipeline_depth}")
        print(f"    Parallelism Lanes:   {win.parallelism}")
        print(f"    Memory Organization: {win.memory_organization}")
        print(f"    Arithmetic Strategy: {win.arithmetic_strategy}")
        print(f"    Interface Strategy:  {win.interface_strategy}")

    # Export machine-readable report
    out_dir = "./sim_build/physical_envelope"
    os.makedirs(out_dir, exist_ok=True)
    report_file = os.path.join(out_dir, "envelope_evolution_report.json")
    
    report_dict = {
        "target_name": report.target_name,
        "passed_all_constraints": report.passed_all_constraints,
        "total_generations": report.total_generations,
        "winning_candidate_id": report.winning_candidate.architecture_id if report.winning_candidate else None,
        "generations": [asdict(g) for g in report.generation_history],
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    genealogy_file = os.path.join(out_dir, "target_genealogy.json")
    with open(genealogy_file, "w", encoding="utf-8") as f:
        json.dump(report.genealogy_metadata, f, indent=2)

    print(f"\n[REPORT PERSISTENCE]")
    print(f"  Evolution Report: {report_file}")
    print(f"  Genealogy Graph:  {genealogy_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
