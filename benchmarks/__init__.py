"""
Benchmark Curriculum Package — 7-level hardware design curriculum.

Levels:
- Level 1: Basic Gates & Combinational Logic (NOT, AND/OR, MUX, Decoder, Counter)
- Level 2: Data-path & Storage (ALU, Register File, FIFO, Shift Register)
- Level 3: Arithmetic Units (8-bit signed MAC, 16-bit signed MAC)
- Level 4: Matrix & Vector Units (Vector ALU, 2x2 Matrix Multiplier)
- Level 5: Memory Subsystems & Interconnect (SRAM Controller, DMA Component)
- Level 6: RISC-V Core Components (ALU, Decoder, Program Counter)
- Level 7: NPU Accelerator Components (Processing Element, Systolic Array Row)
"""

from benchmarks.curriculum import (
    BenchmarkCurriculum,
    BenchmarkTask,
    get_benchmark,
    get_curriculum_level,
)

__all__ = [
    "BenchmarkCurriculum",
    "BenchmarkTask",
    "get_benchmark",
    "get_curriculum_level",
]
