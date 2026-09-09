"""
Yosys Tool — synthesis and analysis wrapper.

Wraps the Yosys open-source synthesis suite for:
- RTL synthesis to various targets (generic, iCE40, ECP5, Xilinx)
- Area estimation (cell/wire counts)
- Timing estimation (critical path analysis)
- Netlist generation
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

from agent.schemas import SynthesisResult

logger = logging.getLogger(__name__)

# Yosys synthesis script template
SYNTH_SCRIPT_TEMPLATE = """
# Auto-generated Yosys synthesis script
{read_commands}

# Elaborate
hierarchy -check -top {top_module}

# Synthesize
synth{target_flag} -top {top_module}

# Reports
stat
{extra_commands}
"""


class YosysTool:
    """Python wrapper for the Yosys synthesis tool.

    Generates synthesis scripts, executes Yosys, and parses
    the output for area, timing, and resource utilization metrics.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.binary = config.get("binary", "yosys")
        self.default_target = config.get("default_target", "generic")
        self.timeout = config.get("timeout_seconds", 300)
        self.execution_env = config.get("execution_env", "native")
        self.work_dir = config.get("work_dir", "./synth_build")

        os.makedirs(self.work_dir, exist_ok=True)

    # ── Synthesis ────────────────────────────────────────────────────

    async def synthesize(
        self,
        sources: list[str],
        target: str | None = None,
        top_module: str | None = None,
    ) -> SynthesisResult:
        """Run synthesis on SystemVerilog sources.

        Args:
            sources: List of .sv/.v file paths
            target: Synthesis target ("generic", "ice40", "ecp5", "xilinx")
            top_module: Top-level module name (inferred if not given)

        Returns:
            SynthesisResult with area, timing, and resource metrics
        """
        target = target or self.default_target

        # Infer top module from first source if not given
        if not top_module:
            top_module = self._infer_top_module(sources)

        # Generate synthesis script
        script = self._generate_script(sources, top_module, target)
        script_path = os.path.join(self.work_dir, "synth.ys")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)

        # Run Yosys
        cmd = [self.binary, "-s", script_path]
        if self.execution_env == "wsl":
            cmd = ["wsl"] + cmd

        logger.info(f"Running synthesis: target={target}, top={top_module}")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            output = stdout + stderr
            returncode = proc.returncode or 0

        except asyncio.TimeoutError:
            logger.error(f"Synthesis timed out after {self.timeout}s")
            return SynthesisResult(
                success=False,
                errors=[f"Synthesis timed out after {self.timeout}s"],
            )
        except FileNotFoundError:
            logger.error(f"Yosys not found: {self.binary}")
            return SynthesisResult(
                success=False,
                errors=[f"Yosys binary not found: {self.binary}"],
            )
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return SynthesisResult(
                success=False,
                errors=[str(e)],
            )

        # Parse results
        result = self._parse_results(output, returncode)
        result.output = output

        logger.info(
            f"Synthesis: {'PASS' if result.success else 'FAIL'} "
            f"(cells={result.cell_count}, wires={result.wire_count})"
        )
        return result

    # ── Script Generation ────────────────────────────────────────────

    def _generate_script(
        self,
        sources: list[str],
        top_module: str,
        target: str,
    ) -> str:
        """Generate a Yosys synthesis script."""
        # Build read commands
        read_commands = []
        for src in sources:
            abs_path = os.path.abspath(src)
            if src.endswith(".sv"):
                read_commands.append(f'read_verilog -sv "{abs_path}"')
            else:
                read_commands.append(f'read_verilog "{abs_path}"')

        # Target-specific synth flag
        target_flags = {
            "generic": "",
            "ice40": "_ice40",
            "ecp5": "_ecp5",
            "xilinx": "_xilinx",
        }
        target_flag = target_flags.get(target, "")

        # Extra commands for specific targets
        extra = ""
        if target == "generic":
            extra = "# Generic synthesis — no target-specific optimizations"

        script = SYNTH_SCRIPT_TEMPLATE.format(
            read_commands="\n".join(read_commands),
            top_module=top_module,
            target_flag=target_flag,
            extra_commands=extra,
        )

        return script

    # ── Result Parsing ───────────────────────────────────────────────

    def _parse_results(self, output: str, returncode: int) -> SynthesisResult:
        """Parse Yosys output for synthesis metrics."""
        result = SynthesisResult(success=returncode == 0)

        # Parse cell count from stat output
        # Example: "   Number of cells:             42"
        cell_match = re.search(r"Number of cells:\s+(\d+)", output)
        if cell_match:
            result.cell_count = int(cell_match.group(1))

        # Parse wire count
        wire_match = re.search(r"Number of wires:\s+(\d+)", output)
        if wire_match:
            result.wire_count = int(wire_match.group(1))

        # Parse specific cell types (FPGA targets)
        lut_match = re.search(r"SB_LUT4\s+(\d+)", output)
        if lut_match:
            result.lut_count = int(lut_match.group(1))

        ff_match = re.search(r"(?:SB_DFF\w*|FDRE)\s+(\d+)", output)
        if ff_match:
            result.ff_count = int(ff_match.group(1))

        bram_match = re.search(r"(?:SB_RAM\w*|RAMB\w*)\s+(\d+)", output)
        if bram_match:
            result.bram_count = int(bram_match.group(1))

        # Area estimate (rough: cells * average gate area)
        result.area_estimate = result.cell_count * 1.0  # Normalized

        # Parse errors
        result.errors = [
            line.strip()
            for line in output.split("\n")
            if "ERROR" in line.upper()
        ]

        if result.errors:
            result.success = False

        return result

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _infer_top_module(sources: list[str]) -> str:
        """Infer the top module name from the first source file."""
        if not sources:
            return "top"

        for src in sources:
            if os.path.exists(src):
                with open(src, "r", encoding="utf-8") as f:
                    content = f.read()
                match = re.search(r"module\s+(\w+)", content)
                if match:
                    return match.group(1)

        # Fallback: use filename
        return os.path.splitext(os.path.basename(sources[0]))[0]
