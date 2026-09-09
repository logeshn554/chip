"""
Cocotb Tool — Python-based functional verification wrapper.

Manages cocotb test execution against Verilator-compiled designs,
including Makefile generation, test execution, and result parsing.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

from agent.schemas import SimulationResult

logger = logging.getLogger(__name__)

# Template for cocotb Makefile
COCOTB_MAKEFILE_TEMPLATE = """
# Auto-generated Makefile for cocotb
SIM ?= verilator
TOPLEVEL_LANG ?= verilog

VERILOG_SOURCES = {sources}
TOPLEVEL = {top_module}
MODULE = {test_module}

# Verilator-specific
EXTRA_ARGS += --trace
EXTRA_ARGS += -Wno-fatal

include $(shell cocotb-config --makefiles)/Makefile.sim
""".strip()


class CocotbTool:
    """Python wrapper for cocotb test execution.

    Generates Makefiles, runs cocotb tests, and parses results.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.sim_build_dir = config.get("sim_build_dir", "./sim_build")
        self.simulator = config.get("simulator", "verilator")
        self.log_level = config.get("log_level", "INFO")
        self.timeout = config.get("timeout_seconds", 300)
        self.execution_env = config.get("execution_env", "native")

        os.makedirs(self.sim_build_dir, exist_ok=True)

    # ── Test Execution ───────────────────────────────────────────────

    async def run_tests(
        self,
        sources: list[str],
        top_module: str,
        test_module: str,
        test_dir: str | None = None,
    ) -> SimulationResult:
        """Run cocotb tests against a design.

        Args:
            sources: SystemVerilog source file paths
            top_module: Top-level module name
            test_module: Python test module name (without .py)
            test_dir: Directory containing the test file

        Returns:
            SimulationResult with test pass/fail counts
        """
        test_dir = test_dir or self.sim_build_dir

        # Generate Makefile
        makefile_path = os.path.join(test_dir, "Makefile")
        self._generate_makefile(
            sources=sources,
            top_module=top_module,
            test_module=test_module,
            output_path=makefile_path,
        )

        # Run make
        cmd = ["make", "-C", test_dir, f"SIM={self.simulator}"]
        if self.execution_env == "wsl":
            cmd = ["wsl"] + cmd

        logger.info(f"Running cocotb tests: {test_module} → {top_module}")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={
                    **os.environ,
                    "SIM_BUILD": self.sim_build_dir,
                    "COCOTB_LOG_LEVEL": self.log_level,
                },
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            output = stdout + stderr
            returncode = proc.returncode or 0

        except asyncio.TimeoutError:
            logger.error(f"Cocotb test timed out after {self.timeout}s")
            return SimulationResult(
                success=False,
                errors=[f"Test execution timed out after {self.timeout}s"],
            )
        except FileNotFoundError:
            logger.error("'make' command not found — is cocotb installed?")
            return SimulationResult(
                success=False,
                errors=["'make' command not found. Install cocotb and GNU make."],
            )
        except Exception as e:
            logger.error(f"Cocotb execution error: {e}")
            return SimulationResult(
                success=False,
                errors=[str(e)],
            )

        # Parse cocotb test results
        tests_total, tests_passed, tests_failed = self._parse_cocotb_results(output)

        result = SimulationResult(
            success=returncode == 0 and tests_failed == 0,
            tests_total=tests_total,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            errors=self._extract_errors(output),
            output=output,
            duration_s=0.0,
        )

        logger.info(
            f"Cocotb: {'PASS' if result.success else 'FAIL'} "
            f"({tests_passed}/{tests_total} tests passed)"
        )
        return result

    # ── Makefile Generation ──────────────────────────────────────────

    def _generate_makefile(
        self,
        sources: list[str],
        top_module: str,
        test_module: str,
        output_path: str,
    ) -> None:
        """Generate a Makefile for cocotb test execution."""
        sources_str = " ".join(os.path.abspath(s) for s in sources)

        makefile_content = COCOTB_MAKEFILE_TEMPLATE.format(
            sources=sources_str,
            top_module=top_module,
            test_module=test_module,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(makefile_content)

        logger.debug(f"Generated Makefile at {output_path}")

    # ── Result Parsing ───────────────────────────────────────────────

    @staticmethod
    def _parse_cocotb_results(output: str) -> tuple[int, int, int]:
        """Parse cocotb test output for pass/fail counts.

        Cocotb outputs lines like:
            ** TEST        STATUS  SIM TIME (ns)  REAL TIME (s)  RATIO (ns/s) **
            ** test_alu    PASS    100.00          0.12           833.33       **
        """
        passed = 0
        failed = 0

        # Look for individual test results
        test_pattern = re.compile(r"\*\*\s+(\w+)\s+(PASS|FAIL)\s+", re.IGNORECASE)
        for match in test_pattern.finditer(output):
            status = match.group(2).upper()
            if status == "PASS":
                passed += 1
            else:
                failed += 1

        # Also check for summary line
        summary = re.search(
            r"(\d+)\s+passed,\s+(\d+)\s+failed",
            output,
            re.IGNORECASE,
        )
        if summary:
            passed = int(summary.group(1))
            failed = int(summary.group(2))

        total = passed + failed
        return total, passed, failed

    @staticmethod
    def _extract_errors(output: str) -> list[str]:
        """Extract error messages from cocotb output."""
        errors = []

        # Python exceptions
        for match in re.finditer(r"(?:Error|Exception|AssertionError):.*", output):
            errors.append(match.group(0).strip())

        # Cocotb errors
        for match in re.finditer(r"ERROR\s+cocotb\..*", output):
            errors.append(match.group(0).strip())

        return errors
