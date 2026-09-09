"""
Verilator Tool — compilation, linting, and simulation wrapper.

Wraps the Verilator CLI for:
- Lint-only checks (fast syntax/semantic validation)
- Full compilation to C++ simulation model
- Simulation execution with optional VCD tracing
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

from agent.schemas import CompileResult, LintResult, SimulationResult

logger = logging.getLogger(__name__)


class VerilatorTool:
    """Python wrapper for the Verilator SystemVerilog simulator.

    Handles execution in both native and WSL2 environments.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.binary = config.get("binary", "verilator")
        self.default_lint_flags = config.get("default_flags", ["--lint-only", "--Wall"])
        self.default_sim_flags = config.get(
            "sim_flags", ["--cc", "--exe", "--build", "--trace"]
        )
        self.timeout = config.get("timeout_seconds", 120)
        self.execution_env = config.get("execution_env", "native")
        self.work_dir = config.get("work_dir", "./sim_build")

        os.makedirs(self.work_dir, exist_ok=True)

    # ── Lint ─────────────────────────────────────────────────────────

    async def lint(self, sources: list[str]) -> LintResult:
        """Run Verilator lint on SystemVerilog sources.

        Fast check for syntax errors, width mismatches,
        undriven signals, etc.

        Args:
            sources: List of .sv file paths

        Returns:
            LintResult with errors, warnings, and info messages
        """
        cmd = [self.binary] + self.default_lint_flags + sources
        stdout, stderr, returncode = await self._run(cmd)

        output = stdout + stderr
        errors = self._parse_messages(output, "Error")
        warnings = self._parse_messages(output, "Warning")
        info = self._parse_messages(output, "Info")

        result = LintResult(
            success=returncode == 0 and len(errors) == 0,
            errors=errors,
            warnings=warnings,
            info=info,
        )

        logger.info(
            f"Lint: {'PASS' if result.success else 'FAIL'} "
            f"({len(errors)} errors, {len(warnings)} warnings)"
        )
        return result

    # ── Compile ──────────────────────────────────────────────────────

    async def compile(
        self,
        sources: list[str],
        top_module: str | None = None,
        trace: bool = True,
    ) -> CompileResult:
        """Compile SystemVerilog sources to a simulation binary.

        Args:
            sources: List of .sv file paths
            top_module: Top-level module name (inferred if not given)
            trace: Enable VCD tracing

        Returns:
            CompileResult with success status and binary path
        """
        flags = list(self.default_sim_flags)
        if top_module:
            flags.extend(["--top-module", top_module])
        if trace and "--trace" not in flags:
            flags.append("--trace")

        flags.extend(["-Mdir", self.work_dir])
        cmd = [self.binary] + flags + sources

        stdout, stderr, returncode = await self._run(cmd)

        output = stdout + stderr
        errors = self._parse_messages(output, "Error")
        warnings = self._parse_messages(output, "Warning")

        # Find the compiled binary
        binary_path = None
        if returncode == 0:
            # Verilator creates V<top_module> binary
            if top_module:
                candidate = os.path.join(self.work_dir, f"V{top_module}")
                if os.path.exists(candidate):
                    binary_path = candidate

        result = CompileResult(
            success=returncode == 0,
            errors=errors,
            warnings=warnings,
            output=output,
            binary_path=binary_path,
        )

        logger.info(f"Compile: {'PASS' if result.success else 'FAIL'}")
        return result

    # ── Simulate ─────────────────────────────────────────────────────

    async def simulate(
        self,
        top_module: str,
        timeout: int | None = None,
    ) -> SimulationResult:
        """Run a compiled simulation.

        Args:
            top_module: Top-level module name
            timeout: Override default timeout

        Returns:
            SimulationResult with pass/fail status and metrics
        """
        binary = os.path.join(self.work_dir, f"V{top_module}")
        if not os.path.exists(binary):
            return SimulationResult(
                success=False,
                errors=[f"Binary not found: {binary}. Run compile() first."],
            )

        timeout = timeout or self.timeout
        stdout, stderr, returncode = await self._run(
            [binary], timeout=timeout
        )

        output = stdout + stderr

        # Parse test results from output
        tests_total, tests_passed, tests_failed = self._parse_test_results(output)

        vcd_path = None
        vcd_candidate = os.path.join(self.work_dir, "dump.vcd")
        if os.path.exists(vcd_candidate):
            vcd_path = vcd_candidate

        result = SimulationResult(
            success=returncode == 0 and tests_failed == 0,
            tests_total=tests_total,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            errors=self._parse_messages(output, "Error"),
            output=output,
            vcd_path=vcd_path,
        )

        logger.info(
            f"Simulate: {'PASS' if result.success else 'FAIL'} "
            f"({tests_passed}/{tests_total} tests passed)"
        )
        return result

    # ── Execution ────────────────────────────────────────────────────

    async def _run(
        self,
        cmd: list[str],
        timeout: int | None = None,
    ) -> tuple[str, str, int]:
        """Execute a command, optionally via WSL2."""
        timeout = timeout or self.timeout

        if self.execution_env == "wsl":
            cmd = ["wsl"] + cmd

        cmd_str = " ".join(cmd)
        logger.debug(f"Running: {cmd_str}")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.work_dir,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            returncode = proc.returncode or 0

            return stdout, stderr, returncode

        except asyncio.TimeoutError:
            logger.error(f"Command timed out after {timeout}s: {cmd_str}")
            proc.kill()
            return "", f"Timeout after {timeout}s", 1

        except FileNotFoundError:
            logger.error(f"Command not found: {cmd[0]}")
            return "", f"Command not found: {cmd[0]}", 127

        except Exception as e:
            logger.error(f"Execution error: {e}")
            return "", str(e), 1

    # ── Parsing ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_messages(output: str, level: str) -> list[str]:
        """Parse Verilator output for messages of a given severity."""
        pattern = rf"%{level}.*?:.*"
        messages = re.findall(pattern, output, re.IGNORECASE)
        return messages

    @staticmethod
    def _parse_test_results(output: str) -> tuple[int, int, int]:
        """Parse test results from simulation output.

        Looks for common test result patterns:
        - "PASS" / "FAIL" keywords
        - "Test X: PASS/FAIL" patterns
        - Summary lines like "X/Y tests passed"
        """
        # Try to find a summary line
        summary = re.search(r"(\d+)/(\d+)\s*(?:tests?\s*)?pass", output, re.IGNORECASE)
        if summary:
            passed = int(summary.group(1))
            total = int(summary.group(2))
            return total, passed, total - passed

        # Count individual PASS/FAIL lines
        passes = len(re.findall(r"\bPASS\b", output))
        fails = len(re.findall(r"\bFAIL\b", output))
        total = passes + fails

        if total == 0:
            # No test indicators — assume single test based on exit code
            return 1, 1, 0  # Will be adjusted by caller based on returncode

        return total, passes, fails
