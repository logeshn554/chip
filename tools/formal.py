"""
Formal Verification Tool — SymbiYosys (sby) wrapper for property checking and BMC.

Supports:
- Bounded Model Checking (BMC) using SymbiYosys
- Formal property extraction (assert / assume / cover)
- Reporting strict status: "AVAILABLE", "PASS", "FAIL", "SKIPPED"
- Never treats "tool unavailable" as "verification passed"
- Counterexample extraction and trace diagnostics
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import tempfile
from typing import Any, Optional

logger = logging.getLogger(__name__)


class FormalVerificationTool:
    """Wrapper for SymbiYosys formal verification."""

    def __init__(self, sby_binary: str = "sby", work_dir: str = "./sim_build/formal"):
        self.sby_binary = sby_binary
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        self._has_binary = shutil.which(sby_binary) is not None

    @property
    def is_available(self) -> bool:
        """Check if SymbiYosys is installed and discoverable on PATH."""
        return self._has_binary

    def generate_sby_config(
        self,
        top_module: str,
        rtl_file: str,
        depth: int = 20,
        engine: str = "smtbmc",
        properties_file: Optional[str] = None,
    ) -> str:
        """Generate a standard SymbiYosys .sby configuration with optional external properties."""
        abs_rtl = os.path.abspath(rtl_file).replace("\\", "/")
        
        script_lines = [f"read -formal {os.path.basename(abs_rtl)}"]
        files_lines = [abs_rtl]

        if properties_file and os.path.exists(properties_file):
            abs_prop = os.path.abspath(properties_file).replace("\\", "/")
            script_lines.append(f"read -formal {os.path.basename(abs_prop)}")
            files_lines.append(abs_prop)

        script_lines.append(f"prep -top {top_module}")

        script_str = "\n".join(script_lines)
        files_str = "\n".join(files_lines)

        config_text = f"""[options]
mode bmc
depth {depth}

[engines]
{engine}

[script]
{script_str}

[files]
{files_str}
"""
        return config_text

    def check_embedded_formal_properties(self, code: str) -> dict[str, Any]:
        """Static inspection of SystemVerilog formal assertions and assumptions."""
        assert_matches = re.findall(r"\bassert\s+property\s*\((.*?)\);", code, re.DOTALL)
        assume_matches = re.findall(r"\bassume\s+property\s*\((.*?)\);", code, re.DOTALL)
        cover_matches = re.findall(r"\bcover\s+property\s*\((.*?)\);", code, re.DOTALL)

        return {
            "num_assertions": len(assert_matches),
            "num_assumptions": len(assume_matches),
            "num_covers": len(cover_matches),
            "assertions": [a.strip() for a in assert_matches],
        }

    async def verify(
        self,
        file_path: str,
        top_module: str = "mac",
        depth: int = 20,
        timeout: float = 60.0,
        external_properties: Optional[str] = None,
        external_properties_file: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run formal verification on target SystemVerilog module.

        Supports external formal property specifications outside the generated RTL,
        ensuring model-generated assertions cannot game the verification criteria.

        Returns structured dictionary:
        {
            "status": "PASS" | "FAIL" | "SKIPPED" | "ERROR",
            "stage": "formal",
            "available": bool,
            "engine": "sby" | "static",
            "properties_checked": int,
            "counterexample": Optional[str],
            "output": str,
            "errors": list[str],
            "file": str,
        }
        """
        if not os.path.exists(file_path):
            return {
                "status": "ERROR",
                "stage": "formal",
                "available": self._has_binary,
                "engine": "none",
                "properties_checked": 0,
                "counterexample": None,
                "output": f"RTL file not found: {file_path}",
                "errors": [f"File not found: {file_path}"],
                "file": file_path,
            }

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            code = f.read()

        formal_meta = self.check_embedded_formal_properties(code)
        
        # Check external properties if provided
        external_prop_file_to_use = external_properties_file
        job_dir = os.path.join(self.work_dir, f"job_{top_module}")
        os.makedirs(job_dir, exist_ok=True)

        if external_properties:
            external_prop_file_to_use = os.path.join(job_dir, f"{top_module}_formal_spec.sv")
            with open(external_prop_file_to_use, "w", encoding="utf-8") as f:
                f.write(external_properties)
            ext_meta = self.check_embedded_formal_properties(external_properties)
            formal_meta["num_assertions"] += ext_meta["num_assertions"]

        # If sby binary is NOT available, report SKIPPED (never falsely claim PASS)
        if not self._has_binary:
            logger.info("SymbiYosys (sby) is not installed. Formal verification reported as SKIPPED.")
            return {
                "status": "SKIPPED",
                "stage": "formal",
                "available": False,
                "engine": "sby",
                "properties_checked": formal_meta["num_assertions"],
                "counterexample": None,
                "output": (
                    "SymbiYosys (sby) is not installed on host. "
                    f"RTL & spec contain {formal_meta['num_assertions']} formal assertion(s). "
                    "Verification stage was SKIPPED."
                ),
                "errors": [],
                "file": file_path,
            }

        # Sby is available: create job and execute
        sby_config = self.generate_sby_config(
            top_module=top_module,
            rtl_file=file_path,
            depth=depth,
            properties_file=external_prop_file_to_use,
        )
        sby_file = os.path.join(job_dir, f"{top_module}.sby")

        with open(sby_file, "w", encoding="utf-8") as f:
            f.write(sby_config)

        try:
            cmd = [self.sby_binary, "-f", f"{top_module}.sby"]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=job_dir,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            output = (stdout + stderr).decode("utf-8", errors="replace")

            if "DONE (PASS" in output:
                return {
                    "status": "PASS",
                    "stage": "formal",
                    "available": True,
                    "engine": "sby",
                    "properties_checked": max(1, formal_meta["num_assertions"]),
                    "counterexample": None,
                    "output": output,
                    "errors": [],
                    "file": file_path,
                }
            elif "DONE (FAIL" in output:
                # Look for counterexample / trace file
                trace_file = os.path.join(job_dir, f"{top_module}", "engine_0", "trace.vcd")
                has_trace = os.path.exists(trace_file)
                counterexample_info = f"Counterexample trace saved: {trace_file}" if has_trace else "Assertion failed during BMC unrolling."
                return {
                    "status": "FAIL",
                    "stage": "formal",
                    "available": True,
                    "engine": "sby",
                    "properties_checked": max(1, formal_meta["num_assertions"]),
                    "counterexample": counterexample_info,
                    "output": output,
                    "errors": ["Formal property violation detected by SymbiYosys BMC."],
                    "file": file_path,
                }
            else:
                return {
                    "status": "ERROR",
                    "stage": "formal",
                    "available": True,
                    "engine": "sby",
                    "properties_checked": formal_meta["num_assertions"],
                    "counterexample": None,
                    "output": output,
                    "errors": [f"SymbiYosys exited with return code {proc.returncode}"],
                    "file": file_path,
                }

        except asyncio.TimeoutError:
            return {
                "status": "TIMEOUT",
                "stage": "formal",
                "available": True,
                "engine": "sby",
                "properties_checked": formal_meta["num_assertions"],
                "counterexample": None,
                "output": f"Formal verification timed out after {timeout} seconds.",
                "errors": ["SBY BMC timeout"],
                "file": file_path,
            }
        except Exception as e:
            logger.error(f"Error running SymbiYosys: {e}")
            return {
                "status": "ERROR",
                "stage": "formal",
                "available": True,
                "engine": "sby",
                "properties_checked": 0,
                "counterexample": None,
                "output": str(e),
                "errors": [str(e)],
                "file": file_path,
            }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Strict tool entrypoint for FORMAL_VERIFY."""
        file_path = kwargs.get("file_path", kwargs.get("file", "mac.sv"))
        top_module = kwargs.get("top_module", "mac")
        return await self.verify(file_path=file_path, top_module=top_module)
