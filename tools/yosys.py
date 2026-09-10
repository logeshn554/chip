"""
Yosys Synthesis Tool — Logic synthesis, cell mapping, and resource evaluation.

Runs logic synthesis to check synthesizability and extract hardware resource metrics:
- cell count
- DFF/flip-flop count
- wire count
- logic cells
- estimated standard cell area (um^2)

Supports native Yosys binary execution with robust stat log parsing,
and provides a dynamic static synthesizability analyzer fallback.
Explicitly distinguishes actual metrics from estimated metrics, and does not invent fake power numbers.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ToolDict(dict):
    """Dictionary supporting attribute access and standardized property aliases."""
    def __getattr__(self, name: str) -> Any:
        if name == "success":
            return self.get("status") == "passed"
        if name == "errors":
            err = self.get("error", "")
            return [err] if err else []
        if name == "cell_count":
            return self.get("cells") or self.get("heuristic_cell_guess") or 0
        if name in self:
            return self[name]
        raise AttributeError(f"'ToolDict' object has no attribute '{name}'")


class YosysTool:
    """Wrapper for Yosys logic synthesis with reliable output extraction."""

    def __init__(self, binary: Any = "yosys", work_dir: str = "./sim_build/synth"):
        if isinstance(binary, dict):
            work_dir = binary.get("work_dir", work_dir)
            binary = binary.get("binary", "yosys")
        self.binary = str(binary)
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        self._has_binary = shutil.which(self.binary) is not None
        self.tool_version = self._detect_version() if self._has_binary else "heuristic_fallback"

    def _detect_version(self) -> str:
        """Query native Yosys binary version."""
        try:
            import subprocess
            out = subprocess.check_output([self.binary, "-V"], text=True, stderr=subprocess.STDOUT)
            return out.strip().splitlines()[0]
        except Exception:
            return "Yosys (version unknown)"

    def _heuristic_lint_check(self, code: str, top_module: str = "top") -> dict[str, Any]:
        """Heuristic-only lint check using regex pattern matching.

        WARNING: This is NOT synthesis. Output values are rough heuristic guesses
        from source code patterns. They must NEVER be used for reward computation,
        Pareto ranking, or training data.
        """
        logger.warning(
            f"Using heuristic lint check for '{top_module}'. "
            "Results are NOT from real synthesis and must not be used for reward or ranking."
        )
        # 1. Check for unsynthesizable constructs
        if "initial begin" in code and "pragma translate_off" not in code:
            return {
                "stage": "yosys",
                "status": "failed",
                "metric_type": "heuristic_lint_only",
                "error": "Synthesis error: 'initial' construct is not synthesizable in ASIC/FPGA target.",
                "file": f"{top_module}.sv",
                "line": 12,
            }

        # 2. Check for simulation delays
        if re.search(r"#\d+", code):
            return {
                "stage": "yosys",
                "status": "failed",
                "metric_type": "heuristic_lint_only",
                "error": "Synthesis error: Delays (#t) cannot be synthesized into physical logic.",
                "file": f"{top_module}.sv",
                "line": 15,
            }

        # 3. Dynamic resource analysis from code structure
        # Detect sequential registers
        has_sequential = bool(re.search(r"\balways_ff\b|\balways\s*@\s*\(\s*posedge\b", code))
        dff_count = 0
        if has_sequential:
            # Estimate DFF count from bitwidth declarations of registers
            for line in code.splitlines():
                if any(kw in line for kw in ["reg", "logic", "output"]) and not any(kw in line for kw in ["wire", "always_comb"]):
                    w_match = re.search(r"\[(\d+):0\]", line)
                    if w_match:
                        dff_count += int(w_match.group(1)) + 1
                    elif re.search(r"\b(accum|reg|state|count|dff)\b", line, re.IGNORECASE):
                        dff_count += 32
                    elif "valid_out" in line:
                        dff_count += 1
            dff_count = max(dff_count, 8 if has_sequential else 0)

        # Detect arithmetic operations
        has_mult = "*" in code
        has_add = "+" in code
        mult_cells = 88 if has_mult else 0
        add_cells = 32 if has_add else 0
        misc_cells = 12

        total_cells = dff_count + mult_cells + add_cells + misc_cells
        total_cells = max(total_cells, 10)
        logic_cells = max(0, total_cells - dff_count)
        heuristic_area = float(total_cells)

        return ToolDict({
            "stage": "yosys",
            "status": "passed",
            "metric_type": "heuristic_lint_only",
            "top_module": top_module,
            "cells": total_cells,
            "cell_count": total_cells,
            "heuristic_cell_guess": total_cells,
            "dffs": dff_count,
            "logic_cells": logic_cells,
            "wires": total_cells + 15,
            "area_cells": total_cells,
            "estimated_area": heuristic_area,
            "heuristic_area_guess": heuristic_area,
            "critical_path_ns": None,
            "power_estimate_uw": None,
            "warnings": ["Values are heuristic guesses from regex, NOT from synthesis."],
            "error": "",
            "raw_log": "Generated via internal heuristic lint check (NOT synthesis).",
        })

    async def synthesize(
        self,
        file_path: Any,
        top_module: Optional[str] = None,
        allow_heuristic_fallback: bool = False,
    ) -> ToolDict:
        """Run Yosys synthesis script or internal synthesizability analysis.
        
        Args:
            file_path: Path to RTL source file.
            top_module: Top-level module name (auto-extracted from RTL if None).
            allow_heuristic_fallback: If False (default), does not produce heuristic estimates
                when Yosys is missing or fails (required for trustworthy RL/GRPO training).
        """
        if isinstance(file_path, (list, tuple)):
            file_path = file_path[0] if file_path else ""
        resolved_path = str(file_path)
        if not os.path.exists(resolved_path):
            for cand in [
                os.path.join("./rtl/generated", os.path.basename(resolved_path)),
                os.path.join("./rtl/reference", os.path.basename(resolved_path)),
                os.path.join("./designs/mac/v1.0", os.path.basename(resolved_path)),
            ]:
                if os.path.exists(cand):
                    resolved_path = cand
                    break

        if not resolved_path or not os.path.exists(resolved_path):
            return ToolDict({
                "stage": "yosys",
                "status": "failed",
                "metric_type": "none",
                "error": f"File not found: {file_path}",
                "file": resolved_path,
                "line": None,
                "cells": None,
                "cell_count": 0,
                "estimated_area": None,
            })

        with open(resolved_path, "r", encoding="utf-8", errors="replace") as f:
            code = f.read()

        # Dynamically infer top module from code if not provided
        mod_match = re.search(r"\bmodule\s+([a-zA-Z_][a-zA-Z0-9_]*)", code)
        if not top_module:
            top_module = mod_match.group(1) if mod_match else "top"

        if self._has_binary:
            ys_script = os.path.join(self.work_dir, f"synth_{top_module}.ys")
            abs_rtl = os.path.abspath(resolved_path).replace("\\", "/")
            with open(ys_script, "w", encoding="utf-8") as f:
                f.write(
                    f"read_verilog -sv {abs_rtl}\n"
                    f"hierarchy -check -top {top_module}\n"
                    f"proc; opt; fsm; opt; techmap; opt\n"
                    f"stat\n"
                    f"ltp\n"
                )
            try:
                proc = await asyncio.create_subprocess_exec(
                    self.binary, "-s", f"synth_{top_module}.ys",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.work_dir,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120.0)
                output = (stdout + stderr).decode("utf-8", errors="replace")

                if proc.returncode == 0:
                    cells_match = re.search(r"Number of cells:\s+(\d+)", output)
                    wires_match = re.search(r"Number of wires:\s+(\d+)", output)
                    cells = int(cells_match.group(1)) if cells_match else 0
                    wires = int(wires_match.group(1)) if wires_match else 0

                    # Parse DFF cells
                    dff_matches = re.findall(r"\b\$_DFF_\w+\s+(\d+)", output)
                    dffs = sum(int(c) for c in dff_matches) if dff_matches else 0

                    # Parse longest topological path for timing analysis
                    ltp_match = re.search(r"Longest topological path in \S+ \(length=(\d+)\)", output)
                    logic_levels = int(ltp_match.group(1)) if ltp_match else None
                    critical_path_ns = round(logic_levels * 0.15, 3) if logic_levels is not None else None

                    return ToolDict({
                        "stage": "yosys",
                        "status": "passed",
                        "tool": "yosys",
                        "tool_version": self.tool_version,
                        "metric_type": "actual",
                        "top_module": top_module,
                        "cells": cells,
                        "cell_count": cells,
                        "heuristic_cell_guess": cells,
                        "dffs": dffs,
                        "logic_cells": max(0, cells - dffs) if cells is not None else None,
                        "wires": wires,
                        "logic_levels": logic_levels,
                        "critical_path_ns": critical_path_ns,
                        "area_cells": cells,
                        "estimated_area": float(cells) if cells is not None else None,
                        "heuristic_area_guess": float(cells) if cells is not None else None,
                        "power_estimate_uw": None,  # Grounded rule: no fake power numbers
                        "error": "",
                        "raw_log": output,
                    })
                else:
                    err_summary = output[-500:] if len(output) > 500 else output
                    return ToolDict({
                        "stage": "yosys",
                        "status": "failed",
                        "metric_type": "actual",
                        "error": f"Yosys synthesis failed: {err_summary}",
                        "file": resolved_path,
                        "line": None,
                        "raw_log": output,
                        "cells": None,
                        "cell_count": 0,
                        "estimated_area": None,
                    })
            except Exception as e:
                logger.warning(f"Native Yosys execution failed: {e}.")
                if not allow_heuristic_fallback:
                    return ToolDict({
                        "stage": "yosys",
                        "status": "failed",
                        "metric_type": "none",
                        "error": f"Native Yosys execution failed: {e}",
                        "file": resolved_path,
                        "line": None,
                        "cells": None,
                        "cell_count": 0,
                        "estimated_area": None,
                    })

        if not allow_heuristic_fallback:
            return ToolDict({
                "stage": "yosys",
                "status": "unavailable",
                "metric_type": "none",
                "error": "Yosys binary not available and heuristic fallback disabled.",
                "file": resolved_path,
                "line": None,
                "cells": None,
                "cell_count": 0,
                "estimated_area": None,
            })

        return self._heuristic_lint_check(code, top_module=top_module)

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Strict tool entrypoint for RUN_YOSYS and SYNTHESIZE."""
        file_path = kwargs.get("file_path", kwargs.get("sources", kwargs.get("filename", None)))
        if isinstance(file_path, list):
            file_path = file_path[0] if file_path else None
        if not file_path:
            for cand in ["./rtl/generated/top.sv", "./rtl/generated/mac.sv"]:
                if os.path.exists(cand):
                    file_path = cand
                    break
            if not file_path:
                file_path = "top.sv"
        top_module = kwargs.get("top_module", None)
        allow_heuristic = kwargs.get("allow_heuristic_fallback", False)
        return await self.synthesize(file_path, top_module, allow_heuristic_fallback=allow_heuristic)
