"""
RTL Parser — analysis and extraction of SystemVerilog module information.

Extracts module names, ports, parameters, and dependencies
from SystemVerilog source files for the agent to reason about.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PortInfo:
    """Information about a module port."""
    name: str
    direction: str          # "input", "output", "inout"
    width: str = "1"        # e.g., "8", "[7:0]"
    port_type: str = "wire" # "wire", "reg", "logic"


@dataclass
class ParameterInfo:
    """Information about a module parameter."""
    name: str
    default_value: str = ""
    param_type: str = ""    # "integer", "real", etc.


@dataclass
class ModuleInfo:
    """Parsed information about a SystemVerilog module."""
    name: str
    ports: list[PortInfo] = field(default_factory=list)
    parameters: list[ParameterInfo] = field(default_factory=list)
    instantiations: list[str] = field(default_factory=list)  # Modules instantiated
    line_count: int = 0


class RTLParser:
    """Parses SystemVerilog files to extract structural information.

    Uses regex-based parsing (not a full SV parser) for speed.
    Sufficient for the agent's needs — extracting module boundaries,
    ports, and dependency relationships.
    """

    # ── Module Extraction ────────────────────────────────────────────

    @staticmethod
    def extract_modules(code: str) -> list[ModuleInfo]:
        """Extract all module definitions from SystemVerilog code.

        Args:
            code: SystemVerilog source code

        Returns:
            List of ModuleInfo objects
        """
        modules = []

        # Find module blocks
        module_pattern = re.compile(
            r"module\s+(\w+)\s*"           # module name
            r"(?:#\s*\((.*?)\))?\s*"       # optional parameters
            r"(?:\((.*?)\))?\s*;",         # optional port list
            re.DOTALL,
        )

        for match in module_pattern.finditer(code):
            name = match.group(1)
            param_text = match.group(2) or ""
            port_text = match.group(3) or ""

            # Find the module's end
            module_start = match.start()
            end_match = re.search(r"\bendmodule\b", code[module_start:])
            module_end = module_start + end_match.end() if end_match else len(code)
            module_code = code[module_start:module_end]

            module = ModuleInfo(
                name=name,
                ports=RTLParser._parse_ports(port_text, module_code),
                parameters=RTLParser._parse_parameters(param_text),
                instantiations=RTLParser._find_instantiations(module_code),
                line_count=module_code.count("\n") + 1,
            )
            modules.append(module)

        logger.debug(f"Extracted {len(modules)} modules")
        return modules

    @staticmethod
    def extract_module_names(code: str) -> list[str]:
        """Extract just the module names from code."""
        return [m.group(1) for m in re.finditer(r"module\s+(\w+)", code)]

    # ── Port Parsing ─────────────────────────────────────────────────

    @staticmethod
    def _parse_ports(port_text: str, full_code: str) -> list[PortInfo]:
        """Parse port declarations from the port list or module body."""
        ports = []

        # ANSI-style port declarations (in the port list)
        port_pattern = re.compile(
            r"(input|output|inout)\s+"              # direction
            r"(?:(wire|reg|logic)\s+)?"             # optional type
            r"(?:(\[[\d:]+\])\s*)?"                 # optional width
            r"(\w+)"                                 # port name
        )

        # Search in port text first, then full module code
        for source in [port_text, full_code]:
            for match in port_pattern.finditer(source):
                direction = match.group(1)
                port_type = match.group(2) or "logic"
                width = match.group(3) or "1"
                name = match.group(4)

                # Avoid duplicates
                if not any(p.name == name for p in ports):
                    ports.append(PortInfo(
                        name=name,
                        direction=direction,
                        width=width,
                        port_type=port_type,
                    ))

        return ports

    # ── Parameter Parsing ────────────────────────────────────────────

    @staticmethod
    def _parse_parameters(param_text: str) -> list[ParameterInfo]:
        """Parse parameter declarations."""
        params = []

        param_pattern = re.compile(
            r"parameter\s+"
            r"(?:(\w+)\s+)?"            # optional type
            r"(\w+)\s*=\s*([^,\)]+)"    # name = value
        )

        for match in param_pattern.finditer(param_text):
            param_type = match.group(1) or ""
            name = match.group(2)
            default = match.group(3).strip()

            params.append(ParameterInfo(
                name=name,
                default_value=default,
                param_type=param_type,
            ))

        return params

    # ── Instantiation Detection ──────────────────────────────────────

    @staticmethod
    def _find_instantiations(module_code: str) -> list[str]:
        """Find which modules are instantiated within this module.

        Detects patterns like: module_name instance_name (...)
        Excludes common keywords that look like instantiations.
        """
        exclude = {
            "module", "endmodule", "input", "output", "inout", "wire", "reg",
            "logic", "assign", "always", "initial", "begin", "end", "if",
            "else", "case", "for", "while", "parameter", "localparam",
            "generate", "endgenerate", "function", "endfunction", "task",
            "endtask", "integer", "real", "time",
        }

        # Pattern: identifier identifier ( ... )
        inst_pattern = re.compile(r"(\w+)\s+(\w+)\s*\(")
        instantiated = set()

        for match in inst_pattern.finditer(module_code):
            module_name = match.group(1)
            if module_name.lower() not in exclude:
                instantiated.add(module_name)

        return sorted(instantiated)

    # ── Dependency Graph ─────────────────────────────────────────────

    @staticmethod
    def build_dependency_graph(modules: list[ModuleInfo]) -> dict[str, list[str]]:
        """Build a dependency graph from module information.

        Returns:
            Dict mapping module name → list of modules it depends on
        """
        module_names = {m.name for m in modules}

        graph = {}
        for module in modules:
            # Only include dependencies that are in our module set
            deps = [inst for inst in module.instantiations if inst in module_names]
            graph[module.name] = deps

        return graph

    # ── Validation ───────────────────────────────────────────────────

    @staticmethod
    def basic_lint(code: str) -> list[str]:
        """Perform basic lint checks on SystemVerilog code.

        These are quick sanity checks, not a replacement for Verilator lint.

        Returns:
            List of warning/error messages
        """
        warnings = []

        # Check for balanced module/endmodule
        module_count = len(re.findall(r"\bmodule\b", code))
        endmodule_count = len(re.findall(r"\bendmodule\b", code))
        if module_count != endmodule_count:
            warnings.append(
                f"Unbalanced module/endmodule: {module_count} modules, "
                f"{endmodule_count} endmodules"
            )

        # Check for balanced begin/end
        begin_count = len(re.findall(r"\bbegin\b", code))
        end_count = len(re.findall(r"\bend\b", code)) - endmodule_count
        if begin_count != end_count:
            warnings.append(
                f"Potentially unbalanced begin/end: {begin_count} begins, "
                f"{end_count} ends"
            )

        # Check for missing timescale
        if "`timescale" not in code and module_count > 0:
            warnings.append("Missing `timescale directive")

        # Check for blocking assignments in always_ff
        if re.search(r"always_ff\s.*?=(?!=)", code, re.DOTALL):
            warnings.append("Possible blocking assignment (=) in always_ff block")

        # Check for missing reset
        if module_count > 0 and "rst" not in code.lower() and "reset" not in code.lower():
            warnings.append("No reset signal detected")

        return warnings
