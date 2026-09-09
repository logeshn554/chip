"""
RTL Generator — uses Qwen3-4B to generate SystemVerilog modules.

Handles module generation, testbench creation, error fixing,
and design optimization through prompt engineering with
few-shot examples and Jinja2 templates.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from agent.qwen import QwenClient
from agent.schemas import Document, RTLModule

logger = logging.getLogger(__name__)


class RTLGenerator:
    """Generates SystemVerilog RTL using Qwen3-4B.

    Combines prompt templates, few-shot examples from design memory,
    and Jinja2 templates for common patterns to produce synthesizable
    SystemVerilog code.
    """

    def __init__(
        self,
        qwen: QwenClient,
        prompts: dict[str, Any],
        output_dir: str = "./designs",
        templates_dir: str | None = None,
    ):
        self.qwen = qwen
        self.prompts = prompts
        self.output_dir = output_dir
        self.templates_dir = templates_dir or os.path.join(
            os.path.dirname(__file__), "templates"
        )

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"RTLGenerator: output_dir={output_dir}")

    # ── Module Generation ────────────────────────────────────────────

    async def generate_module(
        self,
        spec: str,
        context_docs: list[Document] | None = None,
        name: str | None = None,
    ) -> RTLModule:
        """Generate a SystemVerilog module from a specification.

        Args:
            spec: Natural language specification
            context_docs: Relevant documents from memory
            name: Optional module name (inferred from spec if not given)

        Returns:
            RTLModule with generated code
        """
        logger.info(f"Generating module: {spec[:80]}...")

        # Build context from memory docs
        context = ""
        if context_docs:
            context = "Reference information:\n" + "\n\n".join(
                f"---\n{doc.content}" for doc in context_docs[:3]
            )

        # Build prompt
        system_prompt = self.prompts.get("system", {}).get("agent", "")
        generate_prompt = self.prompts.get("rtl", {}).get("generate_module", "")

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": generate_prompt.format(spec=spec, context=context),
            },
        ]

        response = await self.qwen.generate(messages, temperature=0.3)

        # Extract SystemVerilog code from response
        code = self.qwen.extract_code(response.content, "systemverilog")

        # Parse module name from code if not provided
        if not name:
            name = self._extract_module_name(code) or "generated_module"

        # Save to file
        filepath = os.path.join(self.output_dir, f"{name}.sv")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)

        module = RTLModule(
            name=name,
            code=code,
            filepath=filepath,
            description=spec,
        )

        logger.info(f"Generated module '{name}' → {filepath}")
        return module

    async def generate_testbench_code(self, module_code: str) -> str:
        """Generate a cocotb Python testbench for a module.

        Args:
            module_code: The SystemVerilog module code

        Returns:
            Python testbench code string
        """
        logger.info("Generating testbench...")

        generate_tb_prompt = self.prompts.get("rtl", {}).get("generate_testbench", "")

        messages = [
            {
                "role": "system",
                "content": self.prompts.get("system", {}).get("agent", ""),
            },
            {
                "role": "user",
                "content": generate_tb_prompt.format(module_code=module_code),
            },
        ]

        response = await self.qwen.generate(messages, temperature=0.3)
        tb_code = self.qwen.extract_code(response.content, "python")

        return tb_code

    async def generate_testbench(self, module: RTLModule) -> RTLModule:
        """Generate a testbench and attach it to the module.

        Args:
            module: The RTL module to create a testbench for

        Returns:
            Updated RTLModule with testbench attached
        """
        tb_code = await self.generate_testbench_code(module.code)
        module.testbench = tb_code

        # Save testbench file
        if module.filepath:
            tb_dir = os.path.dirname(module.filepath)
            tb_path = os.path.join(tb_dir, f"test_{module.name}.py")
            with open(tb_path, "w", encoding="utf-8") as f:
                f.write(tb_code)
            logger.info(f"Generated testbench → {tb_path}")

        return module

    # ── Error Fixing ─────────────────────────────────────────────────

    async def fix_errors(
        self,
        code: str,
        errors: list[str],
        similar_fixes: list[Document] | None = None,
    ) -> str:
        """Fix errors in SystemVerilog code.

        Args:
            code: The broken SystemVerilog code
            errors: List of error messages
            similar_fixes: Past similar error fixes from experience memory

        Returns:
            Fixed SystemVerilog code
        """
        logger.info(f"Fixing {len(errors)} errors...")

        fix_prompt = self.prompts.get("rtl", {}).get("fix_errors", "")
        fixes_context = ""
        if similar_fixes:
            fixes_context = "\n\n".join(
                f"Previous fix:\n{doc.content}" for doc in similar_fixes[:3]
            )

        messages = [
            {
                "role": "system",
                "content": self.prompts.get("system", {}).get("agent", ""),
            },
            {
                "role": "user",
                "content": fix_prompt.format(
                    code=code,
                    errors="\n".join(errors),
                    similar_fixes=fixes_context or "No similar fixes found.",
                ),
            },
        ]

        response = await self.qwen.generate(messages, temperature=0.2)
        fixed_code = self.qwen.extract_code(response.content, "systemverilog")

        return fixed_code

    # ── Optimization ─────────────────────────────────────────────────

    async def optimize(
        self,
        code: str,
        optimize_for: str = "area",
        current_metrics: dict[str, Any] | None = None,
    ) -> str:
        """Optimize a design for a specific metric.

        Args:
            code: Current SystemVerilog code
            optimize_for: "area", "timing", or "power"
            current_metrics: Current synthesis metrics

        Returns:
            Optimized SystemVerilog code
        """
        logger.info(f"Optimizing for {optimize_for}...")

        metrics = current_metrics or {}
        opt_prompts = self.prompts.get("optimization", {})

        if optimize_for == "area":
            prompt = opt_prompts.get("optimize_area", "").format(
                code=code,
                cell_count=metrics.get("cell_count", "unknown"),
                wire_count=metrics.get("wire_count", "unknown"),
            )
        elif optimize_for == "timing":
            prompt = opt_prompts.get("optimize_timing", "").format(
                code=code,
                critical_path_ns=metrics.get("critical_path_ns", "unknown"),
                target_ns=metrics.get("target_ns", "10.0"),
            )
        else:
            prompt = f"Optimize the following design for {optimize_for}:\n```systemverilog\n{code}\n```"

        messages = [
            {
                "role": "system",
                "content": self.prompts.get("system", {}).get("agent", ""),
            },
            {"role": "user", "content": prompt},
        ]

        response = await self.qwen.generate(messages, temperature=0.3)
        optimized = self.qwen.extract_code(response.content, "systemverilog")

        return optimized

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_module_name(code: str) -> str | None:
        """Extract the module name from SystemVerilog code."""
        import re
        match = re.search(r"module\s+(\w+)", code)
        return match.group(1) if match else None
