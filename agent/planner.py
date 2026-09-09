"""
Task Planner — decomposes hardware design tasks into ordered steps.

Uses Qwen3-4B to break down high-level tasks (e.g., "Design an NPU")
into actionable sub-tasks with dependency tracking.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.qwen import QwenClient
from agent.schemas import Plan, PlanStep

logger = logging.getLogger(__name__)


class Planner:
    """Decomposes hardware design tasks into executable plans."""

    def __init__(self, qwen: QwenClient, prompts: dict[str, Any]):
        self.qwen = qwen
        self.prompts = prompts

    async def decompose(self, task: str, context: str = "") -> Plan:
        """Break a hardware design task into ordered sub-steps.

        Args:
            task: High-level task description
            context: Optional context from memory (relevant past designs, docs)

        Returns:
            Plan with ordered PlanStep objects
        """
        logger.info(f"Decomposing task: {task[:100]}...")

        system_prompt = self.prompts.get("system", {}).get("agent", "")
        decompose_prompt = self.prompts.get("planning", {}).get("decompose", "")

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": decompose_prompt.format(task=task) + (
                    f"\n\nAdditional context:\n{context}" if context else ""
                ),
            },
        ]

        result = await self.qwen.generate_structured(messages, temperature=0.5)

        if "error" in result:
            logger.warning(f"Failed to parse plan, using default single-step plan")
            return self._default_plan(task)

        return self._parse_plan(task, result)

    async def replan(self, plan: Plan, feedback: str) -> Plan:
        """Adjust an existing plan based on feedback or failure.

        Args:
            plan: Current plan
            feedback: Error message, test failure, or new information

        Returns:
            Updated Plan
        """
        logger.info(f"Re-planning due to: {feedback[:100]}...")

        replan_prompt = self.prompts.get("planning", {}).get("replan", "")
        plan_summary = self._plan_to_text(plan)

        messages = [
            {
                "role": "system",
                "content": self.prompts.get("system", {}).get("agent", ""),
            },
            {
                "role": "user",
                "content": replan_prompt.format(plan=plan_summary, feedback=feedback),
            },
        ]

        result = await self.qwen.generate_structured(messages, temperature=0.5)

        if "error" in result:
            logger.warning("Replan failed, keeping original plan")
            return plan

        new_plan = self._parse_plan(plan.task, result)
        new_plan.version = plan.version + 1

        # Preserve completed steps from original plan
        completed_ids = {s.id for s in plan.steps if s.status == "completed"}
        for step in new_plan.steps:
            if step.id in completed_ids:
                step.status = "completed"

        return new_plan

    # ── Parsing Helpers ──────────────────────────────────────────────

    def _parse_plan(self, task: str, data: dict[str, Any]) -> Plan:
        """Parse LLM JSON output into a Plan object."""
        steps = []
        raw_steps = data.get("steps", [])

        for i, raw in enumerate(raw_steps):
            step = PlanStep(
                id=raw.get("id", i + 1),
                description=raw.get("description", f"Step {i + 1}"),
                step_type=raw.get("type", "generate"),
                depends_on=raw.get("depends_on", []),
            )
            steps.append(step)

        plan = Plan(task=task, steps=steps)
        logger.info(f"Created plan with {len(steps)} steps")
        return plan

    def _default_plan(self, task: str) -> Plan:
        """Create a default sequential plan when LLM parsing fails."""
        return Plan(
            task=task,
            steps=[
                PlanStep(id=1, description="Search for relevant documentation", step_type="search"),
                PlanStep(id=2, description="Generate RTL design", step_type="generate", depends_on=[1]),
                PlanStep(id=3, description="Generate testbench", step_type="generate", depends_on=[2]),
                PlanStep(id=4, description="Run simulation", step_type="simulate", depends_on=[2, 3]),
                PlanStep(id=5, description="Run synthesis", step_type="synthesize", depends_on=[4]),
                PlanStep(id=6, description="Evaluate and optimize", step_type="optimize", depends_on=[5]),
            ],
        )

    @staticmethod
    def _plan_to_text(plan: Plan) -> str:
        """Convert a Plan to human-readable text for LLM context."""
        lines = [f"Task: {plan.task}", f"Version: {plan.version}", "Steps:"]
        for step in plan.steps:
            status_icon = {
                "completed": "✓",
                "failed": "✗",
                "in_progress": "→",
                "pending": "○",
                "skipped": "–",
            }.get(step.status, "?")
            deps = f" (depends on: {step.depends_on})" if step.depends_on else ""
            lines.append(f"  {status_icon} [{step.id}] {step.description}{deps}")
        return "\n".join(lines)
