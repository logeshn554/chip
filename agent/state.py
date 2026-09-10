"""
Agent State and Context Budgeting for Qwen-14B.

Maintains bounded active context:
CURRENT TASK
+
RELEVANT MEMORY
+
RELEVANT WEB EXTRACTS
+
CURRENT RTL
+
CURRENT ERROR
+
LATEST TOOL RESULTS

Summarizes old trajectory information to prevent context overflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Working state of a hardware design episode."""
    task: str
    active_filename: str = "mac.sv"
    current_rtl: str = ""
    current_error: Optional[dict[str, Any]] = None  # Structured failure {stage, status, error, file, line}
    latest_tool_result: Optional[dict[str, Any]] = None
    relevant_memory: list[str] = field(default_factory=list)
    relevant_web_extracts: list[str] = field(default_factory=list)
    action_history: list[dict[str, Any]] = field(default_factory=list)
    step_summaries: list[str] = field(default_factory=list)
    current_reward: float = 0.0
    iteration: int = 0
    max_iterations: int = 15
    is_finished: bool = False
    trajectory_steps: list[dict[str, Any]] = field(default_factory=list)
    episode_id: str = field(default_factory=lambda: f"ep_{int(datetime.now(timezone.utc).timestamp())}")

    def add_step(self, action: str, result: str, details: Optional[dict[str, Any]] = None) -> None:
        """Record step and produce compact summary to conserve context."""
        step_dict = {"action": action.lower(), "result": result}
        if details:
            step_dict.update(details)
        self.trajectory_steps.append(step_dict)
        self.action_history.append({"iteration": self.iteration, "action": action, "result": result})

        # Compact summary line
        summary_line = f"Step {self.iteration}: {action} -> {result}"
        self.step_summaries.append(summary_line)

    def build_model_context(self) -> str:
        """
        Assemble the strictly bounded prompt context for Qwen-14B:
        CURRENT TASK
        +
        RELEVANT MEMORY
        +
        RELEVANT WEB EXTRACTS
        +
        CURRENT RTL
        +
        CURRENT ERROR
        +
        LATEST TOOL RESULTS
        """
        sections = []

        # 1. CURRENT TASK
        sections.append(f"## CURRENT TASK\n{self.task}")

        # 2. STEP HISTORY SUMMARY (Compressed)
        if self.step_summaries:
            recent = self.step_summaries[-5:]  # Keep only last 5 steps
            sections.append(f"## RECENT PROGRESS SUMMARY\n" + "\n".join(f"- {s}" for s in recent))

        # 3. RELEVANT MEMORY
        if self.relevant_memory:
            sections.append(f"## RELEVANT MEMORY GUIDELINES\n" + "\n\n".join(self.relevant_memory[-2:]))

        # 4. RELEVANT WEB EXTRACTS (Sanitized compact context only)
        if self.relevant_web_extracts:
            sections.append(f"## RELEVANT TECHNICAL RESEARCH\n" + "\n\n".join(self.relevant_web_extracts[-2:]))

        # 5. CURRENT RTL
        if self.current_rtl:
            # Bound display to 150 lines
            rtl_lines = self.current_rtl.splitlines()
            bounded_rtl = "\n".join(rtl_lines[:150])
            if len(rtl_lines) > 150:
                bounded_rtl += "\n// ... [truncated for context limit]"
            sections.append(f"## CURRENT RTL ({self.active_filename})\n```systemverilog\n{bounded_rtl}\n```")
        else:
            sections.append("## CURRENT RTL\n(None generated yet. Next action should create initial RTL.)")

        # 6. CURRENT ERROR (Structured)
        if self.current_error and self.current_error.get("status") == "failed":
            sections.append(
                f"## CURRENT ERROR (STAGE: {self.current_error.get('stage', 'unknown').upper()})\n"
                f"```json\n{json.dumps(self.current_error, indent=2)}\n```\n"
                f"Analyze this error, identify the root cause in the RTL, and formulate a repair."
            )

        # 7. LATEST TOOL RESULTS
        if self.latest_tool_result:
            clean_res = {k: v for k, v in self.latest_tool_result.items() if k not in ["compact_context"]}
            sections.append(f"## LATEST TOOL RESULT\n```json\n{json.dumps(clean_res, indent=2)}\n```")

        return "\n\n" + ("=" * 40) + "\n\n".join(sections) + "\n\n" + ("=" * 40)
