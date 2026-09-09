"""
Hardware Task Planner.

Decomposes hardware design tasks into an actionable plan.
Decides whether external web research is necessary or if internal memory suffices.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from llm.interface import LLMInterface
from agent.prompts import RESEARCH_DECISION_PROMPT

logger = logging.getLogger(__name__)


class HardwarePlanner:
    """Plans hardware design workflows and decides on research actions."""

    def __init__(self, llm: LLMInterface):
        self.llm = llm

    async def decide_research_need(self, task: str) -> dict[str, Any]:
        """
        Decide whether external research is necessary:
        - Common standard modules (like standard MAC, ALU, FIFO) rely on internal KnowledgeStore.
        - Obscure protocols or novel datasheets trigger targeted SEARCH_WEB.
        """
        prompt = RESEARCH_DECISION_PROMPT.format(task=task)
        try:
            decision = await self.llm.generate_json(prompt)
            if isinstance(decision, dict) and "needs_research" in decision:
                return decision
        except Exception as e:
            logger.warning(f"Research decision parsing failed: {e}")

        # Deterministic fallback based on task keywords
        lower = task.lower()
        if "mac" in lower or "multiplier" in lower or "8-bit signed" in lower:
            return {
                "needs_research": False,
                "reasoning": "Signed MAC is a well-defined standard architecture. Internal knowledge memory is sufficient.",
                "focused_query": "8-bit signed MAC SystemVerilog",
                "recommended_action": "RETRIEVE_MEMORY",
            }

        return {
            "needs_research": True,
            "reasoning": "External specification may be required for complex architecture.",
            "focused_query": task,
            "recommended_action": "SEARCH_WEB",
        }
