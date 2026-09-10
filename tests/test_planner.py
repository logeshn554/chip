"""
Tests for HardwarePlanner — dynamic multi-round LLM-driven research planning.

Verifies:
1. No hardcoded keyword decisions (MAC, multiplier, etc.).
2. Backward-compatible decide_research_need() API.
3. Multi-round plan() returns ResearchPlan with all fields.
4. LLM failure falls back conservatively to SEARCH_WEB (never silently skips research).
5. Component categories are populated from task decomposition.
"""

from __future__ import annotations

import asyncio
from typing import Any
import pytest

from agent.planner import HardwarePlanner, ResearchPlan


# ── Mock LLM ─────────────────────────────────────────────────────────────────

class MockLLM:
    """LLM stub that returns pre-configured JSON responses."""

    def __init__(self, responses: list[dict[str, Any]]):
        self._responses = iter(responses)

    async def generate_json(self, prompt: str) -> dict[str, Any]:
        try:
            return next(self._responses)
        except StopIteration:
            return {}

    async def generate(self, prompt: str, **kwargs) -> str:
        return ""


def make_planner(responses: list[dict[str, Any]]) -> HardwarePlanner:
    return HardwarePlanner(llm=MockLLM(responses))


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPlannerNoDynamicDecision:
    """The planner must never use hardcoded keyword matching."""

    def test_mac_task_always_goes_through_llm(self):
        """'mac' keyword must NOT trigger a hardcoded shortcut."""
        # LLM says memory is sufficient → planner should respect it
        planner = make_planner([
            # Round 0: decompose
            {"open_questions": ["What is the signed multiplication rule?"], "physical_design": False, "component_categories": []},
            # Round 1: research decision
            {"needs_research": False, "memory_sufficient": True, "reasoning": "Standard MAC — textbook knowledge.", "recommended_action": "RETRIEVE_MEMORY"},
        ])
        result = asyncio.run(planner.plan("Design an 8-bit signed MAC unit"))
        # Result should reflect LLM output, not a hardcoded shortcut
        assert result.memory_sufficient is True
        assert result.recommended_action == "RETRIEVE_MEMORY"

    def test_mac_keyword_without_llm_support_defaults_to_search(self):
        """If the LLM fails on a 'mac' task, the planner must default to SEARCH_WEB, not a hardcoded skip."""
        planner = make_planner([])  # LLM always fails
        result = asyncio.run(planner.plan("Design an 8-bit signed MAC"))
        # Conservative default: never skip research
        assert result.needs_research is True
        assert result.recommended_action == "SEARCH_WEB"
        assert result.memory_sufficient is False

    def test_novel_task_triggers_search(self):
        """A novel hardware task should always generate queries."""
        planner = make_planner([
            # Round 0
            {"open_questions": ["What is the RISC-V N extension spec?"], "physical_design": False, "component_categories": []},
            # Round 1
            {"needs_research": True, "memory_sufficient": False, "reasoning": "Unknown ISA extension.", "recommended_action": "SEARCH_WEB"},
            # Round 2
            {"queries": ["RISC-V N extension user-level interrupt specification", "riscv-spec-v2.2 N extension PDF"]},
        ])
        result = asyncio.run(planner.plan("Design a RISC-V processor with N extension"))
        assert result.needs_research is True
        assert len(result.search_queries) == 2
        assert "RISC-V" in result.search_queries[0]


class TestPlannerMultiRound:
    """Multi-round pipeline produces ResearchPlan with all fields."""

    def test_full_pipeline_populates_all_fields(self):
        planner = make_planner([
            # Round 0: decompose
            {
                "open_questions": ["Power budget?", "Package type?"],
                "physical_design": True,
                "component_categories": ["dram", "pmic", "storage"],
            },
            # Round 1: research decision
            {
                "needs_research": True,
                "memory_sufficient": False,
                "reasoning": "Physical BOM requires external datasheets.",
                "recommended_action": "SEARCH_WEB",
            },
            # Round 2: query generation
            {
                "queries": [
                    "LPDDR5 8GB FBGA power consumption datasheet",
                    "USB-C PD controller 5W QFN datasheet site:ti.com OR site:microchip.com",
                    "UFS 3.1 256GB eMMC BGA package dimensions",
                ],
            },
        ])
        plan = asyncio.run(
            planner.plan("Design an independent portable AI computer with USB-C, 8GB LPDDR5, 256GB storage")
        )

        assert isinstance(plan, ResearchPlan)
        assert plan.physical_design is True
        assert set(plan.component_categories) == {"dram", "pmic", "storage"}
        assert len(plan.open_questions) == 2
        assert plan.needs_research is True
        assert len(plan.search_queries) == 3
        assert plan.recommended_action == "SEARCH_WEB"

    def test_memory_sufficient_skips_query_generation(self):
        """If memory_sufficient=True, no queries should be generated."""
        planner = make_planner([
            # Round 0
            {"open_questions": [], "physical_design": False, "component_categories": []},
            # Round 1
            {"needs_research": False, "memory_sufficient": True, "reasoning": "Standard logic.", "recommended_action": "RETRIEVE_MEMORY"},
            # Round 2 should NOT be called
        ])
        plan = asyncio.run(planner.plan("Design a simple 4-bit ripple carry adder"))
        assert plan.memory_sufficient is True
        assert plan.search_queries == []


class TestPlannerBackwardCompatAPI:
    """decide_research_need() must return a dict with expected keys."""

    def test_decide_research_need_returns_expected_keys(self):
        planner = make_planner([
            {"open_questions": ["What voltage?"], "physical_design": True, "component_categories": ["pmic"]},
            {"needs_research": True, "memory_sufficient": False, "reasoning": "Need datasheets.", "recommended_action": "SEARCH_WEB"},
            {"queries": ["TPS65988 USB-C PD controller datasheet"]},
        ])
        result = asyncio.run(
            planner.decide_research_need("Design USB-C power delivery system")
        )
        required_keys = {
            "needs_research", "reasoning", "focused_query", "recommended_action",
            "search_queries", "open_questions", "component_categories",
            "physical_design", "memory_sufficient",
        }
        assert required_keys.issubset(result.keys())
        assert isinstance(result["search_queries"], list)
        assert isinstance(result["component_categories"], list)

    def test_focused_query_is_first_search_query_when_available(self):
        planner = make_planner([
            {"open_questions": ["Spec?"], "physical_design": False, "component_categories": []},
            {"needs_research": True, "memory_sufficient": False, "reasoning": "External spec needed.", "recommended_action": "SEARCH_WEB"},
            {"queries": ["RISC-V privileged spec PDF", "risc-v isa spec github"]},
        ])
        result = asyncio.run(
            planner.decide_research_need("Implement a RISC-V machine-mode exception handler")
        )
        assert result["focused_query"] == "RISC-V privileged spec PDF"

    def test_fallback_focused_query_is_task_when_llm_fails(self):
        """If LLM fails and no queries are produced, focused_query should be the task itself."""
        planner = make_planner([])  # All LLM calls fail
        task = "Design a mysterious hardware widget"
        result = asyncio.run(planner.decide_research_need(task))
        assert result["focused_query"] == task
