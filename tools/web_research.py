"""
Web research tool with strict parameter validation.
Executes targeted technical research using ScrapeGraphAdapter.
Never executes arbitrary shell or script commands.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from pydantic import BaseModel, Field

from scraping.scrapegraph_adapter import ScrapeGraphAdapter, CompactTechnicalContext

logger = logging.getLogger(__name__)


class WebResearchArgs(BaseModel):
    """Strictly validated parameters for SEARCH_WEB tool."""
    query: str = Field(..., min_length=3, max_length=200, description="Focused technical search query")
    url: Optional[str] = Field(None, description="Specific documentation URL to extract from")
    max_results: int = Field(default=3, ge=1, le=5)


class WebResearchTool:
    """Tool for targeted web research on hardware design topics."""

    def __init__(self, adapter: Optional[ScrapeGraphAdapter] = None):
        self.adapter = adapter or ScrapeGraphAdapter()

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Validate arguments and perform targeted research."""
        try:
            args = WebResearchArgs(**kwargs)
        except Exception as e:
            return {"status": "error", "message": f"Invalid arguments for SEARCH_WEB: {e}"}

        logger.info(f"Executing web research query: {args.query}")

        # Curated technical fallback documentation for offline/reproducible execution
        curated_knowledge = {
            "mac": (
                "Title: Signed Multiply-Accumulate (MAC) Architecture in SystemVerilog\n"
                "A standard 8-bit signed MAC unit calculates: result = (a * b) + acc.\n"
                "Key Rules for SystemVerilog synthesizable implementation:\n"
                "1. Port signals must be declared 'input logic signed [7:0] a, b;' and 'output logic signed [31:0] out;'.\n"
                "2. The product is 16-bit signed: 'logic signed [15:0] product; always_comb product = a * b;'.\n"
                "3. In synchronous accumulation, sign-extend product to 32 bits before adding to accumulator register:\n"
                "   acc_reg <= acc_reg + {{16{product[15]}}, product};\n"
                "4. Active-low synchronous or asynchronous reset clears accumulator register."
            ),
            "verilator": (
                "Title: Verilator SystemVerilog Linting Guide\n"
                "Verilator enforces strict typing: %Error-WIDTH occurs when assigning narrower signals to wider signals "
                "without explicit bit-slicing or sign extension. Use signed casts '$signed()' or replication operators."
            ),
        }

        target_url = args.url or "https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation"
        fallback_doc = ""
        for key, doc in curated_knowledge.items():
            if key in args.query.lower():
                fallback_doc = doc
                break

        ctx: CompactTechnicalContext = await self.adapter.extract_compact_context(
            url=target_url,
            focused_query=args.query,
            fallback_content=fallback_doc,
        )

        return {
            "status": "success",
            "source_url": ctx.source_url,
            "title": ctx.title,
            "citation_id": ctx.citation_id,
            "compact_context": ctx.to_prompt_text(),
            "token_count": ctx.token_count,
        }
