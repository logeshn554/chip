"""
Web research tool with strict parameter validation.
Executes targeted technical research using ScrapeGraphAdapter.
Never executes arbitrary shell or script commands.
"""

from __future__ import annotations

import logging
import urllib.parse
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

        target_url = args.url
        if not target_url:
            # When URL is omitted, target query extraction without hardcoded fallbacks
            target_url = f"https://duckduckgo.com/html/?q={urllib.parse.quote_plus(args.query)}"

        try:
            ctx: CompactTechnicalContext = await self.adapter.extract_compact_context(
                url=target_url,
                focused_query=args.query,
                fallback_content="",
            )
        except Exception as e:
            logger.warning(f"Web research extraction failed: {e}")
            return {
                "status": "failed",
                "message": f"Web research extraction failed for query: {args.query} ({e})",
                "query": args.query,
            }

        return {
            "status": "success",
            "source_url": ctx.source_url,
            "title": ctx.title,
            "citation_id": ctx.citation_id,
            "compact_context": ctx.to_prompt_text(),
            "token_count": ctx.token_count,
            "research_record": ctx.to_research_record(),
        }

