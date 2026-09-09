"""
Targeted ScrapeGraphAI Adapter for Hardware Web Research.

Uses ScrapeGraphAI principles for minimal, focused technical information extraction.
Only targeted technical knowledge enters Qwen's context. Never dumps full pages.
Always preserves source URLs and citations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import urllib.error
import urllib.request
from typing import Any, Optional

from scraping.content_filter import ContentFilter
from scraping.deduplicator import ContentDeduplicator
from scraping.source_tracker import SourceItem, SourceTracker

logger = logging.getLogger(__name__)


@dataclass
class CompactTechnicalContext:
    """The only structure allowed into Qwen's working context from web research."""
    query: str
    source_url: str
    title: str
    extracted_summary: str
    code_snippets: list[str] = field(default_factory=list)
    key_specifications: list[str] = field(default_factory=list)
    token_count: int = 0
    citation_id: str = ""

    def to_prompt_text(self) -> str:
        """Format strictly bounded, injection-safe context for Qwen."""
        parts = [
            f"[EXTERNAL RESEARCH CONTEXT: {self.citation_id}]",
            f"Source URL: {self.source_url}",
            f"Topic/Query: {self.query}",
            f"Summary:\n{self.extracted_summary}",
        ]
        if self.key_specifications:
            parts.append("Key Specifications:\n" + "\n".join(f"- {s}" for s in self.key_specifications))
        if self.code_snippets:
            for i, snip in enumerate(self.code_snippets[:2]):
                parts.append(f"Technical Snippet {i+1}:\n```systemverilog\n{snip}\n```")
        parts.append(f"[END EXTERNAL RESEARCH CONTEXT: {self.citation_id}]")
        return "\n\n".join(parts)


class ScrapeGraphAdapter:
    """Adapter implementing targeted ScrapeGraphAI extraction for hardware design tasks."""

    def __init__(
        self,
        ollama_model: str = "qwen2.5-coder:3b",
        max_page_bytes: int = 250_000,
        timeout: float = 15.0,
        max_context_chars: int = 4000,
    ):
        self.ollama_model = ollama_model
        self.max_page_bytes = max_page_bytes
        self.timeout = timeout
        self.filter = ContentFilter(max_chars=max_context_chars)
        self.deduplicator = ContentDeduplicator()
        self.source_tracker = SourceTracker()
        self._has_sgai = False

        try:
            import scrapegraphai  # type: ignore
            self._has_sgai = True
            logger.info("ScrapeGraphAI library detected.")
        except ImportError:
            logger.info("ScrapeGraphAI not installed; using lightweight native targeted graph adapter.")

    def fetch_url(self, url: str) -> tuple[str, str]:
        """Safely fetch HTML/text with timeout, size cap, and user-agent."""
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "HardwareAgent-ResearchBot/1.0 (Hardware RTL Design Extraction; +https://github.com/agent)",
                "Accept": "text/html,application/xhtml+xml,text/plain",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                content_type = resp.headers.get("Content-Type", "")
                raw_bytes = resp.read(self.max_page_bytes)
                text = raw_bytes.decode("utf-8", errors="replace")
                return text, content_type
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return "", ""

    async def extract_compact_context(
        self,
        url: str,
        focused_query: str,
        fallback_content: Optional[str] = None,
    ) -> CompactTechnicalContext:
        """
        Execute targeted extraction:
        1. Fetch content (or use provided content/docs)
        2. Content filtering (strip HTML, ads, navigation, scripts)
        3. Deduplication of paragraphs
        4. Compact technical structuring
        5. Preserves source URL and citation
        """
        source_item = self.source_tracker.register(url=url)

        raw_text = fallback_content or ""
        content_type = "text/html"

        if not raw_text and url.startswith("http"):
            raw_text, content_type = self.fetch_url(url)

        if not raw_text:
            # Return minimal safe placeholder if fetch failed
            return CompactTechnicalContext(
                query=focused_query,
                source_url=url,
                title=source_item.title,
                extracted_summary="Unable to retrieve external content or page was empty.",
                citation_id=source_item.citation_id,
            )

        # ScrapeGraphAI SmartScraperGraph path if available
        if self._has_sgai and url.startswith("http"):
            try:
                from scrapegraphai.graphs import SmartScraperGraph  # type: ignore
                graph_config = {
                    "llm": {
                        "model": f"ollama/{self.ollama_model}",
                        "base_url": "http://localhost:11434",
                        "temperature": 0.1,
                    },
                    "verbose": False,
                    "headless": True,
                }
                smart_scraper = SmartScraperGraph(
                    prompt=f"Extract only technical hardware design specifications, formulas, and SystemVerilog constructs related to: {focused_query}",
                    source=url,
                    config=graph_config,
                )
                result = smart_scraper.run()
                if result:
                    summary_text = json.dumps(result) if isinstance(result, dict) else str(result)
                    return CompactTechnicalContext(
                        query=focused_query,
                        source_url=url,
                        title=source_item.title,
                        extracted_summary=summary_text[:1500],
                        citation_id=source_item.citation_id,
                    )
            except Exception as e:
                logger.warning(f"ScrapeGraphAI graph failed, falling back to native pipeline: {e}")

        # Native targeted pipeline: ContentFilter + Deduplicator
        is_html = "html" in content_type.lower() or "<html" in raw_text.lower()
        extracted_dict = self.filter.extract_from_html(raw_text) if is_html else {
            "title": url, "paragraphs": [raw_text], "code_blocks": [], "tables": []
        }

        # Deduplicate paragraphs
        deduped_paras = self.deduplicator.deduplicate_paragraphs(extracted_dict.get("paragraphs", []))

        # Re-compact with query guidance
        compact_summary = self.filter.filter_and_compact("\n\n".join(deduped_paras), query=focused_query)
        code_snippets = extracted_dict.get("code_blocks", [])[:2]

        # Extract bullet specifications
        specs = []
        for p in deduped_paras:
            if any(term in p.lower() for term in ["bit", "width", "signed", "latency", "pipeline", "saturat", "overflow"]):
                if len(p) < 160:
                    specs.append(p.strip())
            if len(specs) >= 4:
                break

        ctx = CompactTechnicalContext(
            query=focused_query,
            source_url=url,
            title=extracted_dict.get("title") or source_item.title,
            extracted_summary=compact_summary,
            code_snippets=code_snippets,
            key_specifications=specs,
            token_count=len(compact_summary) // 4,
            citation_id=source_item.citation_id,
        )
        return ctx
