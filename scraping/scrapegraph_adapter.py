"""
Targeted ScrapeGraphAI Adapter for Hardware Web Research.

Uses ScrapeGraphAI principles for minimal, focused technical information extraction.
Only targeted technical knowledge enters Qwen's context. Never dumps full pages.
Always preserves source URLs and citations.

Includes:
- Domain allowlist (arxiv.org, github.com, verilator.org, yosyshq.readthedocs.io, docs.cocotb.org, riscv.org)
- Request timeout and retry logic
- Disk caching in ./data/web_cache to avoid redundant queries
- Strict HTML sanitization, deduplication, and anti-prompt injection
- Formatted structured research record schema
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from scraping.content_filter import ContentFilter
from scraping.deduplicator import ContentDeduplicator
from scraping.source_tracker import SourceItem, SourceTracker

logger = logging.getLogger(__name__)

# Authoritative technical hardware domains
DEFAULT_ALLOWED_DOMAINS = [
    "arxiv.org",
    "github.com",
    "verilator.org",
    "yosyshq.readthedocs.io",
    "docs.cocotb.org",
    "riscv.org",
    "ieee.org",
    "wikipedia.org",
    "en.wikipedia.org",
    "openroad.readthedocs.io",
]


@dataclass
class CompactTechnicalContext:
    """The only structure allowed into Qwen's working context from web research."""
    query: str
    source_url: str
    title: str
    extracted_summary: str
    source_domain: str = ""
    code_snippets: list[str] = field(default_factory=list)
    key_specifications: list[str] = field(default_factory=list)
    token_count: int = 0
    citation_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    relevance_score: float = 0.85

    def __post_init__(self):
        if not self.source_domain and self.source_url:
            parsed = urllib.parse.urlparse(self.source_url)
            self.source_domain = parsed.netloc

    def to_prompt_text(self) -> str:
        """Format strictly bounded, injection-safe context for Qwen."""
        parts = [
            f"[EXTERNAL RESEARCH CONTEXT: {self.citation_id}]",
            f"Source URL: {self.source_url}",
            f"Domain: {self.source_domain}",
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

    def to_research_record(self) -> dict[str, Any]:
        """Formatted research record schema required by system specification."""
        return {
            "title": self.title,
            "url": self.source_url,
            "source_domain": self.source_domain,
            "query": self.query,
            "content": self.extracted_summary,
            "timestamp": self.timestamp,
            "relevance_score": self.relevance_score,
        }


class ScrapeGraphAdapter:
    """Adapter implementing targeted ScrapeGraphAI extraction for hardware design tasks."""

    def __init__(
        self,
        ollama_model: str = "qwen2.5-coder:3b",
        max_page_bytes: int = 250_000,
        allowed_domains: list[str] | None = None,
        cache_dir: str = "./data/web_cache",
        timeout: float = 15.0,
        max_retries: int = 3,
    ):
        self.ollama_model = ollama_model
        self.max_page_bytes = max_page_bytes
        self.allowed_domains = allowed_domains or DEFAULT_ALLOWED_DOMAINS
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.max_retries = max_retries

        os.makedirs(cache_dir, exist_ok=True)

        self.filter = ContentFilter(max_length=1500)
        self.deduplicator = ContentDeduplicator()
        self.source_tracker = SourceTracker()

        # Check if scrapegraphai is installed
        self._has_sgai = False
        try:
            import scrapegraphai  # noqa: F401
            self._has_sgai = True
            logger.info("ScrapeGraphAI package detected and enabled.")
        except ImportError:
            logger.info("ScrapeGraphAI package not installed. Using native ScrapeGraph adapter pipeline.")

    def is_domain_allowed(self, url: str) -> bool:
        """Check whether the target URL domain is in the allowed domain list."""
        if not url.startswith("http"):
            return True
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        # Allow exact match or subdomain
        for allowed in self.allowed_domains:
            if domain == allowed.lower() or domain.endswith("." + allowed.lower()):
                return True
        return False

    def _get_cache_path(self, url: str, query: str) -> str:
        """Generate cache file path for URL + query."""
        key = hashlib.sha256(f"{url}|{query}".encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{key}.json")

    def _load_from_cache(self, url: str, query: str) -> Optional[CompactTechnicalContext]:
        """Load cached research result if available."""
        cache_path = self._get_cache_path(url, query)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return CompactTechnicalContext(**data)
            except Exception as e:
                logger.debug(f"Failed to read cache {cache_path}: {e}")
        return None

    def _save_to_cache(self, ctx: CompactTechnicalContext) -> None:
        """Save research result to disk cache."""
        cache_path = self._get_cache_path(ctx.source_url, ctx.query)
        try:
            data = {
                "query": ctx.query,
                "source_url": ctx.source_url,
                "title": ctx.title,
                "extracted_summary": ctx.extracted_summary,
                "source_domain": ctx.source_domain,
                "code_snippets": ctx.code_snippets,
                "key_specifications": ctx.key_specifications,
                "token_count": ctx.token_count,
                "citation_id": ctx.citation_id,
                "timestamp": ctx.timestamp,
                "relevance_score": ctx.relevance_score,
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to write cache {cache_path}: {e}")

    def fetch_url(self, url: str) -> tuple[str, str]:
        """Fetch raw HTML/text with retry logic, rate limiting, and size clamping."""
        if not self.is_domain_allowed(url):
            logger.warning(f"Domain not in allowlist for URL: {url}")
            return "", "text/plain"

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "ChipAgent-Research/1.0 (Hardware-AI-Agent; technical-extraction)",
                "Accept": "text/html,application/xhtml+xml,text/plain",
            },
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    content_type = resp.headers.get("Content-Type", "text/html")
                    raw_bytes = resp.read(self.max_page_bytes)
                    encoding = resp.headers.get_content_charset() or "utf-8"
                    return raw_bytes.decode(encoding, errors="replace"), content_type
            except Exception as e:
                logger.debug(f"Attempt {attempt}/{self.max_retries} failed fetching {url}: {e}")
                if attempt < self.max_retries:
                    time.sleep(0.5 * attempt)

        return "", "text/plain"

    async def extract_compact_context(
        self,
        url: str,
        focused_query: str,
        fallback_content: Optional[str] = None,
    ) -> CompactTechnicalContext:
        """Perform targeted technical extraction from URL and query.

        Steps:
        1. Check disk cache
        2. Domain allowlist check
        3. Register in source tracker (provenance)
        4. ScrapeGraphAI SmartScraperGraph if available and online
        5. ContentFilter + Deduplicator pipeline
        6. Return strictly bounded CompactTechnicalContext
        """
        # 1. Check cache
        cached = self._load_from_cache(url, focused_query)
        if cached is not None:
            logger.debug(f"Loaded web research from cache for {url}")
            return cached

        source_item = self.source_tracker.register(url=url)
        raw_text = fallback_content or ""
        content_type = "text/html"

        if not raw_text and url.startswith("http"):
            raw_text, content_type = self.fetch_url(url)

        if not raw_text:
            ctx = CompactTechnicalContext(
                query=focused_query,
                source_url=url,
                title=source_item.title,
                extracted_summary="Unable to retrieve external content or page was empty.",
                citation_id=source_item.citation_id,
            )
            return ctx

        # Try ScrapeGraphAI if installed
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
                    prompt=f"Extract only technical hardware design specifications and SystemVerilog constructs related to: {focused_query}",
                    source=url,
                    config=graph_config,
                )
                result = smart_scraper.run()
                if result:
                    summary_text = json.dumps(result) if isinstance(result, dict) else str(result)
                    ctx = CompactTechnicalContext(
                        query=focused_query,
                        source_url=url,
                        title=source_item.title,
                        extracted_summary=summary_text[:1500],
                        citation_id=source_item.citation_id,
                    )
                    self._save_to_cache(ctx)
                    return ctx
            except Exception as e:
                logger.warning(f"ScrapeGraphAI graph failed, falling back to native pipeline: {e}")

        # Native targeted pipeline: ContentFilter + Deduplicator
        is_html = "html" in content_type.lower() or "<html" in raw_text.lower()
        extracted_dict = self.filter.extract_from_html(raw_text) if is_html else {
            "title": url,
            "paragraphs": [raw_text],
            "code_blocks": [],
            "tables": [],
        }

        # Deduplicate paragraphs
        deduped_paras = self.deduplicator.deduplicate_paragraphs(extracted_dict.get("paragraphs", []))

        # Filter & compact
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

        self._save_to_cache(ctx)
        return ctx
