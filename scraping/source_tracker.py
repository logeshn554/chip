"""
Source tracking and citation preservation for web research.

Every extracted knowledge item must preserve its provenance:
source URL, title, timestamp, and document domain.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class SourceItem:
    """Provenance metadata for an external technical document."""
    url: str
    title: str = ""
    domain: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    doc_type: str = "web_document"
    snippet: str = ""
    citation_id: str = ""

    def __post_init__(self):
        if not self.domain and self.url:
            parsed = urlparse(self.url)
            self.domain = parsed.netloc or "local"
        if not self.citation_id and self.url:
            self.citation_id = f"[{self.domain}:{hash(self.url) % 10000:04d}]"


class SourceTracker:
    """Tracks sources accessed across agent episodes to maintain citations and prevent duplicate queries."""

    def __init__(self):
        self._sources: dict[str, SourceItem] = {}

    def register(self, url: str, title: str = "", snippet: str = "", doc_type: str = "web_document") -> SourceItem:
        """Register or retrieve an external source."""
        if url in self._sources:
            item = self._sources[url]
            if title and not item.title:
                item.title = title
            if snippet and not item.snippet:
                item.snippet = snippet
            return item

        item = SourceItem(
            url=url,
            title=title or url,
            snippet=snippet[:250],
            doc_type=doc_type,
        )
        self._sources[url] = item
        logger.debug(f"Registered source: {item.citation_id} -> {url}")
        return item

    def get(self, url: str) -> Optional[SourceItem]:
        return self._sources.get(url)

    def format_citations(self) -> str:
        """Format markdown citation list for LLM context grounding."""
        if not self._sources:
            return "No external sources cited."
        lines = ["### Referenced External Sources:"]
        for item in self._sources.values():
            lines.append(f"- **{item.citation_id}** [{item.title}]({item.url}) (Accessed: {item.timestamp[:10]})")
        return "\n".join(lines)

    def to_dict_list(self) -> list[dict]:
        return [asdict(item) for item in self._sources.values()]
