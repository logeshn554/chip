"""
Web Search — controlled technical information retrieval.

Provides a unified search interface with pluggable backends
(DuckDuckGo, Serper, Tavily). Includes domain filtering,
rate limiting, and result caching.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any

from agent.schemas import SearchResult

logger = logging.getLogger(__name__)


class WebSearcher:
    """Web search interface for the hardware agent.

    Treats the web as a knowledge tool — restricts searches to
    technical domains and caches results to avoid redundant queries.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.backend = config.get("search_backend", "duckduckgo")
        self.max_results = config.get("max_results", 10)
        self.allowed_domains = config.get("allowed_domains", [])
        self.rate_limit_s = config.get("rate_limit_seconds", 2.0)
        self.cache_dir = config.get("cache_dir", "./data/web_cache")

        self._last_search_time = 0.0

        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"WebSearcher: backend={self.backend}, domains={len(self.allowed_domains)}")

    async def search(
        self,
        query: str,
        max_results: int | None = None,
        domains: list[str] | None = None,
    ) -> list[SearchResult]:
        """Search the web for technical information.

        Args:
            query: Search query
            max_results: Override default max results
            domains: Override allowed domains filter

        Returns:
            List of SearchResult objects
        """
        max_results = max_results or self.max_results
        domains = domains or self.allowed_domains

        # Check cache first
        cached = self._check_cache(query)
        if cached:
            logger.debug(f"Cache hit for query: {query[:50]}")
            return cached

        # Rate limiting
        await self._rate_limit()

        # Execute search
        if self.backend == "duckduckgo":
            results = await self._search_duckduckgo(query, max_results)
        elif self.backend == "serper":
            results = await self._search_serper(query, max_results)
        elif self.backend == "tavily":
            results = await self._search_tavily(query, max_results)
        else:
            logger.error(f"Unknown search backend: {self.backend}")
            return []

        # Filter by allowed domains
        if domains:
            results = [r for r in results if self._is_allowed_domain(r.url, domains)]

        # Cache results
        self._save_cache(query, results)

        logger.info(f"Search '{query[:50]}' returned {len(results)} results")
        return results[:max_results]

    # ── Backend Implementations ──────────────────────────────────────

    async def _search_duckduckgo(self, query: str, max_results: int) -> list[SearchResult]:
        """Search using DuckDuckGo (free, no API key)."""
        import asyncio

        def _do_search():
            try:
                from duckduckgo_search import DDGS
                with DDGS() as ddgs:
                    raw = list(ddgs.text(query, max_results=max_results))
                    return [
                        SearchResult(
                            title=r.get("title", ""),
                            url=r.get("href", r.get("link", "")),
                            snippet=r.get("body", r.get("snippet", "")),
                        )
                        for r in raw
                    ]
            except Exception as e:
                logger.error(f"DuckDuckGo search failed: {e}")
                return []

        return await asyncio.get_event_loop().run_in_executor(None, _do_search)

    async def _search_serper(self, query: str, max_results: int) -> list[SearchResult]:
        """Search using Serper.dev Google Search API."""
        import httpx

        api_key = os.environ.get("SERPER_API_KEY", "")
        if not api_key:
            logger.error("SERPER_API_KEY not set")
            return []

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://google.serper.dev/search",
                json={"q": query, "num": max_results},
                headers={"X-API-KEY": api_key},
            )

            if response.status_code != 200:
                logger.error(f"Serper API error: {response.status_code}")
                return []

            data = response.json()
            return [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("link", ""),
                    snippet=r.get("snippet", ""),
                )
                for r in data.get("organic", [])
            ]

    async def _search_tavily(self, query: str, max_results: int) -> list[SearchResult]:
        """Search using Tavily AI Search API."""
        import httpx

        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            logger.error("TAVILY_API_KEY not set")
            return []

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "advanced",
                },
            )

            if response.status_code != 200:
                logger.error(f"Tavily API error: {response.status_code}")
                return []

            data = response.json()
            return [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("content", ""),
                )
                for r in data.get("results", [])
            ]

    # ── Helpers ──────────────────────────────────────────────────────

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between searches."""
        import asyncio

        elapsed = time.time() - self._last_search_time
        if elapsed < self.rate_limit_s:
            await asyncio.sleep(self.rate_limit_s - elapsed)
        self._last_search_time = time.time()

    @staticmethod
    def _is_allowed_domain(url: str, domains: list[str]) -> bool:
        """Check if a URL belongs to an allowed domain."""
        if not domains:
            return True
        return any(domain in url for domain in domains)

    def _cache_key(self, query: str) -> str:
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]

    def _check_cache(self, query: str) -> list[SearchResult] | None:
        """Check if results are cached (TTL: 24 hours)."""
        cache_file = os.path.join(self.cache_dir, f"{self._cache_key(query)}.json")
        if not os.path.exists(cache_file):
            return None

        try:
            with open(cache_file, "r") as f:
                data = json.load(f)

            # Check TTL (24 hours)
            if time.time() - data.get("timestamp", 0) > 86400:
                return None

            return [
                SearchResult(
                    title=r["title"],
                    url=r["url"],
                    snippet=r["snippet"],
                    content=r.get("content", ""),
                )
                for r in data.get("results", [])
            ]
        except (json.JSONDecodeError, KeyError):
            return None

    def _save_cache(self, query: str, results: list[SearchResult]) -> None:
        """Save search results to cache."""
        cache_file = os.path.join(self.cache_dir, f"{self._cache_key(query)}.json")
        data = {
            "query": query,
            "timestamp": time.time(),
            "results": [
                {"title": r.title, "url": r.url, "snippet": r.snippet, "content": r.content}
                for r in results
            ],
        }
        with open(cache_file, "w") as f:
            json.dump(data, f)
