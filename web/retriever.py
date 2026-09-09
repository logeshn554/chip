"""
Content Retriever — fetches and extracts useful content from web URLs.

Handles HTML pages, PDF papers, and GitHub repos. Extracts
relevant text sections and optionally ingests into knowledge memory.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class ContentRetriever:
    """Fetches and processes web content for the agent.

    Extracts clean text from HTML pages, stripping navigation,
    headers, footers, and other non-content elements.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.timeout = self.config.get("timeout", 30)
        self.max_content_length = self.config.get("max_content_length", 50000)

    async def fetch(self, url: str) -> str:
        """Fetch and extract text content from a URL.

        Args:
            url: The URL to fetch

        Returns:
            Extracted text content
        """
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
            ) as client:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (compatible; ChipAgent/1.0; "
                            "+https://github.com/self-evolving-chip)"
                        ),
                    },
                )
                response.raise_for_status()
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return ""

        content_type = response.headers.get("content-type", "")

        if "text/html" in content_type:
            return self._extract_html(response.text)
        elif "application/pdf" in content_type:
            return self._extract_pdf_placeholder(url)
        else:
            return response.text[:self.max_content_length]

    async def fetch_multiple(self, urls: list[str]) -> list[dict[str, str]]:
        """Fetch content from multiple URLs concurrently.

        Args:
            urls: List of URLs to fetch

        Returns:
            List of {"url": ..., "content": ...} dicts
        """
        import asyncio

        tasks = [self.fetch(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [
            {
                "url": url,
                "content": content if isinstance(content, str) else f"Error: {content}",
            }
            for url, content in zip(urls, results)
        ]

    # ── Content Extraction ───────────────────────────────────────────

    def _extract_html(self, html: str) -> str:
        """Extract clean text from HTML content.

        Strips navigation, scripts, styles, and non-content elements.
        Preserves code blocks and technical content.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Remove non-content elements
        for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Remove elements by common class/id patterns
        for selector in [
            ".nav", ".navbar", ".sidebar", ".footer", ".menu",
            ".advertisement", ".ads", ".cookie", ".popup",
            "#nav", "#footer", "#sidebar", "#menu",
        ]:
            for el in soup.select(selector):
                el.decompose()

        # Extract text, preserving structure
        text_parts = []

        # Get title
        title = soup.find("title")
        if title:
            text_parts.append(f"# {title.get_text().strip()}\n")

        # Get main content
        main = soup.find("main") or soup.find("article") or soup.find("body")
        if main:
            for element in main.find_all(["h1", "h2", "h3", "h4", "p", "pre", "code", "li", "td"]):
                tag_name = element.name
                text = element.get_text().strip()

                if not text:
                    continue

                if tag_name in ("h1", "h2", "h3", "h4"):
                    level = int(tag_name[1])
                    text_parts.append(f"\n{'#' * level} {text}\n")
                elif tag_name in ("pre", "code"):
                    text_parts.append(f"\n```\n{text}\n```\n")
                elif tag_name == "li":
                    text_parts.append(f"- {text}")
                else:
                    text_parts.append(text)

        content = "\n".join(text_parts)

        # Clean up excessive whitespace
        content = re.sub(r"\n{3,}", "\n\n", content)

        return content[:self.max_content_length]

    @staticmethod
    def _extract_pdf_placeholder(url: str) -> str:
        """Placeholder for PDF extraction.

        Full PDF extraction would require PyPDF2 or pdfplumber.
        For now, return a note about the PDF.
        """
        return f"[PDF document at {url} — full extraction requires PyPDF2]"

    # ── GitHub-specific extraction ───────────────────────────────────

    async def fetch_github_file(self, repo_url: str, filepath: str) -> str:
        """Fetch a specific file from a GitHub repository.

        Converts GitHub URLs to raw content URLs.

        Args:
            repo_url: GitHub repository URL (e.g., "https://github.com/user/repo")
            filepath: Path within the repo (e.g., "src/module.sv")

        Returns:
            File content
        """
        # Convert to raw URL
        raw_url = repo_url.replace("github.com", "raw.githubusercontent.com")
        raw_url = f"{raw_url}/main/{filepath}"

        return await self.fetch(raw_url)

    async def fetch_github_tree(self, repo_url: str) -> list[str]:
        """List files in a GitHub repository using the API.

        Args:
            repo_url: GitHub repository URL

        Returns:
            List of file paths in the repo
        """
        # Extract owner/repo from URL
        match = re.match(r"https://github\.com/([^/]+)/([^/]+)", repo_url)
        if not match:
            logger.error(f"Invalid GitHub URL: {repo_url}")
            return []

        owner, repo = match.groups()
        api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(api_url)
                response.raise_for_status()
                data = response.json()
                return [
                    item["path"]
                    for item in data.get("tree", [])
                    if item["type"] == "blob"
                ]
        except httpx.HTTPError as e:
            logger.error(f"GitHub API error: {e}")
            return []
