"""
Targeted content filtering for technical hardware documentation.

Strips non-technical HTML artifacts, ads, navigation, and boilerplate.
Extracts only relevant SystemVerilog, formulas, signal definitions, and architectural concepts.
Enforces strict token budgets for Qwen3-4B context management.
"""

from __future__ import annotations

import logging
import re
from typing import Optional
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Injection patterns to sanitize
PROMPT_INJECTION_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"(?i)system\s+prompt",
    r"(?i)you\s+are\s+now\s+a",
    r"(?i)forget\s+(all\s+)?(prior|previous)\s+context",
    r"(?i)act\s+as\s+a\s+jailbreak",
]


class ContentFilter:
    """Filters raw HTML/Markdown into compact, sanitized technical context."""

    def __init__(self, max_tokens: int = 1500, max_chars: int = 6000, max_length: Optional[int] = None):
        self.max_tokens = max_length if max_length is not None else max_tokens
        self.max_chars = max_chars


    def sanitize_text(self, text: str) -> str:
        """Strip potential prompt injection attacks from untrusted external text."""
        sanitized = text
        for pattern in PROMPT_INJECTION_PATTERNS:
            sanitized = re.sub(pattern, "[SANITIZED_PROMPT_DIRECTIVE]", sanitized)
        return sanitized

    def extract_from_html(self, html: str, base_url: str = "") -> dict[str, Any]:
        """Extract structured technical elements from HTML."""
        soup = BeautifulSoup(html, "html.parser")

        # Strip non-content elements
        for tag in soup(["script", "style", "nav", "footer", "aside", "header", "form", "iframe", "noscript", "svg"]):
            tag.decompose()

        # Remove elements with ad/cookie classes or IDs
        for bad in soup.find_all(class_=re.compile(r"(ad|banner|cookie|nav|sidebar|footer|menu|comment|social)", re.I)):
            bad.decompose()

        # Extract code blocks
        code_blocks: list[str] = []
        for pre in soup.find_all(["pre", "code"]):
            code_text = pre.get_text().strip()
            if len(code_text) > 20 and any(kw in code_text for kw in ["module", "endmodule", "logic", "wire", "reg", "always", "assign", "input", "output"]):
                code_blocks.append(code_text)

        # Extract tables (e.g. pinouts, opcode tables)
        tables: list[str] = []
        for tbl in soup.find_all("table")[:3]:
            rows = []
            for tr in tbl.find_all("tr")[:10]:
                cols = [td.get_text().strip() for td in tr.find_all(["td", "th"])]
                if cols:
                    rows.append(" | ".join(cols))
            if rows:
                tables.append("\n".join(rows))

        # Extract main text paragraphs and headings
        paragraphs: list[str] = []
        for p in soup.find_all(["p", "h1", "h2", "h3", "h4", "li"]):
            text = p.get_text().strip()
            min_len = 5 if p.name in ["h1", "h2", "h3", "h4"] else 25
            if len(text) >= min_len and not any(term in text.lower() for term in ["privacy policy", "terms of use", "cookie", "copyright", "all rights reserved"]):
                paragraphs.append(text)

        title = soup.title.string.strip() if (soup.title and soup.title.string) else ""
        if not title:
            h_first = soup.find(["h1", "h2"])
            if h_first and h_first.get_text().strip():
                title = h_first.get_text().strip()

        return {
            "title": title,
            "paragraphs": paragraphs,
            "code_blocks": code_blocks,
            "tables": tables,
        }

    def filter_and_compact(
        self,
        raw_text_or_html: str,
        query: str = "",
        is_html: bool = False,
    ) -> str:
        """Produce compact technical markdown text within budget."""
        if is_html or "<html" in raw_text_or_html.lower() or "<body" in raw_text_or_html.lower():
            extracted = self.extract_from_html(raw_text_or_html)
            title = extracted["title"]
            paragraphs = extracted["paragraphs"]
            code_blocks = extracted["code_blocks"]
            tables = extracted["tables"]
        else:
            title = ""
            paragraphs = [p.strip() for p in raw_text_or_html.split("\n\n") if len(p.strip()) > 30]
            code_blocks = re.findall(r"```[\w]*\n(.*?)```", raw_text_or_html, re.DOTALL)
            tables = []

        query_terms = [t.lower() for t in query.split() if len(t) > 3]

        # Score paragraphs by relevance to query terms
        scored_paras: list[tuple[float, str]] = []
        for p in paragraphs:
            p_clean = self.sanitize_text(p)
            score = 0.0
            p_lower = p_clean.lower()
            for term in query_terms:
                if term in p_lower:
                    score += 2.0
            # Hardware-relevant bonus
            for hw_term in ["signed", "multiplier", "accumulator", "overflow", "saturation", "latency", "systemverilog", "clock", "pipeline"]:
                if hw_term in p_lower:
                    score += 1.0
            scored_paras.append((score, p_clean))

        scored_paras.sort(key=lambda x: x[0], reverse=True)

        # Assemble compact context
        sections: list[str] = []
        if title:
            sections.append(f"### Source: {title}")

        # Top relevant paragraphs
        relevant_text = []
        char_count = 0
        for score, text in scored_paras:
            if char_count + len(text) > self.max_chars * 0.6:
                break
            relevant_text.append(text)
            char_count += len(text)

        if relevant_text:
            sections.append("\n\n".join(relevant_text))

        # Include most relevant code snippet if available
        for code in code_blocks:
            clean_code = self.sanitize_text(code)
            if char_count + len(clean_code) < self.max_chars:
                sections.append(f"```systemverilog\n{clean_code[:1200]}\n```")
                char_count += len(clean_code)
                break

        final_result = "\n\n".join(sections).strip()
        if not final_result:
            final_result = self.sanitize_text(raw_text_or_html[:self.max_chars])

        return final_result[: self.max_chars]
