"""
Content deduplication for web research and extracted technical documentation.

Prevents duplicate or near-duplicate technical paragraphs from polluting the LLM context.
Uses SHA-256 for exact match and token set Jaccard similarity for near-duplicate filtering.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable


class ContentDeduplicator:
    """Deduplicates text chunks and paragraphs."""

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self._seen_hashes: set[str] = set()
        self._seen_token_sets: list[set[str]] = []

    def _hash_text(self, text: str) -> str:
        """Normalized hash for exact deduplication."""
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _tokenize(self, text: str) -> set[str]:
        """Extract word shingles for fuzzy similarity."""
        words = re.findall(r"\b[a-zA-Z0-9_]{3,}\b", text.lower())
        return set(words)

    def _jaccard(self, set_a: set[str], set_b: set[str]) -> float:
        """Compute Jaccard similarity between two token sets."""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    def is_duplicate(self, text: str) -> bool:
        """Check if text is an exact or near duplicate of previously seen content."""
        if len(text.strip()) < 20:
            return False

        h = self._hash_text(text)
        if h in self._seen_hashes:
            return True

        tokens = self._tokenize(text)
        if len(tokens) >= 5:
            for seen_set in self._seen_token_sets:
                sim = self._jaccard(tokens, seen_set)
                if sim >= self.similarity_threshold:
                    return True

        return False

    def add(self, text: str) -> None:
        """Record text as seen."""
        h = self._hash_text(text)
        self._seen_hashes.add(h)
        tokens = self._tokenize(text)
        if len(tokens) >= 5:
            self._seen_token_sets.append(tokens)

    def deduplicate_paragraphs(self, paragraphs: Iterable[str]) -> list[str]:
        """Filter out duplicates from a collection of paragraphs."""
        unique = []
        for p in paragraphs:
            if not self.is_duplicate(p):
                self.add(p)
                unique.append(p)
        return unique

    def reset(self) -> None:
        self._seen_hashes.clear()
        self._seen_token_sets.clear()
