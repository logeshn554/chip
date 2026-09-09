"""
Memory System — three-tier external memory for the hardware agent.

Provides unified access to:
- Knowledge Memory (technical documentation RAG)
- Experience Memory (past attempts, errors, fixes)
- Design Memory (successful RTL, testbenches, synthesis results)
"""

from __future__ import annotations

import logging
from typing import Any

from agent.schemas import Document
from memory.knowledge import KnowledgeMemory
from memory.experience import ExperienceMemory
from memory.design_memory import DesignMemory

logger = logging.getLogger(__name__)


class MemorySystem:
    """Unified facade over the three-tier memory system.

    This class provides a single interface for the agent to interact
    with all memory types, handling routing and result merging.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        chroma_path = config.get("chroma_path", "./data/chroma")
        embedding_model = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")

        self.knowledge = KnowledgeMemory(
            chroma_path=chroma_path,
            embedding_model=embedding_model,
            config=config.get("knowledge", {}),
        )
        self.experience = ExperienceMemory(
            chroma_path=chroma_path,
            embedding_model=embedding_model,
            config=config.get("experience", {}),
        )
        self.design = DesignMemory(
            chroma_path=chroma_path,
            embedding_model=embedding_model,
            config=config.get("design", {}),
        )

        logger.info("MemorySystem initialized (knowledge + experience + design)")

    async def query(
        self,
        question: str,
        n: int = 5,
        memory_type: str = "all",
    ) -> list[Document]:
        """Query memory for relevant documents.

        Args:
            question: The query string
            n: Max results per memory type
            memory_type: "all", "knowledge", "experience", or "design"

        Returns:
            List of Document objects sorted by relevance score
        """
        docs: list[Document] = []

        if memory_type in ("all", "knowledge"):
            docs.extend(self.knowledge.query(question, n=n))
        if memory_type in ("all", "experience"):
            docs.extend(self.experience.query(question, n=n))
        if memory_type in ("all", "design"):
            docs.extend(self.design.query(question, n=n))

        # Sort by relevance score (higher is better)
        docs.sort(key=lambda d: d.score, reverse=True)

        # Deduplicate by content similarity
        seen = set()
        unique_docs = []
        for doc in docs:
            key = doc.content[:100]
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        return unique_docs[:n * 2]  # Return at most 2x the per-type limit

    async def record_experience(
        self,
        task: str,
        action: str,
        result: str,
        errors: list[str],
        success: bool,
    ) -> None:
        """Record an experience (attempt, error, fix) into experience memory."""
        self.experience.record(
            task=task,
            action=action,
            result=result,
            errors=errors,
            success=success,
        )

    async def save_design(
        self,
        name: str,
        code: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save a design to design memory."""
        self.design.save(name=name, code=code, metadata=metadata or {})

    async def find_similar_errors(self, error: str, n: int = 5) -> list[Document]:
        """Find similar past errors and their fixes."""
        return self.experience.find_similar_errors(error, n=n)

    async def find_similar_designs(self, description: str, n: int = 3) -> list[Document]:
        """Find similar past designs."""
        return self.design.query(description, n=n)


__all__ = ["MemorySystem", "KnowledgeMemory", "ExperienceMemory", "DesignMemory"]
