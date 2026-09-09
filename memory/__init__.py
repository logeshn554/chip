"""
Memory System — Unified three-tier external memory + trajectory store.

Provides unified access to:
1. Knowledge Memory (Technical documentation, SystemVerilog standards, RISC-V, web extracts)
2. Experience Memory (Past attempts, errors, fixes, outcomes, trajectory IDs)
3. Design Memory (Versioned RTL, testbenches, synthesis metrics, lineage)
4. Trajectory Store (Complete JSONL episode histories)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.schemas import Document

from memory.knowledge import KnowledgeMemory
from memory.knowledge_store import KnowledgeStore, KnowledgeChunk
from memory.experience import ExperienceMemory
from memory.experience_store import ExperienceStore, Experience
from memory.design_memory import DesignMemory
from memory.design_store import DesignStore, DesignRecord
from memory.trajectory_store import TrajectoryStore, TrajectoryStep, TrajectoryRecord

logger = logging.getLogger(__name__)


class MemorySystem:
    """Unified facade over the four memory systems.

    Provides a single clean interface for the agent to query and update
    knowledge, experience, design catalogs, and trajectories.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.config = config
        chroma_path = config.get("chroma_path", "./data/chroma")
        embedding_model = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")

        # Initialize the 3 memory tiers
        self.knowledge = KnowledgeMemory(
            chroma_path=os.path.join(chroma_path, "knowledge") if not chroma_path.endswith("knowledge") else chroma_path,
            embedding_model=embedding_model,
            config=config.get("knowledge", {}),
        )
        self.experience = ExperienceMemory(
            chroma_path=os.path.join(chroma_path, "experience") if not chroma_path.endswith("experience") else chroma_path,
            embedding_model=embedding_model,
            config=config.get("experience", {}),
        )
        self.design = DesignMemory(
            chroma_path=os.path.join(chroma_path, "design") if not chroma_path.endswith("design") else chroma_path,
            embedding_model=embedding_model,
            config=config.get("design", {}),
        )
        self.trajectory = TrajectoryStore(
            store_dir=config.get("trajectory", {}).get("store_dir", "./trajectories")
        )

        logger.info("MemorySystem initialized (Knowledge + Experience + Design + Trajectory)")

    async def query(
        self,
        question: str,
        n: int = 5,
        memory_type: str = "all",
    ) -> list[Document]:
        """Query memory across one or all memory tiers.

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

        # Sort by score descending
        docs.sort(key=lambda d: d.score, reverse=True)

        # Deduplicate
        seen = set()
        unique_docs = []
        for doc in docs:
            snippet = doc.content[:100]
            if snippet not in seen:
                seen.add(snippet)
                unique_docs.append(doc)

        return unique_docs[: n * 2]

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


__all__ = [
    "MemorySystem",
    "KnowledgeMemory",
    "KnowledgeStore",
    "KnowledgeChunk",
    "ExperienceMemory",
    "ExperienceStore",
    "Experience",
    "DesignMemory",
    "DesignStore",
    "DesignRecord",
    "TrajectoryStore",
    "TrajectoryRecord",
    "TrajectoryStep",
]
