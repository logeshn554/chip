"""
Memory search tool with strict validation for RETRIEVE_MEMORY action.
Queries KnowledgeStore, ExperienceStore, or DesignStore.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

from memory.knowledge_store import KnowledgeStore
from memory.experience_store import ExperienceStore
from memory.design_store import DesignStore

logger = logging.getLogger(__name__)


class MemorySearchArgs(BaseModel):
    """Strictly validated parameters for RETRIEVE_MEMORY."""
    query: str = Field(..., min_length=2, max_length=300, description="Semantic search query")
    memory_type: Literal["knowledge", "experience", "design"] = Field(
        default="knowledge", description="Target memory tier"
    )
    n_results: int = Field(default=2, ge=1, le=5)


class MemorySearchTool:
    """Tool wrapper providing strict semantic search over memory subsystems."""

    def __init__(
        self,
        knowledge_store: Optional[KnowledgeStore] = None,
        experience_store: Optional[ExperienceStore] = None,
        design_store: Optional[DesignStore] = None,
    ):
        self.knowledge_store = knowledge_store or KnowledgeStore()
        self.experience_store = experience_store or ExperienceStore()
        self.design_store = design_store or DesignStore()

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute validated memory query."""
        try:
            args = MemorySearchArgs(**kwargs)
        except Exception as e:
            return {"status": "error", "message": f"Invalid arguments for RETRIEVE_MEMORY: {e}"}

        if args.memory_type == "knowledge":
            results = self.knowledge_store.query(args.query, n_results=args.n_results)
            formatted = []
            for r in results:
                formatted.append({
                    "title": r.get("title", ""),
                    "source_url": r.get("source_url", ""),
                    "content": r.get("content", ""),
                })
            return {
                "status": "success",
                "memory_type": "knowledge",
                "matches": formatted,
                "count": len(formatted),
            }

        elif args.memory_type == "experience":
            results = self.experience_store.retrieve_similar_failures(
                error_text=args.query, n_results=args.n_results
            )
            return {
                "status": "success",
                "memory_type": "experience",
                "matches": results,
                "count": len(results),
            }

        elif args.memory_type == "design":
            design = self.design_store.get_design(module_name=args.query)
            if design:
                return {
                    "status": "success",
                    "memory_type": "design",
                    "design_id": design.design_id,
                    "version": design.version,
                    "reward": design.reward,
                    "summary": f"Module {design.module_name} v{design.version} with reward {design.reward}",
                }
            return {"status": "success", "memory_type": "design", "matches": [], "message": "No design found"}

        return {"status": "error", "message": f"Unknown memory type: {args.memory_type}"}
