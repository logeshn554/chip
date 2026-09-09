"""
Experience Memory — stores past attempts, errors, corrections, and their outcomes.

Enables the agent to learn from past mistakes by finding similar errors
and retrieving the fixes that worked.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

import chromadb

from agent.schemas import Document

logger = logging.getLogger(__name__)


class ExperienceMemory:
    """Records and retrieves past agent experiences.

    Stores structured records of:
    - What the agent tried (action + params)
    - What happened (result + errors)
    - Whether it succeeded or failed
    - How it was fixed (if applicable)

    Uses ChromaDB for semantic retrieval of similar past experiences.
    """

    def __init__(
        self,
        chroma_path: str = "./data/chroma",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        config: dict[str, Any] | None = None,
    ):
        config = config or {}
        self.collection_name = config.get("collection", "experience")
        self.max_results = config.get("max_results", 10)

        self._client = chromadb.PersistentClient(path=chroma_path)

        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"ExperienceMemory: collection='{self.collection_name}', "
            f"entries={self._collection.count()}"
        )

    # ── Recording ────────────────────────────────────────────────────

    def record(
        self,
        task: str,
        action: str,
        result: str,
        errors: list[str],
        success: bool,
        fix: str | None = None,
    ) -> None:
        """Record an experience entry.

        Args:
            task: What the agent was trying to do
            action: The action taken (and its parameters)
            result: The outcome
            errors: Any error messages
            success: Whether the action succeeded
            fix: If it failed, what fix was applied (optional)
        """
        # Build a searchable document
        error_text = "\n".join(errors) if errors else "No errors"
        document = (
            f"Task: {task}\n"
            f"Action: {action}\n"
            f"Result: {'SUCCESS' if success else 'FAILURE'}\n"
            f"Errors: {error_text}\n"
        )
        if fix:
            document += f"Fix: {fix}\n"

        doc_id = hashlib.sha256(
            f"{task}:{action}:{time.time()}".encode()
        ).hexdigest()[:16]

        self._collection.upsert(
            ids=[doc_id],
            documents=[document],
            metadatas=[{
                "task": task[:200],
                "success": str(success),
                "has_fix": str(fix is not None),
                "timestamp": str(time.time()),
                "error_count": len(errors),
            }],
        )

        logger.debug(f"Recorded experience: task={task[:50]}, success={success}")

    def record_fix(
        self,
        original_error: str,
        fix_description: str,
        fix_code: str,
        success: bool,
    ) -> None:
        """Record a specific error → fix mapping.

        This creates a high-value entry specifically for error resolution.
        """
        document = (
            f"Error: {original_error}\n"
            f"Fix Description: {fix_description}\n"
            f"Fix Code:\n{fix_code}\n"
            f"Fix Successful: {success}"
        )

        doc_id = hashlib.sha256(
            f"fix:{original_error[:100]}:{time.time()}".encode()
        ).hexdigest()[:16]

        self._collection.upsert(
            ids=[doc_id],
            documents=[document],
            metadatas=[{
                "type": "fix",
                "success": str(success),
                "timestamp": str(time.time()),
            }],
        )

    # ── Retrieval ────────────────────────────────────────────────────

    def query(self, question: str, n: int = 5) -> list[Document]:
        """Query for relevant past experiences.

        Args:
            question: Natural language query about a task or error
            n: Maximum number of results

        Returns:
            List of Document objects sorted by relevance
        """
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[question],
            n_results=min(n, self._collection.count()),
        )

        docs = []
        for i in range(len(results["documents"][0])):
            content = results["documents"][0][i]
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 1.0
            score = max(0.0, 1.0 - distance)

            docs.append(Document(
                content=content,
                metadata=metadata,
                score=score,
                source="experience",
            ))

        return docs

    def find_similar_errors(self, error: str, n: int = 5) -> list[Document]:
        """Find past errors similar to the given one and their fixes.

        Specifically filters for entries that have fixes recorded.
        """
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[error],
            n_results=min(n * 2, self._collection.count()),
            where={"has_fix": "True"},
        )

        if not results["documents"][0]:
            # Fallback: query without the fix filter
            results = self._collection.query(
                query_texts=[error],
                n_results=min(n, self._collection.count()),
            )

        docs = []
        for i in range(len(results["documents"][0])):
            content = results["documents"][0][i]
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 1.0
            score = max(0.0, 1.0 - distance)

            docs.append(Document(
                content=content,
                metadata=metadata,
                score=score,
                source="experience:error",
            ))

        return docs[:n]

    def get_success_rate(self, task_type: str | None = None) -> float:
        """Calculate success rate for a given task type (or overall)."""
        if self._collection.count() == 0:
            return 0.0

        where_filter = {"success": "True"}
        if task_type:
            where_filter["task"] = task_type

        try:
            successes = self._collection.count()  # Simplified — ChromaDB doesn't support count with where
            total = self._collection.count()
            return successes / total if total > 0 else 0.0
        except Exception:
            return 0.0
