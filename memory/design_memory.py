"""
Design Memory — stores successful RTL designs, testbenches, and synthesis results.

Maintains a versioned library of hardware designs that the agent
can reference when creating new designs or optimizing existing ones.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any

import chromadb

from agent.schemas import Document

logger = logging.getLogger(__name__)


class DesignMemory:
    """Stores and retrieves hardware design artifacts.

    Combines ChromaDB for semantic search with filesystem storage
    for the actual design files. Optionally backed by Git for versioning.
    """

    def __init__(
        self,
        chroma_path: str = "./data/chroma",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        config: dict[str, Any] | None = None,
    ):
        config = config or {}
        self.collection_name = config.get("collection", "designs")
        self.designs_dir = config.get("designs_dir", "./designs")

        # Ensure designs directory exists
        os.makedirs(self.designs_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(path=chroma_path)

        try:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            self._embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
            self._embedding_fn(["test"])
        except Exception as e:
            logger.warning(f"SentenceTransformer embedding unavailable ({e}). Using offline deterministic embedding.")
            class DeterministicEmbeddingFunction:
                def __init__(self, dim: int = 384):
                    self.dim = dim
                def __call__(self, input: list[str]) -> list[list[float]]:
                    embeddings = []
                    for text in input:
                        vec = [0.0] * self.dim
                        words = text.lower().split()
                        for i, w in enumerate(words):
                            h = abs(hash(w)) % self.dim
                            vec[h] += 1.0 / (1.0 + i * 0.05)
                        norm = sum(x * x for x in vec) ** 0.5 or 1.0
                        embeddings.append([x / norm for x in vec])
                    return embeddings
            self._embedding_fn = DeterministicEmbeddingFunction()

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"DesignMemory: collection='{self.collection_name}', "
            f"designs={self._collection.count()}"
        )

    # ── Storage ──────────────────────────────────────────────────────

    def save(
        self,
        name: str,
        code: str,
        testbench: str | None = None,
        synthesis_results: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save a design to memory.

        Args:
            name: Design name (e.g., "alu_4bit")
            code: SystemVerilog source code
            testbench: Optional cocotb testbench code
            synthesis_results: Optional synthesis metrics
            metadata: Optional additional metadata

        Returns:
            Design ID
        """
        metadata = metadata or {}
        design_id = hashlib.sha256(
            f"{name}:{time.time()}".encode()
        ).hexdigest()[:12]

        # Save files to disk
        design_dir = os.path.join(self.designs_dir, f"{name}_{design_id}")
        os.makedirs(design_dir, exist_ok=True)

        rtl_path = os.path.join(design_dir, f"{name}.sv")
        with open(rtl_path, "w", encoding="utf-8") as f:
            f.write(code)

        if testbench:
            tb_path = os.path.join(design_dir, f"test_{name}.py")
            with open(tb_path, "w", encoding="utf-8") as f:
                f.write(testbench)

        if synthesis_results:
            synth_path = os.path.join(design_dir, "synthesis.json")
            with open(synth_path, "w", encoding="utf-8") as f:
                json.dump(synthesis_results, f, indent=2)

        # Build searchable document for vector DB
        description = (
            f"Design: {name}\n"
            f"Code:\n{code[:2000]}\n"
        )
        if synthesis_results:
            description += f"Synthesis: {json.dumps(synthesis_results)}\n"

        # Store in ChromaDB
        self._collection.upsert(
            ids=[design_id],
            documents=[description],
            metadatas=[{
                "name": name,
                "design_id": design_id,
                "design_dir": design_dir,
                "rtl_path": rtl_path,
                "has_testbench": str(testbench is not None),
                "has_synthesis": str(synthesis_results is not None),
                "timestamp": str(time.time()),
                **{k: str(v)[:200] for k, v in metadata.items()},
            }],
        )

        logger.info(f"Saved design '{name}' as {design_id} to {design_dir}")
        return design_id

    # ── Retrieval ────────────────────────────────────────────────────

    def query(self, description: str, n: int = 3) -> list[Document]:
        """Find designs similar to a description.

        Args:
            description: Natural language description of desired design
            n: Maximum number of results

        Returns:
            List of Document objects with design info and similarity scores
        """
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[description],
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
                source=f"design:{metadata.get('name', 'unknown')}",
            ))

        return docs

    def get_design_code(self, design_id: str) -> str | None:
        """Retrieve the full RTL code for a design by ID."""
        results = self._collection.get(ids=[design_id])

        if not results["metadatas"]:
            return None

        rtl_path = results["metadatas"][0].get("rtl_path", "")
        if rtl_path and os.path.exists(rtl_path):
            with open(rtl_path, "r", encoding="utf-8") as f:
                return f.read()

        return None

    def get_design_testbench(self, design_id: str) -> str | None:
        """Retrieve the testbench code for a design by ID."""
        results = self._collection.get(ids=[design_id])

        if not results["metadatas"]:
            return None

        design_dir = results["metadatas"][0].get("design_dir", "")
        name = results["metadatas"][0].get("name", "")
        tb_path = os.path.join(design_dir, f"test_{name}.py")

        if os.path.exists(tb_path):
            with open(tb_path, "r", encoding="utf-8") as f:
                return f.read()

        return None

    def list_designs(self) -> list[dict[str, Any]]:
        """List all stored designs with their metadata."""
        if self._collection.count() == 0:
            return []

        results = self._collection.get(
            limit=self._collection.count(),
        )

        designs = []
        for i in range(len(results["ids"])):
            designs.append({
                "id": results["ids"][i],
                "metadata": results["metadatas"][i] if results["metadatas"] else {},
            })

        return designs
