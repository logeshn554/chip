"""
Knowledge Memory — technical documentation RAG via ChromaDB.

Stores and retrieves chunks from SystemVerilog references,
RISC-V specs, NPU architecture papers, and datasheets.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Any

import chromadb

from agent.schemas import Document

logger = logging.getLogger(__name__)


class KnowledgeMemory:
    """Vector-based retrieval for technical documentation.

    Uses ChromaDB with sentence-transformer embeddings for
    semantic search over ingested documents.
    """

    def __init__(
        self,
        chroma_path: str = "./data/chroma",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        config: dict[str, Any] | None = None,
    ):
        config = config or {}
        self.collection_name = config.get("collection", "knowledge")
        self.chunk_size = config.get("chunk_size", 512)
        self.chunk_overlap = config.get("chunk_overlap", 64)

        # Initialize ChromaDB
        self._client = chromadb.PersistentClient(path=chroma_path)

        # Try loading SentenceTransformer, fallback to deterministic vectorizer if offline or auth fails
        try:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            self._embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
            # Test call to ensure it didn't fail authentication
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
            f"KnowledgeMemory: collection='{self.collection_name}', "
            f"docs={self._collection.count()}"
        )

    # ── Ingestion ────────────────────────────────────────────────────

    def ingest(self, text: str, metadata: dict[str, Any] | None = None) -> int:
        """Ingest a text document by chunking and storing embeddings.

        Args:
            text: Full document text
            metadata: Optional metadata (source, topic, date, etc.)

        Returns:
            Number of chunks stored
        """
        metadata = metadata or {}
        chunks = self._chunk_text(text)

        if not chunks:
            logger.warning("No chunks generated from text")
            return 0

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{self._make_id(chunk)}_{i}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                **metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info(f"Ingested {len(chunks)} chunks (source: {metadata.get('source', 'unknown')})")
        return len(chunks)

    def ingest_file(self, filepath: str, metadata: dict[str, Any] | None = None) -> int:
        """Ingest a text file.

        Args:
            filepath: Path to the text file
            metadata: Optional additional metadata

        Returns:
            Number of chunks stored
        """
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return 0

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        meta = {"source": os.path.basename(filepath), "filepath": filepath}
        if metadata:
            meta.update(metadata)

        return self.ingest(text, metadata=meta)

    def ingest_directory(
        self,
        dirpath: str,
        extensions: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Ingest all matching files in a directory.

        Args:
            dirpath: Directory path
            extensions: File extensions to include (e.g., [".sv", ".v", ".md"])
            metadata: Optional metadata applied to all files

        Returns:
            Total number of chunks stored
        """
        extensions = extensions or [".sv", ".v", ".vh", ".md", ".txt", ".rst"]
        total = 0

        for root, _, files in os.walk(dirpath):
            for fname in files:
                if any(fname.endswith(ext) for ext in extensions):
                    fpath = os.path.join(root, fname)
                    total += self.ingest_file(fpath, metadata=metadata)

        logger.info(f"Ingested {total} total chunks from {dirpath}")
        return total

    # ── Retrieval ────────────────────────────────────────────────────

    def query(self, question: str, n: int = 5) -> list[Document]:
        """Query for relevant document chunks.

        Args:
            question: Natural language query
            n: Maximum number of results

        Returns:
            List of Document objects with content and similarity scores
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

            # ChromaDB returns distances; convert to similarity score
            # For cosine distance: similarity = 1 - distance
            score = max(0.0, 1.0 - distance)

            docs.append(Document(
                content=content,
                metadata=metadata,
                score=score,
                source=f"knowledge:{metadata.get('source', 'unknown')}",
            ))

        return docs

    # ── Chunking ─────────────────────────────────────────────────────

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks.

        Uses a simple word-boundary-aware splitting strategy.
        """
        words = text.split()
        if not words:
            return []

        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk.strip())
            start = end - self.chunk_overlap
            if start >= len(words):
                break

        return chunks

    @staticmethod
    def _make_id(text: str) -> str:
        """Generate a deterministic ID from text content."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
