"""
Knowledge Memory Store — Technical documentation RAG.

Stores SystemVerilog standards, digital design concepts, signed arithmetic rules,
NPU/MAC specifications, FPGA docs, and retrieved web research with strict metadata.
Uses ChromaDB for vector similarity search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import os
from typing import Any, Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeChunk:
    """A chunk of technical knowledge with metadata."""
    content: str
    source_url: str
    title: str
    topic: str
    document_type: str = "specification"  # "specification", "guideline", "paper", "web_extract"
    chunk_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = f"k_{abs(hash(self.content + self.source_url)) % 1000000:06d}"


class KnowledgeStore:
    """Vector database backed knowledge memory store."""

    def __init__(self, persist_dir: str = "./data/chroma/knowledge"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="hardware_knowledge",
            metadata={"description": "Hardware design technical knowledge and specs"},
        )
        # Pre-seed essential knowledge if empty
        try:
            if self.collection.count() == 0:
                self._seed_default_knowledge()
        except Exception as e:
            logger.warning(f"Incompatible or corrupted Chroma collection detected ({e}). Re-creating collection.")
            try:
                self.client.delete_collection("hardware_knowledge")
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(
                name="hardware_knowledge",
                metadata={"description": "Hardware design technical knowledge and specs"},
            )
            try:
                self._seed_default_knowledge()
            except Exception as e2:
                logger.warning(f"Could not re-seed knowledge store: {e2}")

    def _seed_default_knowledge(self) -> None:
        """Seed core SystemVerilog and MAC design guidelines."""
        seeds = [
            KnowledgeChunk(
                title="SystemVerilog Signed Arithmetic and Multiply-Accumulate (MAC)",
                topic="signed_arithmetic",
                source_url="https://ieeexplore.ieee.org/document/8299595",
                document_type="standard",
                content=(
                    "In SystemVerilog, signed arithmetic on logic vectors requires the 'signed' keyword: "
                    "'input logic signed [7:0] a, b;'. Multiplication of two N-bit signed numbers yields a 2N-bit signed result. "
                    "When accumulating into an M-bit accumulator (where M > 2N), sign extension must be performed: "
                    "sign extension preserves the sign bit: '{{ (ACC_WIDTH - 2*DATA_WIDTH){product[2*DATA_WIDTH-1]} }, product}'. "
                    "Alternatively, casting with '$signed(product)' before addition ensures proper two's complement sign extension. "
                    "A standard synchronous MAC evaluates: result = (a * b) + acc on positive clock edge with active-low reset."
                ),
            ),
            KnowledgeChunk(
                title="Synthesizable SystemVerilog Coding Standards for EDA (Verilator & Yosys)",
                topic="eda_conventions",
                source_url="https://verilator.org/guide/latest/verilating.html",
                document_type="guideline",
                content=(
                    "For synthesizable SystemVerilog: Avoid initial blocks in synthesizable modules; use synchronous or asynchronous "
                    "resets within 'always_ff @(posedge clk or negedge rst_n)'. Combinational logic should strictly use 'always_comb' "
                    "to prevent inferred latches. Ensure all branches of an 'if' or 'case' assign all variables. "
                    "Port widths must match: assigning an 8-bit expression to a 16-bit net without sign-extension generates "
                    "Verilator WIDTH warnings or lint errors. Do not use vendor-specific primitives (like Xilinx DSP48E1) directly."
                ),
            ),
            KnowledgeChunk(
                title="MAC Unit Parameterization and Interface Protocols",
                topic="mac_architecture",
                source_url="https://github.com/chipsalliance/hardware-standards",
                document_type="specification",
                content=(
                    "An 8-bit signed MAC unit should be parameterized with 'parameter int DATA_WIDTH = 8' and 'parameter int ACC_WIDTH = 32'. "
                    "The primary interface includes: 'clk', 'rst_n' (asynchronous active-low reset), 'en' (clock enable), "
                    "'clr' (synchronous clear/accumulate restart), 'a' (signed input [DATA_WIDTH-1:0]), 'b' (signed input [DATA_WIDTH-1:0]), "
                    "'out' (signed accumulator output [ACC_WIDTH-1:0]), and 'valid' (indicates output is valid)."
                ),
            ),
        ]
        self.add_chunks(seeds)
        logger.info(f"Seeded {len(seeds)} default knowledge chunks into KnowledgeStore.")

    def add_chunk(self, chunk: KnowledgeChunk) -> str:
        """Add a single knowledge chunk."""
        meta = {
            "source_url": chunk.source_url,
            "title": chunk.title,
            "topic": chunk.topic,
            "document_type": chunk.document_type,
            "timestamp": chunk.timestamp,
            "chunk_id": chunk.chunk_id,
        }
        meta.update(chunk.metadata)

        self.collection.upsert(
            ids=[chunk.chunk_id],
            documents=[chunk.content],
            metadatas=[meta],
        )
        return chunk.chunk_id

    def add_chunks(self, chunks: list[KnowledgeChunk]) -> list[str]:
        """Batch add chunks."""
        ids = [c.chunk_id for c in chunks]
        docs = [c.content for c in chunks]
        metas = []
        for c in chunks:
            m = {
                "source_url": c.source_url,
                "title": c.title,
                "topic": c.topic,
                "document_type": c.document_type,
                "timestamp": c.timestamp,
                "chunk_id": c.chunk_id,
            }
            m.update(c.metadata)
            metas.append(m)

        self.collection.upsert(ids=ids, documents=docs, metadatas=metas)
        return ids

    def query(
        self,
        query_text: str,
        n_results: int = 3,
        topic: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Query knowledge base for relevant chunks."""
        where = {"topic": topic} if topic else None
        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(n_results, max(1, self.collection.count())),
            where=where,
        )

        formatted = []
        if results and results.get("documents"):
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
            for doc, meta, dist in zip(docs, metas, dists):
                formatted.append({
                    "content": doc,
                    "metadata": meta,
                    "distance": dist,
                    "source_url": meta.get("source_url", ""),
                    "title": meta.get("title", ""),
                })
        return formatted
