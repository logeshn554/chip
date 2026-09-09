"""
Experience Memory Store — Semantic retrieval of previous failures, fixes, and outcomes.

Stores:
- task
- attempted solution
- error
- correction
- result
- reward
- tool sequence

Supports vector similarity search on error signatures so the agent can reuse past debugging fixes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from typing import Any, Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """Record of an attempt, failure, correction, and outcome."""
    task: str
    attempted_solution: str
    error: str
    correction: str
    result: str  # "passed", "failed"
    reward: float = 0.0
    tool_sequence: list[str] = field(default_factory=list)
    id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.id:
            h = abs(hash(self.task + self.error + self.correction)) % 1000000
            self.id = f"exp_{h:06d}"

    def to_embedding_text(self) -> str:
        """Text used for embedding similarity retrieval."""
        return f"Task: {self.task}\nError: {self.error}\nAttempted: {self.attempted_solution[:200]}"


class ExperienceStore:
    """Experience memory store for learning from failures."""

    def __init__(self, persist_dir: str = "./data/chroma/experience"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="hardware_experience",
            metadata={"description": "Records of past hardware errors, fixes, and rewards"},
        )
        if self.collection.count() == 0:
            self._seed_common_experiences()

    def _seed_common_experiences(self) -> None:
        """Seed known SystemVerilog MAC failure modes and their corrections."""
        seed_cases = [
            Experience(
                task="Design an 8-bit signed MAC",
                attempted_solution="assign out = acc + (a * b); // unsigned extension bug",
                error="%Error-WIDTH: mac.sv: Operator '+' expects 32 bits on LHS, but RHS generates 16 bits without sign extension",
                correction="acc_reg <= acc_reg + {{ (ACC_WIDTH - 2*DATA_WIDTH){product[2*DATA_WIDTH-1]} }, product};",
                result="passed",
                reward=8.0,
                tool_sequence=["CREATE_RTL", "RUN_VERILATOR", "INSPECT_ERROR", "EDIT_RTL", "RUN_VERILATOR", "RUN_COCOTB", "RUN_YOSYS"],
            ),
            Experience(
                task="Signed arithmetic in SystemVerilog",
                attempted_solution="logic [7:0] a, b; // Missing signed qualifier",
                error="Functional test failed: -5 * 4 evaluated to 1004 instead of -20",
                correction="Change input port declarations to: input logic signed [DATA_WIDTH-1:0] a, b;",
                result="passed",
                reward=8.0,
                tool_sequence=["CREATE_RTL", "RUN_COCOTB", "INSPECT_ERROR", "EDIT_RTL", "RUN_COCOTB"],
            ),
        ]
        for exp in seed_cases:
            self.add(exp)
        logger.info(f"Seeded {len(seed_cases)} common experiences into ExperienceStore.")

    def add(self, exp: Experience) -> str:
        """Save an experience entry."""
        meta = {
            "task": exp.task,
            "result": exp.result,
            "reward": float(exp.reward),
            "tool_sequence": json.dumps(exp.tool_sequence),
            "timestamp": exp.timestamp,
            "correction": exp.correction,
            "error": exp.error,
        }
        self.collection.upsert(
            ids=[exp.id],
            documents=[exp.to_embedding_text()],
            metadatas=[meta],
        )
        return exp.id

    def retrieve_similar_failures(
        self,
        error_text: str,
        task: str = "",
        n_results: int = 2,
    ) -> list[dict[str, Any]]:
        """Find past experiences addressing similar errors."""
        query_text = f"Task: {task}\nError: {error_text}"
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(n_results, self.collection.count()),
        )

        matches = []
        if results and results.get("documents"):
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            for doc, meta in zip(docs, metas):
                tools = []
                try:
                    tools = json.loads(meta.get("tool_sequence", "[]"))
                except Exception:
                    pass
                matches.append({
                    "task": meta.get("task", ""),
                    "error": meta.get("error", ""),
                    "correction": meta.get("correction", ""),
                    "result": meta.get("result", ""),
                    "reward": meta.get("reward", 0.0),
                    "tool_sequence": tools,
                })
        return matches
