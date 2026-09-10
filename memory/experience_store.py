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
    error: str = ""
    correction: str = ""
    result: str = "passed"  # "passed", "failed"
    reward: float = 0.0
    tool_sequence: list[str] = field(default_factory=list)
    error_category: str = "GENERAL"  # "SYNTAX_LINT", "FUNCTIONAL_ASSERT", "SYNTHESIS_ERROR", "FORMAL_FAIL"
    error_signature: str = ""
    # Enhanced structured fields for self-evolution memory
    failure_type: str = ""
    tool: str = ""
    error_message: str = ""
    context: str = ""
    root_cause: str = ""
    fix: str = ""
    task_family: str = ""
    architecture_family: str = ""
    source_trajectory: str = ""
    confidence: float = 1.0
    id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        # Harmonize aliased and legacy field names
        if not self.failure_type:
            self.failure_type = self.error_category or "GENERAL"
        if not self.error_category:
            self.error_category = self.failure_type
        if not self.error_message:
            self.error_message = self.error
        if not self.error:
            self.error = self.error_message
        if not self.fix:
            self.fix = self.correction
        if not self.correction:
            self.correction = self.fix

        if not self.id:
            h = abs(hash(self.task + self.error + self.correction)) % 1000000
            self.id = f"exp_{h:06d}"

    def to_embedding_text(self) -> str:
        """Text used for embedding similarity retrieval."""
        return (
            f"Failure Type: {self.failure_type}\n"
            f"Tool: {self.tool}\n"
            f"Task: {self.task}\n"
            f"Task Family: {self.task_family}\n"
            f"Architecture Family: {self.architecture_family}\n"
            f"Error Signature: {self.error_signature or self.error_message[:150]}\n"
            f"Error: {self.error_message[:300]}\n"
            f"Root Cause: {self.root_cause}\n"
            f"Fix: {self.fix[:300]}"
        )


def categorize_error(error_text: str) -> str:
    """Categorize hardware error message into explicit failure taxonomy."""
    low = (error_text or "").lower()
    if any(k in low for k in ["formal", "sby", "bmc", "assertion violation"]):
        return "FORMAL_FAIL"
    elif any(k in low for k in ["%error-width", "%error-syntax", "syntax error", "undeclared", "expected", "lint"]):
        return "SYNTAX_LINT"
    elif any(k in low for k in ["assert", "test failed", "cocotb", "mismatch", "failed test"]):
        return "FUNCTIONAL_ASSERT"
    elif any(k in low for k in ["synthesis error", "cannot be synthesized", "yosys", "techmap"]):
        return "SYNTHESIS_ERROR"
    return "GENERAL"


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
                error_category="SYNTAX_LINT",
                error_signature="%Error-WIDTH: Operator '+' expects 32 bits, RHS 16 bits",
                tool_sequence=["CREATE_RTL", "RUN_VERILATOR", "INSPECT_ERROR", "EDIT_RTL", "RUN_VERILATOR", "RUN_COCOTB", "RUN_YOSYS"],
            ),
            Experience(
                task="Signed arithmetic in SystemVerilog",
                attempted_solution="logic [7:0] a, b; // Missing signed qualifier",
                error="Functional test failed: -5 * 4 evaluated to 1004 instead of -20",
                correction="Change input port declarations to: input logic signed [DATA_WIDTH-1:0] a, b;",
                result="passed",
                reward=8.0,
                error_category="FUNCTIONAL_ASSERT",
                error_signature="Functional test failed: negative multiplication sign bit error",
                tool_sequence=["CREATE_RTL", "RUN_COCOTB", "INSPECT_ERROR", "EDIT_RTL", "RUN_COCOTB"],
            ),
            Experience(
                task="Synthesizable delays in RTL",
                attempted_solution="#10 clk = ~clk; // Unsynthesizable delay construct",
                error="Synthesis error: Delays (#t) cannot be synthesized into physical logic.",
                correction="Use clocked sequential registers (always_ff @(posedge clk)) instead of simulation delays.",
                result="passed",
                reward=8.0,
                error_category="SYNTHESIS_ERROR",
                error_signature="Synthesis error: Delays (#t) cannot be synthesized",
                tool_sequence=["CREATE_RTL", "RUN_YOSYS", "INSPECT_ERROR", "EDIT_RTL", "RUN_YOSYS"],
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
            "error_category": exp.error_category,
            "error_signature": exp.error_signature,
            "failure_type": exp.failure_type or exp.error_category,
            "tool": exp.tool,
            "root_cause": exp.root_cause,
            "task_family": exp.task_family,
            "architecture_family": exp.architecture_family,
            "source_trajectory": exp.source_trajectory,
            "confidence": float(exp.confidence),
        }
        self.collection.upsert(
            ids=[exp.id],
            documents=[exp.to_embedding_text()],
            metadatas=[meta],
        )
        return exp.id

    def record_episode_experience(self, episode: Any) -> list[str]:
        """Deconstruct an episode into first-class failure/fix experiences."""
        steps = getattr(episode, "steps", []) or []
        added_ids = []

        for i in range(len(steps) - 1):
            curr = steps[i]
            nxt = steps[i + 1]

            curr_obs = getattr(curr, "observation", "") or ""
            nxt_obs = getattr(nxt, "observation", "") or ""
            curr_rew = getattr(curr, "reward", 0.0) or 0.0
            nxt_rew = getattr(nxt, "reward", 0.0) or 0.0

            has_error = "error" in curr_obs.lower() or "failed" in curr_obs.lower() or curr_rew < 0
            is_fixed = "passed" in nxt_obs.lower() or nxt_rew > curr_rew

            if has_error and is_fixed:
                curr_params = getattr(curr, "action_params", {}) or {}
                nxt_params = getattr(nxt, "action_params", {}) or {}
                attempted = curr_params.get("code", curr_params.get("current_rtl", str(curr_params)))
                correction = nxt_params.get("code", nxt_params.get("current_rtl", str(nxt_params)))
                cat = categorize_error(curr_obs)

                # Determine root cause estimate
                cause = "unspecified"
                if cat == "SYNTAX_LINT":
                    cause = "syntax or bitwidth mismatch in declaration"
                elif cat == "FUNCTIONAL_ASSERT":
                    cause = "functional arithmetic or reset behavior discrepancy"
                elif cat == "SYNTHESIS_ERROR":
                    cause = "unsynthesizable construct or missing techmap primitive"
                elif cat == "FORMAL_FAIL":
                    cause = "formal property assertion boundary failure"

                exp = Experience(
                    task=getattr(episode, "task", ""),
                    attempted_solution=str(attempted)[:300],
                    error=curr_obs[:300],
                    correction=str(correction)[:300],
                    result="passed",
                    reward=getattr(episode, "final_reward", 0.0),
                    error_category=cat,
                    error_signature=curr_obs[:120],
                    tool_sequence=[s.action for s in steps],
                    failure_type=cat,
                    tool=getattr(curr, "action", "TOOL"),
                    root_cause=cause,
                    task_family=getattr(episode, "task_id", "") or "hardware",
                    source_trajectory=getattr(episode, "episode_id", ""),
                    confidence=1.0,
                )
                exp_id = self.add(exp)
                added_ids.append(exp_id)

        return added_ids

    def retrieve_similar_failures(
        self,
        error_text: str,
        task: str = "",
        n_results: int = 2,
    ) -> list[dict[str, Any]]:
        """Find past experiences addressing similar errors."""
        cat = categorize_error(error_text)
        query_text = f"Category: {cat}\nTask: {task}\nError: {error_text}"
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
                    "error_category": meta.get("error_category", "GENERAL"),
                    "tool_sequence": tools,
                })
        return matches

    def retrieve_experiences_for_task(
        self,
        task: str,
        current_error: Optional[str] = None,
        n_results: int = 2,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant past debugging lessons and reference patterns for a task."""
        if current_error:
            return self.retrieve_similar_failures(error_text=current_error, task=task, n_results=n_results)

        query_text = f"Task: {task}\nRelevant design practices and past hardware solutions"
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
                matches.append({
                    "task": meta.get("task", ""),
                    "error": meta.get("error", ""),
                    "correction": meta.get("correction", ""),
                    "result": meta.get("result", ""),
                    "reward": meta.get("reward", 0.0),
                    "error_category": meta.get("error_category", "GENERAL"),
                })
        return matches
