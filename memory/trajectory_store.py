"""
Trajectory Memory Store — Records complete agent episodes for future SFT/GRPO/RL training.

Adheres strictly to the requested trajectory schema:
{
  "task": "...",
  "steps": [
    {"action": "retrieve_memory", "result": "..."},
    ...
  ],
  "reward": 0.91
}
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from typing import Any, Optional

from llm.interface import get_default_model

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryStep:
    """A single step within an agent episode."""
    action: str
    result: Optional[str] = None
    parameters: dict[str, Any] = field(default_factory=dict)
    query: Optional[str] = None
    thinking: Optional[str] = None

    def to_clean_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"action": self.action.lower()}
        if self.query:
            d["query"] = self.query
        if self.result is not None:
            d["result"] = self.result
        return d


@dataclass
class TrajectoryRecord:
    """A complete agent trajectory episode."""
    task: str
    steps: list[dict[str, Any]]
    reward: float
    trajectory_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)
    provider: str = "ollama"
    model: str = field(default_factory=get_default_model)
    fallback_used: bool = False
    rtl_source: str = "qwen"

    def __post_init__(self):
        if not self.trajectory_id:
            h = abs(hash(self.task + str(self.timestamp))) % 1000000
            self.trajectory_id = f"traj_{h:06d}"

    def to_standard_dict(self) -> dict[str, Any]:
        """Format matching user's requested specification."""
        return {
            "trajectory_id": self.trajectory_id,
            "task": self.task,
            "steps": self.steps,
            "reward": round(self.reward, 4),
            "timestamp": self.timestamp,
            "provider": self.provider,
            "model": self.model,
            "fallback_used": self.fallback_used,
            "rtl_source": self.rtl_source,
            "metadata": self.metadata,
        }

from memory.trajectory import TrajectoryStore as CanonicalTrajectoryStore


class TrajectoryStore(CanonicalTrajectoryStore):
    """File-backed JSONL trajectory store aliased to the canonical TrajectoryStore."""
    pass
