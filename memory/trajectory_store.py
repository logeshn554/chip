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
        }


class TrajectoryStore:
    """File-backed JSONL trajectory store."""

    def __init__(self, log_dir: str = "./trajectories"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "trajectories.jsonl")

    def save_trajectory(
        self,
        task: str,
        steps: list[dict[str, Any] | TrajectoryStep],
        reward: float,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Persist a finished episode trajectory to JSONL."""
        cleaned_steps = []
        for s in steps:
            if isinstance(s, TrajectoryStep):
                cleaned_steps.append(s.to_clean_dict())
            elif isinstance(s, dict):
                cleaned_steps.append(s)

        record = TrajectoryRecord(
            task=task,
            steps=cleaned_steps,
            reward=reward,
            metadata=metadata or {},
        )

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_standard_dict()) + "\n")

        logger.info(f"Trajectory recorded: {record.trajectory_id} (Reward: {reward}, Steps: {len(cleaned_steps)})")
        return record.trajectory_id

    def list_trajectories(self, min_reward: float = 0.0) -> list[dict[str, Any]]:
        """Read and return trajectories, optionally filtered by reward."""
        if not os.path.exists(self.log_file):
            return []

        results = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        if item.get("reward", 0.0) >= min_reward:
                            results.append(item)
                    except Exception:
                        continue
        return results
