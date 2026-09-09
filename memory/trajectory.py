"""
Trajectory Store — records complete agent episodes for RL training.

Each episode is a sequence of (state, action, observation, reward) tuples
from task start to completion, stored in JSONL format for efficient
append and streaming during SFT/GRPO training.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from typing import Any

from agent.schemas import Episode, TrajectoryStep

logger = logging.getLogger(__name__)


class TrajectoryStore:
    """Persistent storage for agent trajectory episodes.

    Format: JSON Lines (.jsonl) — one line per episode.
    Each episode contains the full sequence of steps with
    state summaries, actions, observations, and rewards.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.store_dir = config.get("store_dir", "./trajectories")
        self.format = config.get("format", "jsonl")

        os.makedirs(self.store_dir, exist_ok=True)

        logger.info(f"TrajectoryStore: dir={self.store_dir}")

    # ── Episode Management ───────────────────────────────────────────

    def start_episode(self, task: str) -> Episode:
        """Create a new episode for recording.

        Args:
            task: The task description

        Returns:
            New Episode object
        """
        episode = Episode(task=task)
        logger.debug(f"Started episode {episode.episode_id} for task: {task[:50]}")
        return episode

    def record_step(
        self,
        episode: Episode,
        state_summary: str,
        action: str,
        action_params: dict[str, Any] | None = None,
        observation: str = "",
        reward: float = 0.0,
    ) -> TrajectoryStep:
        """Record a single step in an episode.

        Args:
            episode: The episode to record into
            state_summary: Brief summary of current state
            action: Action taken
            action_params: Action parameters
            observation: Result/observation after action
            reward: Step reward

        Returns:
            The recorded TrajectoryStep
        """
        step = TrajectoryStep(
            step_index=len(episode.steps),
            state_summary=state_summary,
            action=action,
            action_params=action_params or {},
            observation=observation[:2000],  # Cap observation length
            reward=reward,
        )
        episode.steps.append(step)
        return step

    async def save_episode(self, episode: Episode) -> str:
        """Save a completed episode to disk.

        Args:
            episode: Completed episode with all steps

        Returns:
            Path to the saved file
        """
        episode.completed_at = episode.completed_at or time.time()

        # Determine filename
        filename = f"episode_{episode.episode_id}.jsonl"
        filepath = os.path.join(self.store_dir, filename)

        # Also append to the master trajectory file
        master_path = os.path.join(self.store_dir, "all_trajectories.jsonl")

        # Serialize episode
        episode_data = self._serialize_episode(episode)

        # Write individual episode file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json.dumps(episode_data) + "\n")

        # Append to master file
        with open(master_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(episode_data) + "\n")

        logger.info(
            f"Saved episode {episode.episode_id}: "
            f"{len(episode.steps)} steps, reward={episode.final_reward:.3f}, "
            f"success={episode.success}"
        )

        return filepath

    # ── Loading ──────────────────────────────────────────────────────

    def load_episode(self, episode_id: str) -> Episode | None:
        """Load a specific episode by ID."""
        filename = f"episode_{episode_id}.jsonl"
        filepath = os.path.join(self.store_dir, filename)

        if not os.path.exists(filepath):
            logger.warning(f"Episode file not found: {filepath}")
            return None

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.loads(f.readline())

        return self._deserialize_episode(data)

    def load_all_episodes(
        self,
        min_reward: float | None = None,
        success_only: bool = False,
    ) -> list[Episode]:
        """Load all episodes, optionally filtered.

        Args:
            min_reward: Minimum final reward threshold
            success_only: Only load successful episodes

        Returns:
            List of Episode objects
        """
        master_path = os.path.join(self.store_dir, "all_trajectories.jsonl")

        if not os.path.exists(master_path):
            return []

        episodes = []
        with open(master_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    episode = self._deserialize_episode(data)

                    # Apply filters
                    if min_reward is not None and episode.final_reward < min_reward:
                        continue
                    if success_only and not episode.success:
                        continue

                    episodes.append(episode)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping malformed trajectory line: {e}")

        logger.info(f"Loaded {len(episodes)} episodes from {master_path}")
        return episodes

    # ── Statistics ────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Get summary statistics over all stored trajectories."""
        episodes = self.load_all_episodes()

        if not episodes:
            return {
                "total_episodes": 0,
                "success_rate": 0.0,
                "avg_reward": 0.0,
                "avg_steps": 0.0,
                "total_steps": 0,
            }

        rewards = [e.final_reward for e in episodes]
        successes = sum(1 for e in episodes if e.success)
        total_steps = sum(len(e.steps) for e in episodes)

        return {
            "total_episodes": len(episodes),
            "success_rate": successes / len(episodes),
            "avg_reward": sum(rewards) / len(rewards),
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "avg_steps": total_steps / len(episodes),
            "total_steps": total_steps,
        }

    # ── Serialization ────────────────────────────────────────────────

    @staticmethod
    def _serialize_episode(episode: Episode) -> dict[str, Any]:
        """Convert an Episode to a JSON-serializable dict."""
        return {
            "episode_id": episode.episode_id,
            "task": episode.task,
            "steps": [
                {
                    "step_index": s.step_index,
                    "state_summary": s.state_summary,
                    "action": s.action,
                    "action_params": s.action_params,
                    "observation": s.observation,
                    "reward": s.reward,
                    "timestamp": s.timestamp,
                }
                for s in episode.steps
            ],
            "final_reward": episode.final_reward,
            "total_iterations": episode.total_iterations,
            "success": episode.success,
            "started_at": episode.started_at,
            "completed_at": episode.completed_at,
            "metadata": episode.metadata,
        }

    @staticmethod
    def _deserialize_episode(data: dict[str, Any]) -> Episode:
        """Convert a dict back to an Episode."""
        steps = [
            TrajectoryStep(
                step_index=s["step_index"],
                state_summary=s.get("state_summary", ""),
                action=s["action"],
                action_params=s.get("action_params", {}),
                observation=s.get("observation", ""),
                reward=s.get("reward", 0.0),
                timestamp=s.get("timestamp", 0.0),
            )
            for s in data.get("steps", [])
        ]

        return Episode(
            episode_id=data["episode_id"],
            task=data["task"],
            steps=steps,
            final_reward=data.get("final_reward", 0.0),
            total_iterations=data.get("total_iterations", 0),
            success=data.get("success", False),
            started_at=data.get("started_at", 0.0),
            completed_at=data.get("completed_at"),
            metadata=data.get("metadata", {}),
        )
