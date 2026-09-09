"""
Dataset Builder — converts trajectory data into training formats.

Transforms raw trajectory episodes into SFT instruction/response pairs
and GRPO/DPO preference pairs for fine-tuning Qwen3-4B.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from agent.schemas import Episode

logger = logging.getLogger(__name__)


class TrajectoryDatasetBuilder:
    """Builds training datasets from stored trajectories.

    Supports:
    - SFT: instruction/response pairs from successful episodes
    - GRPO/DPO: preference pairs (better vs worse trajectories)
    - Filtering by reward threshold
    """

    def __init__(
        self,
        trajectory_dir: str = "./trajectories",
        output_dir: str = "./data/training",
    ):
        self.trajectory_dir = trajectory_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def build_sft_dataset(
        self,
        episodes: list[Episode],
        min_reward: float = 0.7,
    ) -> list[dict[str, str]]:
        """Build SFT dataset from successful trajectory episodes.

        Converts each successful episode step into an instruction/response pair:
        - Instruction = state + context
        - Response = action + params that led to reward

        Args:
            episodes: List of trajectory episodes
            min_reward: Minimum episode reward to include

        Returns:
            List of {"instruction": ..., "response": ...} dicts
        """
        dataset = []

        filtered = [e for e in episodes if e.final_reward >= min_reward and e.success]
        logger.info(
            f"Building SFT dataset: {len(filtered)}/{len(episodes)} episodes "
            f"(reward >= {min_reward})"
        )

        for episode in filtered:
            for step in episode.steps:
                instruction = (
                    f"Task: {episode.task}\n"
                    f"Current State: {step.state_summary}\n"
                    f"What action should you take next?"
                )

                response = json.dumps({
                    "thinking": f"Based on the current state, I should {step.action.lower()}.",
                    "action": step.action,
                    "params": step.action_params,
                })

                dataset.append({
                    "instruction": instruction,
                    "response": response,
                    "reward": step.reward,
                    "episode_id": episode.episode_id,
                })

        logger.info(f"Built SFT dataset: {len(dataset)} examples")
        return dataset

    def build_preference_dataset(
        self,
        episodes: list[Episode],
    ) -> list[dict[str, Any]]:
        """Build preference dataset for GRPO/DPO training.

        Pairs higher-reward episodes against lower-reward episodes
        for the same or similar tasks.

        Args:
            episodes: List of trajectory episodes

        Returns:
            List of {"prompt": ..., "chosen": ..., "rejected": ...} dicts
        """
        # Group episodes by task similarity (simple: exact match)
        task_groups: dict[str, list[Episode]] = {}
        for ep in episodes:
            key = ep.task[:100]  # Simple grouping key
            task_groups.setdefault(key, []).append(ep)

        dataset = []

        for task, group in task_groups.items():
            if len(group) < 2:
                continue

            # Sort by reward
            sorted_eps = sorted(group, key=lambda e: e.final_reward, reverse=True)

            # Pair best with worst
            for i in range(min(len(sorted_eps) // 2, 10)):
                better = sorted_eps[i]
                worse = sorted_eps[-(i + 1)]

                if better.final_reward <= worse.final_reward:
                    continue

                prompt = f"Task: {task}"
                chosen = self._episode_to_text(better)
                rejected = self._episode_to_text(worse)

                dataset.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "chosen_reward": better.final_reward,
                    "rejected_reward": worse.final_reward,
                })

        logger.info(f"Built preference dataset: {len(dataset)} pairs")
        return dataset

    def save_dataset(
        self,
        dataset: list[dict[str, Any]],
        name: str = "sft_dataset",
        format: str = "jsonl",
    ) -> str:
        """Save a dataset to disk.

        Args:
            dataset: List of training examples
            name: Dataset name
            format: "jsonl" or "json"

        Returns:
            Path to the saved file
        """
        filepath = os.path.join(self.output_dir, f"{name}.{format}")

        if format == "jsonl":
            with open(filepath, "w", encoding="utf-8") as f:
                for entry in dataset:
                    f.write(json.dumps(entry) + "\n")
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2)

        logger.info(f"Saved {len(dataset)} examples to {filepath}")
        return filepath

    @staticmethod
    def _episode_to_text(episode: Episode) -> str:
        """Convert an episode to a text representation for preference training."""
        lines = [f"Task: {episode.task}"]
        for step in episode.steps:
            lines.append(f"Step {step.step_index}: {step.action} → {step.observation[:200]}")
        lines.append(f"Final Reward: {episode.final_reward:.3f}")
        return "\n".join(lines)
