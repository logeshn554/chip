"""
Dataset Builder — Converts trajectory data into reproducible training datasets.

Transforms raw trajectory episodes into:
- SFT instruction/response pairs from verified, high-reward episodes
- Failure/Fix pairs teaching the model how to repair syntax/lint/test errors
- GRPO/DPO preference pairs (chosen vs rejected trajectories)
- Filtered, quality-scored, deduplicated dataset formats
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Optional

from agent.schemas import Episode, TrajectoryStep

logger = logging.getLogger(__name__)


class TrajectoryDatasetBuilder:
    """Builds reproducible training datasets from stored trajectories."""

    def __init__(
        self,
        trajectory_dir: str = "./trajectories",
        output_dir: str = "./data/training",
    ):
        self.trajectory_dir = trajectory_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def filter_trajectories(
        self,
        episodes: list[Episode],
        min_reward: float = 0.7,
        require_success: bool = True,
        deduplicate: bool = True,
    ) -> list[Episode]:
        """Filter episodes by quality score and deduplicate.

        Args:
            episodes: Input list of episodes
            min_reward: Minimum final reward threshold
            require_success: Whether episode must be flagged success
            deduplicate: Filter out identical trajectories
        """
        filtered = []
        seen_hashes = set()

        for ep in episodes:
            if ep.final_reward < min_reward:
                continue
            if require_success and not ep.success:
                continue

            if deduplicate:
                # Hash task + action sequence
                action_seq = "-".join(s.action for s in ep.steps)
                h = hashlib.sha256(f"{ep.task}|{action_seq}".encode("utf-8")).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

            filtered.append(ep)

        logger.info(f"Filtered {len(episodes)} episodes down to {len(filtered)} high-quality episodes.")
        return filtered

    def build_sft_dataset(
        self,
        episodes: list[Episode],
        min_reward: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Build SFT dataset from successful trajectory episodes."""
        filtered = self.filter_trajectories(episodes, min_reward=min_reward, require_success=True)
        dataset = []

        for episode in filtered:
            for step in episode.steps:
                instruction = (
                    f"Task: {episode.task}\n"
                    f"Current State: {step.state_summary}\n"
                    f"What action should you take next?"
                )
                response = json.dumps({
                    "thinking": f"Based on the hardware requirements, I should execute {step.action}.",
                    "action": step.action,
                    "params": step.action_params,
                }, indent=2)

                dataset.append({
                    "instruction": instruction,
                    "response": response,
                    "reward": step.reward,
                    "episode_id": episode.episode_id,
                    "step_index": step.step_index,
                })

        logger.info(f"Built SFT dataset: {len(dataset)} examples")
        return dataset

    def build_failure_fix_pairs(
        self,
        episodes: list[Episode],
    ) -> list[dict[str, Any]]:
        """Extract failure and successful repair pairs from trajectories.

        Teaches the model: when encountering error X, produce repair Y.
        """
        pairs = []

        for ep in episodes:
            for i in range(len(ep.steps) - 1):
                curr = ep.steps[i]
                nxt = ep.steps[i + 1]

                # If current step encountered an error and next step repaired it
                has_error = "error" in curr.observation.lower() or curr.reward < 0
                next_success = nxt.reward > 0 or "passed" in nxt.observation.lower()

                if has_error and next_success:
                    instruction = (
                        f"Task: {ep.task}\n"
                        f"Failed Action: {curr.action}\n"
                        f"Error Observed: {curr.observation}\n"
                        f"How do you repair this failure?"
                    )
                    response = json.dumps({
                        "thinking": "Analyzing failure and proposing corrective action.",
                        "action": nxt.action,
                        "params": nxt.action_params,
                    }, indent=2)

                    pairs.append({
                        "instruction": instruction,
                        "response": response,
                        "episode_id": ep.episode_id,
                        "error_stage": curr.action,
                    })

        logger.info(f"Built failure/fix dataset: {len(pairs)} pairs")
        return pairs

    def build_preference_dataset(
        self,
        episodes: list[Episode],
    ) -> list[dict[str, Any]]:
        """Build preference dataset for GRPO/DPO training."""
        task_groups: dict[str, list[Episode]] = {}
        for ep in episodes:
            key = ep.task.strip().lower()[:80]
            task_groups.setdefault(key, []).append(ep)

        dataset = []

        for task, group in task_groups.items():
            if len(group) < 2:
                continue

            sorted_eps = sorted(group, key=lambda e: e.final_reward, reverse=True)

            for i in range(min(len(sorted_eps) // 2, 5)):
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
        """Save a dataset to disk."""
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
