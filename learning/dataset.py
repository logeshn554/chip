"""
Dataset Builder — Converts trajectory data into reproducible training datasets.

Transforms raw trajectory episodes into:
- SFT instruction/response pairs from verified, high-reward episodes
- Failure/Fix pairs teaching the model how to repair syntax/lint/test errors
- GRPO/DPO preference pairs (chosen vs rejected trajectories)
- Filtered, quality-scored, deduplicated dataset formats
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
from typing import Any, Optional

from agent.schemas import Episode, TrajectoryStep

logger = logging.getLogger(__name__)


@dataclass
class RLTransition:
    """Standardized Markov Decision Process (MDP) transition: (s_t, a_t, r_t, s_{t+1}, done)."""
    state: dict[str, Any]
    action: str
    action_params: dict[str, Any]
    reward: float
    next_state: dict[str, Any]
    done: bool
    info: dict[str, Any]
    episode_id: str
    step_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "action": self.action,
            "action_params": self.action_params,
            "reward": self.reward,
            "next_state": self.next_state,
            "done": self.done,
            "info": self.info,
            "episode_id": self.episode_id,
            "step_index": self.step_index,
        }


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
        """Build pairwise preference dataset (prompt, chosen, rejected) for DPO training."""
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

    def build_grpo_prompt_dataset(
        self,
        task_ids_or_episodes: Optional[list[Any]] = None,
        include_all_levels: bool = False,
    ) -> list[dict[str, Any]]:
        """Build prompt dataset for online/group-relative GRPO training.
        
        Unlike pairwise DPO datasets, GRPO requires only prompt inputs with task specifications.
        The model samples G candidate completions per prompt, which are evaluated by
        HardwareRewardEvaluator against the genuine EDA pipeline to calculate group advantages.
        """
        import re
        from benchmarks.curriculum import BenchmarkCurriculum
        curriculum = BenchmarkCurriculum()

        resolved_task_ids: list[str] = []
        if task_ids_or_episodes:
            for item in task_ids_or_episodes:
                if isinstance(item, str):
                    resolved_task_ids.append(item)
                elif hasattr(item, "metadata") and isinstance(item.metadata, dict):
                    tid = item.metadata.get("benchmark_task_id")
                    if tid:
                        resolved_task_ids.append(tid)
                    else:
                        m = re.search(r"\b(L[1-7]_[A-Z0-9_]+)\b", getattr(item, "task", ""))
                        if m:
                            resolved_task_ids.append(m.group(1))

        if resolved_task_ids:
            tasks = [curriculum.get_task(tid) for tid in resolved_task_ids if curriculum.get_task(tid)]
        elif include_all_levels:
            tasks = list(curriculum._tasks.values())
        else:
            tasks = curriculum.get_level_tasks(1) + curriculum.get_level_tasks(3)

        dataset = []
        for task in tasks:
            system_prompt = (
                "You are an expert digital design engineer writing synthesizable SystemVerilog (IEEE 1800-2012).\n"
                "Produce only clean, synthesizable SystemVerilog code without markdown explanations."
            )
            user_prompt = task.get_public_spec()
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            dataset.append({
                "prompt": full_prompt,
                "benchmark_task_id": task.id,
                "top_module": task.top_module,
                "level": task.level,
                "name": task.name,
                "verification_criteria": task.verification_criteria,
            })

        logger.info(f"Built GRPO prompt dataset: {len(dataset)} prompts across curriculum.")
        return dataset

    def save_grpo_prompts(
        self,
        dataset: list[dict[str, Any]],
        name: str = "grpo_prompts",
    ) -> str:
        """Save GRPO prompt dataset as JSONL."""
        path = os.path.join(self.output_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item) + "\n")
        logger.info(f"Saved {len(dataset)} GRPO prompt records to {path}")
        return path

    def build_rl_transitions(
        self,
        episodes: list[Episode],
    ) -> list[RLTransition]:
        """Convert trajectory episodes into standardized (s_t, a_t, r_t, s_{t+1}, done) RL transitions.
        
        Transforms raw agent logs into an explicit RL transition dataset suitable for
        offline RL, policy optimization (PPO/GRPO), and Q/value learning.
        """
        transitions = []

        for ep in episodes:
            num_steps = len(ep.steps)
            for i, step in enumerate(ep.steps):
                is_last_step = (i == num_steps - 1)
                
                # Current state representation
                curr_state = {
                    "task": ep.task,
                    "step_index": step.step_index,
                    "state_summary": step.state_summary,
                }

                # Next state representation
                if not is_last_step:
                    next_step = ep.steps[i + 1]
                    next_state = {
                        "task": ep.task,
                        "step_index": next_step.step_index,
                        "state_summary": next_step.state_summary,
                        "last_observation": step.observation,
                    }
                else:
                    next_state = {
                        "task": ep.task,
                        "step_index": step.step_index + 1,
                        "state_summary": "terminal",
                        "last_observation": step.observation,
                    }

                transition = RLTransition(
                    state=curr_state,
                    action=step.action,
                    action_params=step.action_params,
                    reward=step.reward,
                    next_state=next_state,
                    done=is_last_step or (step.action == "COMPLETE"),
                    info={
                        "model": ep.metadata.get("model", "Qwen3-4B"),
                        "duration": getattr(ep, "duration_s", 0.0),
                        "success": ep.success,
                        "final_reward": ep.final_reward,
                    },
                    episode_id=ep.episode_id,
                    step_index=step.step_index,
                )
                transitions.append(transition)

        logger.info(f"Built {len(transitions)} standardized RL transitions from {len(episodes)} episodes.")
        return transitions

    def save_rl_transitions(
        self,
        transitions: list[RLTransition],
        name: str = "rl_transitions",
    ) -> str:
        """Save RL transitions as JSONL dataset."""
        dataset = [t.to_dict() for t in transitions]
        return self.save_dataset(dataset, name=name, format="jsonl")

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
