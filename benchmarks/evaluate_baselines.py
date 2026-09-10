"""
Baseline vs. Trained Model Evaluation Protocol for Hardware RL.

Establishes scientific verification across 3 paradigms on held-out benchmark tasks:
- BASELINE 1: Direct RTL generation (one-shot LLM completion -> EDA verification)
- BASELINE 2: Interactive Agent Loop (untrained model + memory + tools + iterative EDA repair)
- TRAINED: Fine-Tuned / GRPO Model (trained adapter + interactive agent loop -> EDA)

Metrics Tracked on Completely Held-Out Benchmark Tests:
1. Functional Pass Rate (public + held-out test vectors)
2. Formal Verification Pass Rate (SymbiYosys assertions)
3. Synthesis Success Rate and Gate/Cell Area
4. Average Steps / Iteration Retries to convergence
5. Tool-Grounded Design Quality Reward
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import logging
import os
from typing import Any, Optional

from benchmarks.curriculum import BenchmarkCurriculum, BenchmarkTask
from learning.grpo import HardwareRewardEvaluator
from learning.environment import HardwareDesignEnv

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetricRecord:
    """Evaluation metrics for a single task and paradigm."""
    paradigm: str           # "baseline_1_direct", "baseline_2_agent", "trained_model"
    task_id: str
    top_module: str
    level: int
    functional_pass: bool
    test_pass_rate: float
    formal_pass: bool
    synthesis_pass: bool
    cells: Optional[int]
    steps_taken: int
    reward: float
    error: str = ""


@dataclass
class ParadigmSummary:
    """Aggregated results across all benchmark tasks for an evaluation paradigm."""
    paradigm: str
    total_tasks: int = 0
    functional_pass_rate: float = 0.0
    formal_pass_rate: float = 0.0
    synthesis_pass_rate: float = 0.0
    average_cells: float = 0.0
    average_steps: float = 0.0
    average_reward: float = 0.0
    records: list[EvaluationMetricRecord] = field(default_factory=list)


class BaselineEvaluationHarness:
    """Executes the 3-paradigm scientific evaluation protocol on held-out tasks."""

    def __init__(
        self,
        work_dir: str = "./sim_build/baseline_eval",
        curriculum: Optional[BenchmarkCurriculum] = None,
        evaluation_mode: str = "research_fast",
    ):
        self.work_dir = work_dir
        self.curriculum = curriculum or BenchmarkCurriculum()
        self.evaluation_mode = evaluation_mode
        self.evaluator = HardwareRewardEvaluator(
            work_dir=os.path.join(work_dir, "eda"),
            evaluation_mode=evaluation_mode,
            require_exact_task=True,
        )
        os.makedirs(work_dir, exist_ok=True)

    async def evaluate_direct_rtl(
        self,
        task: BenchmarkTask,
        rtl_code: str,
    ) -> EvaluationMetricRecord:
        """Baseline 1: Direct one-shot RTL completion evaluated by the genuine EDA pipeline."""
        res = await self.evaluator.evaluate_completion_async(
            completion=rtl_code,
            benchmark_task_id=task.id,
            sub_work_dir=f"direct_{task.id}",
        )
        compile_pass = bool(res.get("compile_pass", False))
        pass_rate = float(res.get("pass_rate", 0.0))
        formal_status = str(res.get("formal_status", "SKIPPED"))
        area = res.get("area")
        cells = int(area) if (area is not None and area > 0) else None

        return EvaluationMetricRecord(
            paradigm="baseline_1_direct",
            task_id=task.id,
            top_module=task.top_module,
            level=task.level,
            functional_pass=(compile_pass and pass_rate == 1.0),
            test_pass_rate=pass_rate,
            formal_pass=(formal_status == "PASS"),
            synthesis_pass=(cells is not None and cells > 0),
            cells=cells,
            steps_taken=1,
            reward=float(res.get("reward", 0.0)),
            error=str(res.get("error", "")),
        )

    async def evaluate_agent_loop(
        self,
        task: BenchmarkTask,
        policy: Any,
        paradigm_name: str = "baseline_2_agent",
        max_steps: int = 8,
    ) -> EvaluationMetricRecord:
        """Baseline 2 or Trained Model: Multi-turn agent interaction in HardwareDesignEnv."""
        task_dir = os.path.join(self.work_dir, f"{paradigm_name}_{task.id}")
        env = HardwareDesignEnv(
            work_dir=task_dir,
            benchmark_task_id=task.id,
            max_steps=max_steps,
        )

        obs, info = env.reset()
        done = False
        step_count = 0

        while not done and step_count < max_steps:
            step_count += 1
            action = policy.select_action(obs) if hasattr(policy, "select_action") else "COMPLETE"
            obs, reward, terminated, truncated, info = await env.step_async(action)
            done = terminated or truncated

        # Terminal EDA verification with strict held-out evaluation
        final_code = obs.get("current_rtl", "")
        if not final_code and env.current_rtl_path and os.path.exists(env.current_rtl_path):
            try:
                with open(env.current_rtl_path, "r", encoding="utf-8") as f:
                    final_code = f.read()
            except Exception:
                pass

        eval_res = await self.evaluator.evaluate_completion_async(
            completion=final_code,
            benchmark_task_id=task.id,
            sub_work_dir=f"agent_final_{paradigm_name}_{task.id}",
        )

        compile_pass = bool(eval_res.get("compile_pass", False))
        pass_rate = float(eval_res.get("pass_rate", 0.0))
        formal_status = str(eval_res.get("formal_status", "SKIPPED"))
        area = eval_res.get("area")
        cells = int(area) if (area is not None and area > 0) else None

        return EvaluationMetricRecord(
            paradigm=paradigm_name,
            task_id=task.id,
            top_module=task.top_module,
            level=task.level,
            functional_pass=(compile_pass and pass_rate == 1.0),
            test_pass_rate=pass_rate,
            formal_pass=(formal_status == "PASS"),
            synthesis_pass=(cells is not None and cells > 0),
            cells=cells,
            steps_taken=step_count,
            reward=float(eval_res.get("reward", 0.0)),
            error=str(eval_res.get("error", "")),
        )

    def aggregate_summary(self, paradigm_name: str, records: list[EvaluationMetricRecord]) -> ParadigmSummary:
        """Aggregate evaluation metrics across tasks for an evaluation paradigm."""
        if not records:
            return ParadigmSummary(paradigm=paradigm_name)

        n = len(records)
        func_pass = sum(1 for r in records if r.functional_pass) / n
        formal_pass = sum(1 for r in records if r.formal_pass) / n
        synth_pass = sum(1 for r in records if r.synthesis_pass) / n
        measured_cells = [r.cells for r in records if r.cells is not None]
        avg_cells = (sum(measured_cells) / len(measured_cells)) if measured_cells else 0.0
        avg_steps = sum(r.steps_taken for r in records) / n
        avg_reward = sum(r.reward for r in records) / n

        return ParadigmSummary(
            paradigm=paradigm_name,
            total_tasks=n,
            functional_pass_rate=round(func_pass, 4),
            formal_pass_rate=round(formal_pass, 4),
            synthesis_pass_rate=round(synth_pass, 4),
            average_cells=round(avg_cells, 2),
            average_steps=round(avg_steps, 2),
            average_reward=round(avg_reward, 4),
            records=records,
        )

    def generate_markdown_report(self, summaries: list[ParadigmSummary]) -> str:
        """Generate structured comparison report formatted in GitHub Markdown."""
        lines = [
            "# Hardware RL Baseline vs. Trained Model Evaluation Report",
            "",
            f"**Evaluation Mode**: `{self.evaluation_mode}`",
            "",
            "| Paradigm | Tasks | Functional Pass | Formal Pass | Synth Pass | Avg Cells | Avg Steps | Mean Reward |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for s in summaries:
            lines.append(
                f"| **{s.paradigm}** | {s.total_tasks} | {s.functional_pass_rate * 100:.1f}% | "
                f"{s.formal_pass_rate * 100:.1f}% | {s.synthesis_pass_rate * 100:.1f}% | "
                f"{s.average_cells:.1f} | {s.average_steps:.1f} | {s.average_reward:.3f} |"
            )

        lines.extend([
            "",
            "### Verification Criteria & Safety Gates",
            "- **Functional Pass**: 100% of public and held-out test vectors match expected outputs.",
            "- **Formal Pass**: Zero assertion violations detected by SymbiYosys formal model checker.",
            "- **Synthesis Pass**: Actual logic synthesis completed by Yosys (never estimated cells).",
            "- **Mean Reward**: Grounded tool reward derived strictly from genuine EDA outputs.",
        ])
        return "\n".join(lines)
