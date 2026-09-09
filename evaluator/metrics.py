"""
Metrics Tracker — aggregates and tracks evaluation metrics over time.

Provides historical tracking, comparison, and export of
design quality metrics across iterations.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from typing import Any

from agent.schemas import EvaluationResult

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Tracks and aggregates evaluation metrics over design iterations.

    Maintains a history of evaluation results for trend analysis,
    comparison against baselines, and export to CSV/JSON.
    """

    def __init__(self, output_dir: str = "./data/metrics"):
        self.output_dir = output_dir
        self.history: list[dict[str, Any]] = []

        os.makedirs(output_dir, exist_ok=True)

        # Load existing history if available
        history_path = os.path.join(output_dir, "metrics_history.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                self.history = json.load(f)
            logger.info(f"Loaded {len(self.history)} historical metric entries")

    def record(
        self,
        eval_result: EvaluationResult,
        design_name: str = "",
        iteration: int = 0,
    ) -> None:
        """Record an evaluation result.

        Args:
            eval_result: The evaluation result to record
            design_name: Name of the design
            iteration: Iteration number
        """
        entry = {
            "timestamp": time.time(),
            "design_name": design_name,
            "iteration": iteration,
            "reward": eval_result.reward,
            "reward_breakdown": eval_result.reward_breakdown,
            "functional": {
                "compile_pass": eval_result.functional.compile_pass,
                "lint_pass": eval_result.functional.lint_pass,
                "lint_warnings": eval_result.functional.lint_warnings,
                "test_pass_rate": eval_result.functional.test_pass_rate,
                "tests_total": eval_result.functional.tests_total,
                "tests_passed": eval_result.functional.tests_passed,
            },
            "synthesis": {
                "synthesizable": eval_result.synthesis.synthesizable,
                "area_score": eval_result.synthesis.area_score,
                "timing_score": eval_result.synthesis.timing_score,
                "power_score": eval_result.synthesis.power_score,
            },
        }

        self.history.append(entry)
        self._save()

        logger.debug(f"Recorded metrics: {design_name} iter={iteration} reward={eval_result.reward:.3f}")

    def get_trend(self, metric: str = "reward", last_n: int = 50) -> list[float]:
        """Get the trend of a specific metric over recent iterations.

        Args:
            metric: Metric name ("reward", "test_pass_rate", etc.)
            last_n: Number of recent entries

        Returns:
            List of metric values
        """
        entries = self.history[-last_n:]

        if metric == "reward":
            return [e["reward"] for e in entries]
        elif metric in ("test_pass_rate", "compile_pass", "lint_pass"):
            return [e["functional"].get(metric, 0.0) for e in entries]
        elif metric in ("area_score", "timing_score", "power_score"):
            return [e["synthesis"].get(metric, 0.0) for e in entries]
        else:
            return []

    def get_best(self, metric: str = "reward") -> dict[str, Any] | None:
        """Get the entry with the best value for a given metric."""
        if not self.history:
            return None

        trend = self.get_trend(metric, len(self.history))
        if not trend:
            return None

        best_idx = max(range(len(trend)), key=lambda i: trend[i])
        return self.history[best_idx]

    def compare_to_baseline(
        self,
        current: EvaluationResult,
        baseline_name: str = "best",
    ) -> dict[str, float]:
        """Compare current evaluation to a baseline.

        Args:
            current: Current evaluation result
            baseline_name: "best" or "latest"

        Returns:
            Dict of metric deltas (positive = improvement)
        """
        if baseline_name == "best":
            baseline = self.get_best()
        else:
            baseline = self.history[-1] if self.history else None

        if baseline is None:
            return {"reward": current.reward}

        return {
            "reward": current.reward - baseline["reward"],
            "test_pass_rate": (
                current.functional.test_pass_rate
                - baseline["functional"].get("test_pass_rate", 0.0)
            ),
            "area_score": (
                current.synthesis.area_score
                - baseline["synthesis"].get("area_score", 0.0)
            ),
        }

    def export_csv(self, filepath: str | None = None) -> str:
        """Export metrics history to CSV.

        Returns:
            Path to the exported CSV file
        """
        filepath = filepath or os.path.join(self.output_dir, "metrics.csv")

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "design_name", "iteration", "reward",
                "compile_pass", "lint_pass", "test_pass_rate",
                "area_score", "timing_score", "power_score",
            ])

            for entry in self.history:
                writer.writerow([
                    entry["timestamp"],
                    entry["design_name"],
                    entry["iteration"],
                    entry["reward"],
                    entry["functional"]["compile_pass"],
                    entry["functional"]["lint_pass"],
                    entry["functional"]["test_pass_rate"],
                    entry["synthesis"]["area_score"],
                    entry["synthesis"]["timing_score"],
                    entry["synthesis"]["power_score"],
                ])

        logger.info(f"Exported {len(self.history)} entries to {filepath}")
        return filepath

    def _save(self) -> None:
        """Save history to disk."""
        path = os.path.join(self.output_dir, "metrics_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
