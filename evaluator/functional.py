"""
Functional Evaluator — correctness scoring.

Evaluates hardware designs on:
- Compilation success
- Lint pass/warnings
- Simulation test pass rate
"""

from __future__ import annotations

import logging
from typing import Any

from agent.schemas import FunctionalScore, CompileResult, LintResult, SimulationResult

logger = logging.getLogger(__name__)


class FunctionalEvaluator:
    """Evaluates functional correctness of hardware designs.

    Combines compile, lint, and simulation results into a
    single FunctionalScore normalized to [0, 1].
    """

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.lint_warnings_max = config.get("lint_warnings_max", 5)
        self.test_pass_rate_min = config.get("test_pass_rate_min", 0.95)

    def evaluate(
        self,
        compile_result: CompileResult | None = None,
        lint_result: LintResult | None = None,
        sim_result: SimulationResult | None = None,
    ) -> FunctionalScore:
        """Compute functional correctness score.

        Args:
            compile_result: Verilator compilation result
            lint_result: Verilator lint result
            sim_result: Simulation/cocotb result

        Returns:
            FunctionalScore with component scores
        """
        score = FunctionalScore()

        # Compilation
        if compile_result is not None:
            score.compile_pass = compile_result.success

        # Linting
        if lint_result is not None:
            score.lint_pass = lint_result.success
            score.lint_warnings = len(lint_result.warnings)

        # Simulation / Tests
        if sim_result is not None:
            score.tests_total = sim_result.tests_total
            score.tests_passed = sim_result.tests_passed
            score.test_pass_rate = sim_result.pass_rate

        logger.info(
            f"Functional score: {score.score:.3f} "
            f"(compile={'✓' if score.compile_pass else '✗'}, "
            f"lint={'✓' if score.lint_pass else '✗'}, "
            f"tests={score.test_pass_rate:.0%})"
        )

        return score

    def meets_threshold(self, score: FunctionalScore) -> bool:
        """Check if a functional score meets minimum quality thresholds."""
        if not score.compile_pass:
            return False
        if score.lint_warnings > self.lint_warnings_max:
            return False
        if score.test_pass_rate < self.test_pass_rate_min:
            return False
        return True
