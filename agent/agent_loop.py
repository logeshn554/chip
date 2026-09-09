"""
Agent Loop — the self-evolution cycle.

This is the main orchestration loop that ties together:
- Memory retrieval
- Qwen3-4B reasoning
- Action execution via the router
- Evaluation and reward computation
- Trajectory recording
"""

from __future__ import annotations

import logging
import time
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent.qwen import QwenClient
from agent.planner import Planner
from agent.action_router import ActionRouter
from agent.schemas import (
    ActionResult,
    ActionStatus,
    ActionType,
    AgentAction,
    AgentState,
    EvaluationResult,
    Episode,
    TrajectoryStep,
)

logger = logging.getLogger(__name__)
console = Console()


class AgentLoop:
    """Main agent loop — the self-evolving hardware design cycle.

    Orchestrates the complete flow:
        Retrieve Memory → Reason (Qwen) → Act (Tools) → Evaluate → Reward → Store → Learn

    Args:
        qwen: Qwen3-4B client
        planner: Task planner
        router: Action router with tools registered
        memory: Memory system (knowledge + experience + design)
        evaluator: Evaluation and reward engine
        trajectory_store: Trajectory recording store
        config: Agent configuration dict
    """

    def __init__(
        self,
        qwen: QwenClient,
        planner: Planner,
        router: ActionRouter,
        memory=None,
        evaluator=None,
        trajectory_store=None,
        config: dict[str, Any] | None = None,
    ):
        self.qwen = qwen
        self.planner = planner
        self.router = router
        self.memory = memory
        self.evaluator = evaluator
        self.trajectory_store = trajectory_store
        self.config = config or {}

        self.max_iterations = self.config.get("max_iterations", 50)
        self.max_retries = self.config.get("max_retries_per_step", 3)
        self.early_stop_reward = self.config.get("early_stop_reward", 0.95)

    # ── Main Entry Point ─────────────────────────────────────────────

    async def run(self, task: str) -> Episode:
        """Run the complete agent loop on a hardware design task.

        Args:
            task: Human-readable task description
                  (e.g., "Design a 4-bit ALU with add, subtract, AND, OR")

        Returns:
            Episode containing the full trajectory
        """
        console.print(Panel(f"[bold cyan]Task:[/] {task}", title="🧠 Hardware Agent"))

        # Initialize state
        state = AgentState(task=task)
        episode = Episode(task=task)
        retries = 0

        # Step 1: Retrieve relevant memory context
        context = await self._retrieve_context(task)

        # Step 2: Create a plan
        console.print("[bold yellow]📋 Planning...[/]")
        state.plan = await self.planner.decompose(task, context)
        self._display_plan(state.plan)

        # Step 3: Execute the loop
        while state.iteration < self.max_iterations:
            state.iteration += 1
            console.print(f"\n[bold]─── Iteration {state.iteration}/{self.max_iterations} ───[/]")

            # 3a. Build context for this iteration
            iter_context = self._build_iteration_context(state, context)

            # 3b. Ask Qwen for the next action
            action = await self._get_next_action(state, iter_context)
            state.last_action = action

            # Check for completion
            if action.action_type == ActionType.COMPLETE:
                console.print("[bold green]✓ Agent decided task is complete[/]")
                break

            # 3c. Execute the action
            console.print(f"  [cyan]Action:[/] {action.action_type.value}")
            if action.thinking:
                console.print(f"  [dim]Thinking: {action.thinking[:200]}...[/]")

            result = await self.router.execute(action)
            state.last_result = result

            # 3d. Record the step
            step_reward = 0.0
            if result.status == ActionStatus.SUCCESS:
                console.print(f"  [green]✓ Success[/]")
                retries = 0
            elif result.status == ActionStatus.FAILURE:
                console.print(f"  [red]✗ Failed: {result.output[:200]}[/]")
                retries += 1
            else:
                console.print(f"  [yellow]⚠ {result.status.value}: {result.output[:200]}[/]")
                retries += 1

            # 3e. Stage-driven Evaluation: evaluate upon design generation, simulation, synthesis, or optimization
            should_evaluate = bool(
                result.artifacts
                or action.action_type in (ActionType.SIMULATE, ActionType.SYNTHESIZE, ActionType.GENERATE_RTL, ActionType.OPTIMIZE)
            )
            if should_evaluate and self.evaluator:
                eval_result = await self._evaluate(state)
                step_reward = eval_result.reward
                state.current_design_reward = eval_result.reward
                state.cumulative_reward += step_reward
                state.episode_return += step_reward
                state.best_reward = max(state.best_reward, eval_result.reward)

                # Record stage-specific score
                if action.action_type == ActionType.SIMULATE:
                    state.verification_stages["functional"] = eval_result.breakdown
                elif action.action_type == ActionType.SYNTHESIZE:
                    state.verification_stages["synthesis"] = eval_result.breakdown

                self._display_evaluation(eval_result)

            # 3f. Record trajectory step
            traj_step = TrajectoryStep(
                step_index=state.iteration,
                state_summary=self._summarize_state(state),
                action=action.action_type.value,
                action_params=action.params,
                observation=result.output[:1000],
                reward=step_reward,
            )
            episode.steps.append(traj_step)

            # 3g. Update memory with results
            await self._update_memory(action, result)

            # 3h. Store design artifacts
            if result.artifacts:
                state.design_files.update(result.artifacts)

            # 3i. Update plan step status
            if state.plan and state.plan.next_step:
                step = state.plan.next_step
                step.status = "completed" if result.status == ActionStatus.SUCCESS else "failed"
                step.result = result

            # 3j. Check early stopping based on best design quality achieved
            if state.best_reward >= self.early_stop_reward:
                console.print(
                    f"[bold green]🎯 Early stop — best reward {state.best_reward:.3f} "
                    f">= {self.early_stop_reward}[/]"
                )
                break

            # 3k. Check retry limit
            if retries >= self.max_retries:
                console.print("[bold red]💥 Max retries reached, replanning...[/]")
                error_context = result.output if result else "Unknown error"
                state.plan = await self.planner.replan(state.plan, error_context)
                retries = 0

            # 3l. Add to history
            state.history.append({
                "iteration": state.iteration,
                "action": action.action_type.value,
                "status": result.status.value,
                "step_reward": step_reward,
                "current_design_reward": state.current_design_reward,
                "best_reward": state.best_reward,
            })

        # Finalize episode
        episode.final_reward = state.current_design_reward
        episode.best_reward = state.best_reward
        episode.episode_return = state.episode_return
        episode.total_iterations = state.iteration
        episode.success = state.best_reward >= self.early_stop_reward
        episode.completed_at = time.time()

        # Save trajectory
        if self.trajectory_store:
            await self.trajectory_store.save_episode(episode)

        # Final summary
        self._display_summary(episode)

        return episode

    # ── Context Building ─────────────────────────────────────────────

    async def _retrieve_context(self, task: str) -> str:
        """Retrieve relevant context from all memory sources."""
        if self.memory is None:
            return ""

        try:
            docs = await self.memory.query(task, n=5)
            if not docs:
                return ""

            sections = []
            for doc in docs:
                sections.append(f"[{doc.source}] (relevance: {doc.score:.3f})\n{doc.content}")

            return "\n\n---\n\n".join(sections)
        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")
            return ""

    def _build_iteration_context(self, state: AgentState, base_context: str) -> str:
        """Build the full context for the current iteration."""
        parts = [f"Task: {state.task}"]

        if base_context:
            parts.append(f"Relevant Knowledge:\n{base_context}")

        if state.plan:
            from agent.planner import Planner
            parts.append(f"Current Plan:\n{Planner._plan_to_text(state.plan)}")

        if state.last_result:
            parts.append(
                f"Last Action Result:\n"
                f"  Action: {state.last_action.action_type.value if state.last_action else 'N/A'}\n"
                f"  Status: {state.last_result.status.value}\n"
                f"  Output: {state.last_result.output[:500]}"
            )

        if state.design_files:
            parts.append(f"Current Design Files: {list(state.design_files.keys())}")

        parts.append(f"Iteration: {state.iteration}/{self.max_iterations}")
        parts.append(f"Cumulative Reward: {state.cumulative_reward:.3f}")

        return "\n\n".join(parts)

    # ── LLM Interaction ──────────────────────────────────────────────

    async def _get_next_action(self, state: AgentState, context: str) -> AgentAction:
        """Ask Qwen for the next action given the current context."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a hardware design agent. Based on the current state, "
                    "decide the next action. Respond with a JSON object containing "
                    "'thinking', 'action', and 'params' fields."
                ),
            },
            {"role": "user", "content": context},
        ]

        result = await self.qwen.generate_structured(messages)
        return self.router.parse_action(result)

    # ── Evaluation ───────────────────────────────────────────────────

    async def _evaluate(self, state: AgentState) -> EvaluationResult:
        """Run evaluation on current design artifacts."""
        if self.evaluator is None:
            return EvaluationResult(reward=0.0)

        try:
            return await self.evaluator.evaluate(state.design_files)
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return EvaluationResult(reward=0.0)

    # ── Memory Updates ───────────────────────────────────────────────

    async def _update_memory(self, action: AgentAction, result: ActionResult) -> None:
        """Store action results in experience memory."""
        if self.memory is None:
            return

        try:
            # Record experience (especially errors and fixes)
            if result.status == ActionStatus.FAILURE and result.errors:
                await self.memory.record_experience(
                    task=action.action_type.value,
                    action=str(action.params),
                    result=result.output,
                    errors=result.errors,
                    success=False,
                )
            elif result.status == ActionStatus.SUCCESS:
                await self.memory.record_experience(
                    task=action.action_type.value,
                    action=str(action.params),
                    result=result.output[:500],
                    errors=[],
                    success=True,
                )

            # Store successful designs
            if (
                action.action_type == ActionType.GENERATE_RTL
                and result.status == ActionStatus.SUCCESS
            ):
                await self.memory.save_design(
                    name=action.params.get("name", "design"),
                    code=result.output,
                    metadata={"action_params": action.params},
                )
        except Exception as e:
            logger.warning(f"Memory update failed: {e}")

    # ── Display ──────────────────────────────────────────────────────

    def _display_plan(self, plan) -> None:
        """Display the plan in a Rich table."""
        table = Table(title="Design Plan", show_lines=True)
        table.add_column("ID", style="cyan", width=4)
        table.add_column("Step", style="white")
        table.add_column("Type", style="green")
        table.add_column("Deps", style="yellow")
        table.add_column("Status", style="magenta")

        for step in plan.steps:
            table.add_row(
                str(step.id),
                step.description[:60],
                step.step_type,
                str(step.depends_on) if step.depends_on else "-",
                step.status,
            )

        console.print(table)

    def _display_evaluation(self, eval_result: EvaluationResult) -> None:
        """Display evaluation results."""
        table = Table(title="Evaluation", show_lines=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        for key, value in eval_result.reward_breakdown.items():
            table.add_row(key, f"{value:.3f}")

        table.add_row("[bold]Total Reward[/]", f"[bold]{eval_result.reward:.3f}[/]")
        console.print(table)

    def _display_summary(self, episode: Episode) -> None:
        """Display final episode summary."""
        status = "[green]SUCCESS[/]" if episode.success else "[red]INCOMPLETE[/]"
        console.print(Panel(
            f"Status: {status}\n"
            f"Iterations: {episode.total_iterations}\n"
            f"Final Reward: {episode.final_reward:.3f}\n"
            f"Duration: {episode.duration_s:.1f}s\n"
            f"Steps Recorded: {len(episode.steps)}",
            title="📊 Episode Summary",
        ))

    @staticmethod
    def _summarize_state(state: AgentState) -> str:
        """Create a brief state summary for trajectory recording."""
        parts = [
            f"iter={state.iteration}",
            f"files={list(state.design_files.keys())}",
            f"reward={state.cumulative_reward:.3f}",
        ]
        if state.plan:
            parts.append(f"plan_progress={state.plan.progress:.0%}")
        return " | ".join(parts)
