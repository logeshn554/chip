"""
Agent Loop — the self-evolution cycle.

This is the main orchestration loop that ties together:
- Memory retrieval
- Qwen-14B reasoning
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
from agent.action_router import ActionRouter, build_router
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
        qwen: Qwen-14B client
        planner: Task planner
        router: Action router with tools registered
        memory: Memory system (knowledge + experience + design)
        evaluator: Evaluation and reward engine
        trajectory_store: Trajectory recording store
        config: Agent configuration dict
    """

    def __init__(
        self,
        qwen: QwenClient | None = None,
        planner: Planner | None = None,
        router: ActionRouter | None = None,
        memory=None,
        evaluator=None,
        trajectory_store=None,
        config: dict[str, Any] | None = None,
    ):
        self.config = config or {}
        self.qwen = qwen if qwen is not None else QwenClient(self.config)
        self.planner = planner if planner is not None else Planner()
        self.router = router if router is not None else build_router()
        self.memory = memory
        self.evaluator = evaluator
        self.trajectory_store = trajectory_store

        self.max_iterations = self.config.get("max_iterations", 50)
        self.max_retries = self.config.get("max_retries_per_step", 3)
        self.early_stop_reward = self.config.get("early_stop_reward", 0.95)

    def _log_observability(self, event_type: str, data: dict[str, Any], experiment_id: str = "") -> None:
        """Standardized structured observability logging across all agent runs (Issue 53)."""
        record = {
            "event": event_type,
            "timestamp": time.time(),
            "experiment_id": experiment_id,
            "data": data,
        }
        logger.info(f"[OBSERVABILITY] {event_type} -> {json.dumps(record, default=str)}")

    # ── Main Entry Point ─────────────────────────────────────────────

    async def run(
        self,
        task: str,
        experiment_id: str | None = None,
        seed: int | None = None,
        max_iterations: int | None = None,
    ) -> Episode:
        """Run the complete agent loop on a hardware design task.

        Args:
            task: Human-readable task description
                  (e.g., "Design a 4-bit ALU with add, subtract, AND, OR")
            experiment_id: Optional immutable experiment ID
            seed: Optional centralized random seed for reproducibility
            max_iterations: Optional override for max loop iterations

        Returns:
            Episode containing the full trajectory
        """
        import random
        import uuid
        from evaluator.physical_feasibility import PhysicalFeasibilityEngine
        from agent.schemas import FeasibilityLabel, TargetSpecification

        effective_max_iterations = max_iterations if max_iterations is not None else self.max_iterations

        # Issue 56: Centralized seeding
        run_seed = seed if seed is not None else self.config.get("seed", 42)
        random.seed(run_seed)

        # Issue 54: Immutable experiment ID threading
        exp_id = experiment_id or self.config.get("experiment_id") or f"exp_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        console.print(Panel(f"[bold cyan]Task:[/] {task}\n[dim]Experiment ID: {exp_id} | Seed: {run_seed}[/]", title="🧠 Hardware Agent"))

        # Initialize state
        state = AgentState(task=task)
        episode = Episode(task=task, experiment_id=exp_id, seed=run_seed)
        retries = 0

        # Step 1: Retrieve relevant memory context
        context = await self._retrieve_context(task)

        # Step 2: Create a plan
        console.print("[bold yellow]📋 Planning...[/]")
        state.plan = await self.planner.decompose(task, context)
        self._display_plan(state.plan)

        # Step 2b: Execute targeted initial research contract if planned by LLM
        if state.plan and getattr(state.plan, "needs_research", False) and getattr(state.plan, "search_queries", None):
            console.print("[bold cyan]🔍 Executing targeted research plan...[/]")
            for query in state.plan.search_queries[:2]:
                research_action = AgentAction(
                    action_type=ActionType.SEARCH_WEB,
                    params={"query": query},
                    thinking=f"Executing research contract query: {query}",
                )
                r_result = await self.router.execute(research_action)
                if r_result.status == ActionStatus.SUCCESS and r_result.output:
                    context = f"{context}\n\n[Research Evidence - {query}]\n{r_result.output[:1000]}"

        # Step 3: Execute the loop
        while state.iteration < effective_max_iterations:
            state.iteration += 1
            console.print(f"\n[bold]─── Iteration {state.iteration}/{effective_max_iterations} ───[/]")

            # 3a. Build context for this iteration
            iter_context = self._build_iteration_context(state, context)

            # 3b. Ask Qwen for the next action
            action = await self._get_next_action(state, iter_context)
            # Check for completion with single mandatory end-to-end measured-success contract
            if action.action_type == ActionType.COMPLETE:
                failed_contract_gates = []

                # Gate 1: Synthesizable RTL exists and non-empty
                has_rtl = bool(state.design_files and any(f.endswith((".sv", ".v")) for f in state.design_files))
                if not has_rtl:
                    failed_contract_gates.append("Missing synthesizable RTL (.sv or .v source)")

                # Gate 2: Functional simulation verification pass
                has_sim = "functional" in state.verification_stages
                if not has_sim:
                    failed_contract_gates.append("Missing functional simulation pass (Cocotb)")

                # Gate 3: Logic synthesis with genuine cell count > 0
                synth_data = state.verification_stages.get("synthesis", {})
                has_synth = "synthesis" in state.verification_stages
                synth_cells = 0
                if isinstance(synth_data, dict):
                    synth_cells = synth_data.get("cells", synth_data.get("cell_count", 0)) or 0
                elif hasattr(synth_data, "cell_count"):
                    synth_cells = synth_data.cell_count
                if not has_synth or synth_cells <= 0:
                    failed_contract_gates.append("Missing logic synthesis pass with measured cells > 0 (Yosys)")

                # Gate 4: Formal verification check
                has_formal = "formal" in state.verification_stages
                if not has_formal:
                    failed_contract_gates.append("Missing formal verification assertion check (SymbiYosys)")

                # Gate 5: Physical feasibility envelope check
                from evaluator.physical_feasibility import PhysicalFeasibilityEngine
                from agent.schemas import TargetSpecification, FeasibilityLabel, HardwareArchitectureCandidate
                phys_target = TargetSpecification()
                phys_engine = PhysicalFeasibilityEngine(phys_target)
                active_cand = getattr(state, "active_candidate", None)
                if active_cand is None:
                    active_cand = HardwareArchitectureCandidate(
                        architecture_id=f"cand_eval_iter_{state.iteration}",
                        task_id=state.task[:30].replace(" ", "_"),
                        actual_synthesis_metrics={"cells": synth_cells} if synth_cells else {},
                        parallelism=2,
                        pipeline_depth=2,
                    )
                phys_report = phys_engine.evaluate_candidate(active_cand)
                if phys_report.feasibility == FeasibilityLabel.INFEASIBLE_ESTIMATE:
                    failed_contract_gates.append(f"Physical feasibility violated: {', '.join(phys_report.violations)}")

                if failed_contract_gates:
                    console.print(f"[bold red]❌ Single Mandatory Completion Contract Rejected ({len(failed_contract_gates)} gates failed):[/]")
                    for gate_err in failed_contract_gates:
                        console.print(f"  [red]• {gate_err}[/]")
                    result = ActionResult(
                        action=ActionType.COMPLETE,
                        status=ActionStatus.FAILURE,
                        output=f"Completion rejected by mandatory contract: {'; '.join(failed_contract_gates)}",
                        errors=failed_contract_gates,
                    )
                    state.last_result = result
                    if any("Physical" in g for g in failed_contract_gates):
                        retries = self.max_retries  # Force architectural redesign escalation
                    else:
                        retries += 1
                    continue

                console.print("[bold green]✓ Single mandatory measured-success contract satisfied! Design verified end-to-end.[/]")
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
                console.print(f"  [green][OK] Success[/]")
                retries = 0
            elif result.status == ActionStatus.FAILURE:
                console.print(f"  [red][FAIL] Failed: {result.output[:200]}[/]")
                retries += 1
            else:
                console.print(f"  [yellow][WARN] {result.status.value}: {result.output[:200]}[/]")
                retries += 1


            # 3e. Stage-driven Evaluation: evaluate upon design generation, simulation, synthesis, or optimization
            should_evaluate = bool(
                result.artifacts
                or action.action_type in (ActionType.SIMULATE, ActionType.SYNTHESIZE, ActionType.GENERATE_RTL, ActionType.OPTIMIZE)
            )
            if should_evaluate and self.evaluator:
                eval_result = await self._evaluate(state)
                step_reward = eval_result.reward
                improvement = max(0.0, eval_result.reward - state.current_design_reward)
                state.current_design_reward = eval_result.reward
                state.cumulative_reward += improvement
                state.episode_return += improvement
                state.best_reward = max(state.best_reward, eval_result.reward)

                # Record stage-specific score
                if action.action_type in (ActionType.SIMULATE, ActionType.RUN_SIMULATION, ActionType.RUN_TESTS):
                    state.verification_stages["functional"] = eval_result.breakdown
                elif action.action_type in (ActionType.SYNTHESIZE, ActionType.RUN_YOSYS):
                    state.verification_stages["synthesis"] = eval_result.breakdown
                elif action.action_type == ActionType.FORMAL_VERIFY:
                    state.verification_stages["formal"] = eval_result.breakdown

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

            # 3j. Check early stopping based on normalized design quality achieved
            normalized_best = state.best_reward / 8.0 if state.best_reward > 1.0 else state.best_reward
            has_valid_rtl = bool(state.design_files and any(f.endswith((".sv", ".v")) for f in state.design_files))
            if (
                normalized_best >= self.early_stop_reward
                and has_valid_rtl
                and "functional" in state.verification_stages
                and "synthesis" in state.verification_stages
            ):
                console.print(
                    f"[bold green]🎯 Early stop — normalized best reward {normalized_best:.3f} "
                    f">= {self.early_stop_reward} with valid verified RTL[/]"
                )
                break

            # 3k. Check retry limit: separate RTL repair from architecture redesign
            if retries >= self.max_retries:
                console.print("[bold red]💥 Max retries reached: escalating failure to architectural mutation...[/]")
                error_context = result.output if result else "Unknown error"

                from agent.architecture_search import ArchitectureSearchEngine
                from agent.schemas import TargetSpecification, HardwareArchitectureCandidate
                arch_engine = ArchitectureSearchEngine()
                failed_cand = getattr(state, "active_candidate", None) or HardwareArchitectureCandidate(
                    architecture_id=f"cand_iter_{state.iteration}",
                    task_id=state.task[:20].replace(" ", "_"),
                    parallelism=2,
                )
                failed_reasons = []
                err_lower = error_context.lower()
                if "timing" in err_lower:
                    failed_reasons.append("timing")
                elif "power" in err_lower or "thermal" in err_lower:
                    failed_reasons.append("power")
                elif "area" in err_lower or "size" in err_lower:
                    failed_reasons.append("area")
                else:
                    failed_reasons.append("synthesis_syntax_exhaustion")

                mutated = arch_engine.evolve_after_rejection(failed_cand, failed_reasons, TargetSpecification())
                arch_engine.genealogy.register_candidate(failed_cand)
                arch_engine.genealogy.register_candidate(
                    mutated,
                    parent_id=failed_cand.architecture_id,
                    mutation_type=mutated.mutation_type,
                    rationale=f"Mutated after failure: {error_context[:100]}",
                )
                state.active_candidate = mutated
                console.print(f"[bold magenta]🧬 Architecture evolved to: {mutated.architecture_id} via {mutated.mutation_type}[/]")
                state.plan = await self.planner.replan(state.plan, f"Architecture mutated ({mutated.mutation_type}): {error_context[:100]}")
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
