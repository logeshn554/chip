"""
Action Router — maps agent decisions to tool execution.

Parses the LLM's structured JSON action output and routes it
to the appropriate tool handler (RTL generator, EDA tools,
web search, memory, etc.).
"""

from __future__ import annotations

import logging
from typing import Any

from agent.schemas import (
    ActionResult,
    ActionStatus,
    ActionType,
    AgentAction,
)

logger = logging.getLogger(__name__)


class ActionRouter:
    """Routes agent actions to the appropriate tool handlers.

    Each handler is registered by action type and receives the
    action parameters. Handlers are injected at construction time,
    allowing the router to be tested with mock handlers.
    """

    def __init__(self):
        self._handlers: dict[ActionType, Any] = {}

    def register(self, action_type: ActionType, handler: Any) -> None:
        """Register a handler for an action type.

        The handler must be a callable (sync or async) that accepts
        an `ActionResult`-returning signature:
            async def handler(params: dict) -> ActionResult
        """
        self._handlers[action_type] = handler
        logger.debug(f"Registered handler for {action_type.value}")

    # ── Action Parsing ───────────────────────────────────────────────

    def parse_action(self, llm_output: dict[str, Any]) -> AgentAction:
        """Parse an LLM JSON response into a typed AgentAction.

        Expected LLM output format:
        {
            "thinking": "...",
            "action": "GENERATE_RTL",
            "params": {"spec": "...", ...}
        }
        """
        action_str = llm_output.get("action", "COMPLETE")
        thinking = llm_output.get("thinking", "")
        params = llm_output.get("params", {})

        try:
            action_type = ActionType(action_str.upper())
        except ValueError:
            logger.warning(f"Unknown action: {action_str}, defaulting to COMPLETE")
            action_type = ActionType.COMPLETE

        return AgentAction(
            action_type=action_type,
            params=params,
            thinking=thinking,
        )

    # ── Action Execution ─────────────────────────────────────────────

    async def execute(self, action: AgentAction) -> ActionResult:
        """Execute an action by routing to its registered handler.

        Args:
            action: Parsed agent action with type and parameters

        Returns:
            ActionResult with status, output, and artifacts
        """
        logger.info(f"Executing action: {action.action_type.value}")

        if action.action_type == ActionType.COMPLETE:
            return ActionResult(
                action=ActionType.COMPLETE,
                status=ActionStatus.SUCCESS,
                output="Task marked as complete by agent.",
            )

        handler = self._handlers.get(action.action_type)
        if handler is None:
            logger.error(f"No handler registered for {action.action_type.value}")
            return ActionResult(
                action=action.action_type,
                status=ActionStatus.ERROR,
                output=f"No handler for action: {action.action_type.value}",
                errors=[f"Unregistered action type: {action.action_type.value}"],
            )

        try:
            import asyncio
            import inspect

            if inspect.iscoroutinefunction(handler):
                result = await handler(action.params)
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, handler, action.params
                )

            # Ensure we got an ActionResult back
            if not isinstance(result, ActionResult):
                result = ActionResult(
                    action=action.action_type,
                    status=ActionStatus.SUCCESS,
                    output=str(result),
                )

            return result

        except TimeoutError:
            logger.error(f"Timeout executing {action.action_type.value}")
            return ActionResult(
                action=action.action_type,
                status=ActionStatus.TIMEOUT,
                output="Action timed out",
                errors=["Execution timeout"],
            )
        except Exception as e:
            logger.error(f"Error executing {action.action_type.value}: {e}", exc_info=True)
            return ActionResult(
                action=action.action_type,
                status=ActionStatus.ERROR,
                output=str(e),
                errors=[str(e)],
            )


# ── Tool Handler Factories ───────────────────────────────────────────
# These connect the abstract ActionRouter to concrete tool implementations.


def build_router(
    rtl_generator=None,
    web_searcher=None,
    memory_system=None,
    verilator=None,
    cocotb=None,
    yosys=None,
) -> ActionRouter:
    """Build an ActionRouter with all tool handlers wired up.

    Any tool can be None — the router will return an error for
    actions that target unregistered tools.
    """
    router = ActionRouter()

    # ── RTL Generation ───────────────────────────────────────────
    if rtl_generator is not None:
        async def handle_generate_rtl(params: dict) -> ActionResult:
            spec = params.get("spec", params.get("description", ""))
            context_docs = params.get("context", [])
            module = await rtl_generator.generate_module(spec, context_docs)
            return ActionResult(
                action=ActionType.GENERATE_RTL,
                status=ActionStatus.SUCCESS,
                output=module.code,
                artifacts={"rtl": module.filepath or "generated.sv"},
            )

        router.register(ActionType.GENERATE_RTL, handle_generate_rtl)

        async def handle_generate_tb(params: dict) -> ActionResult:
            module_code = params.get("module_code", "")
            tb_code = await rtl_generator.generate_testbench_code(module_code)
            return ActionResult(
                action=ActionType.GENERATE_TESTBENCH,
                status=ActionStatus.SUCCESS,
                output=tb_code,
                artifacts={"testbench": "test_generated.py"},
            )

        router.register(ActionType.GENERATE_TESTBENCH, handle_generate_tb)

        async def handle_debug(params: dict) -> ActionResult:
            code = params.get("code", "")
            errors = params.get("errors", [])
            fixed = await rtl_generator.fix_errors(code, errors)
            return ActionResult(
                action=ActionType.DEBUG,
                status=ActionStatus.SUCCESS,
                output=fixed,
            )

        router.register(ActionType.DEBUG, handle_debug)

        async def handle_optimize(params: dict) -> ActionResult:
            code = params.get("code", "")
            opt_type = params.get("optimize_for", "area")
            metrics = params.get("current_metrics", {})
            optimized = await rtl_generator.optimize(code, opt_type, metrics)
            return ActionResult(
                action=ActionType.OPTIMIZE,
                status=ActionStatus.SUCCESS,
                output=optimized,
            )

        router.register(ActionType.OPTIMIZE, handle_optimize)

    # ── Targeted Web Research (ScrapeGraphAI Pipeline) ───────────
    async def handle_search(params: dict) -> ActionResult:
        query = params.get("query", "")
        url = params.get("url")

        from scraping.scrapegraph_adapter import ScrapeGraphAdapter
        adapter = ScrapeGraphAdapter()
        target_url = url or "https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation"

        ctx = await adapter.extract_compact_context(url=target_url, focused_query=query)
        output = ctx.to_prompt_text()

        # Ingest extracted knowledge into memory if available
        if memory_system is not None and hasattr(memory_system, "ingest"):
            try:
                memory_system.ingest(
                    ctx.extracted_summary,
                    metadata={"source": ctx.source_url, "title": ctx.title},
                )
            except Exception as e:
                logger.debug(f"Knowledge ingestion skipped: {e}")

        return ActionResult(
            action=ActionType.SEARCH,
            status=ActionStatus.SUCCESS,
            output=output,
            metrics={"source_url": ctx.source_url, "token_count": ctx.token_count},
        )

    router.register(ActionType.SEARCH, handle_search)

    # ── Memory Inspection (3-Tier Separation) ────────────────────
    if memory_system is not None:
        async def handle_memory(params: dict) -> ActionResult:
            query = params.get("query", "")
            memory_type = params.get("type", "knowledge")  # "knowledge", "experience", "design"
            docs = await memory_system.query(query, memory_type=memory_type)
            output = "\n\n---\n\n".join(
                f"[{d.source}] (score: {d.score:.3f})\n{d.content}" for d in docs
            )
            return ActionResult(
                action=ActionType.INSPECT_MEMORY,
                status=ActionStatus.SUCCESS,
                output=output,
                metrics={"docs_retrieved": len(docs), "memory_type": memory_type},
            )

        router.register(ActionType.INSPECT_MEMORY, handle_memory)

    # ── Simulation (Strict Parameter Validation) ──────────────────
    if verilator is not None:
        async def handle_simulate(params: dict) -> ActionResult:
            sources = params.get("sources", [])
            top_module = params.get("top_module", "")

            # Validate parameter types and file paths (prevent arbitrary shell or unsafe paths)
            if not isinstance(sources, list) or not sources:
                return ActionResult(
                    action=ActionType.SIMULATE,
                    status=ActionStatus.FAILURE,
                    output="Parameter validation error: 'sources' must be a non-empty list of .sv/.v files.",
                    errors=["Invalid parameter: sources must be non-empty list"],
                )
            for src in sources:
                if not isinstance(src, str) or not src.endswith((".sv", ".v")):
                    return ActionResult(
                        action=ActionType.SIMULATE,
                        status=ActionStatus.FAILURE,
                        output=f"Security/Validation error: '{src}' is not a valid .sv or .v source file.",
                        errors=[f"Untrusted or invalid source file: {src}"],
                    )

            compile_res = await verilator.compile(sources)
            if not compile_res.success:
                return ActionResult(
                    action=ActionType.SIMULATE,
                    status=ActionStatus.FAILURE,
                    output=compile_res.output,
                    errors=compile_res.errors,
                )

            sim_res = await verilator.simulate(top_module)
            return ActionResult(
                action=ActionType.SIMULATE,
                status=ActionStatus.SUCCESS if sim_res.success else ActionStatus.FAILURE,
                output=sim_res.output,
                errors=sim_res.errors,
                metrics={
                    "tests_total": sim_res.tests_total,
                    "tests_passed": sim_res.tests_passed,
                    "pass_rate": sim_res.pass_rate,
                },
            )

        router.register(ActionType.SIMULATE, handle_simulate)

    # ── Synthesis (Strict Parameter Validation) ───────────────────
    if yosys is not None:
        async def handle_synthesize(params: dict) -> ActionResult:
            sources = params.get("sources", [])
            target = params.get("target", "generic")

            if not isinstance(sources, list) or not sources:
                return ActionResult(
                    action=ActionType.SYNTHESIZE,
                    status=ActionStatus.FAILURE,
                    output="Parameter validation error: 'sources' must be a non-empty list of .sv/.v files.",
                    errors=["Invalid parameter: sources must be non-empty list"],
                )
            for src in sources:
                if not isinstance(src, str) or not src.endswith((".sv", ".v")):
                    return ActionResult(
                        action=ActionType.SYNTHESIZE,
                        status=ActionStatus.FAILURE,
                        output=f"Security/Validation error: '{src}' is not a valid .sv or .v source file.",
                        errors=[f"Untrusted or invalid source file: {src}"],
                    )

            synth_res = await yosys.synthesize(sources, target)
            return ActionResult(
                action=ActionType.SYNTHESIZE,
                status=ActionStatus.SUCCESS if synth_res.success else ActionStatus.FAILURE,
                output=synth_res.output,
                errors=synth_res.errors,
                metrics={
                    "cell_count": synth_res.cell_count,
                    "wire_count": synth_res.wire_count,
                    "critical_path_ns": synth_res.critical_path_ns,
                },
            )

        router.register(ActionType.SYNTHESIZE, handle_synthesize)

    logger.info(f"ActionRouter built with {len(router._handlers)} handlers")
    return router
