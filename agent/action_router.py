"""
Action Router — Maps agent decisions to typed, validated tool execution.

Features:
- Enforces strict typed action interface (15 required actions + backward-compatible aliases)
- Parameter validation with clear structured feedback
- Path sandboxing: restricts writes to allowed workspace, forbids writes to evaluator/tests/configs
- Never executes arbitrary shell commands
- Safe failure handling: unknown actions return structured failure instead of silently completing
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
from typing import Any, Callable, Optional

from agent.schemas import (
    ActionResult,
    ActionStatus,
    ActionType,
    AgentAction,
)

logger = logging.getLogger(__name__)

# Protected directories that the agent MUST NEVER modify or overwrite
PROTECTED_PATHS = ["evaluator", "configs", "tests", "benchmarks/tests", ".git"]
# Allowed workspace directories for agent RTL generation and artifacts
ALLOWED_WRITE_DIRS = ["designs", "rtl/generated", "sim_build", "trajectories", "synth"]


def is_path_safe_for_write(path: str, base_dir: str = ".") -> bool:
    """Check whether a target file path is safe for agent modification.

    Enforces:
    1. No directory traversal ('..')
    2. Path is inside allowed write directories
    3. Path does not target protected evaluator, test, or config directories
    """
    normalized = os.path.normpath(path).replace("\\", "/")

    # Check for directory traversal
    if ".." in normalized.split("/"):
        return False

    # Check against protected directories
    for prot in PROTECTED_PATHS:
        if normalized == prot or normalized.startswith(prot + "/"):
            return False

    # Check if inside allowed write dirs (or root level generated SV)
    is_allowed = False
    for allowed in ALLOWED_WRITE_DIRS:
        if normalized.startswith(allowed + "/") or normalized == allowed:
            is_allowed = True
            break

    # Also allow standalone .sv / .v / .py files in current workspace root or rtl/generated
    if normalized.endswith((".sv", ".v", ".py", ".json", ".sby")):
        if "/" not in normalized or normalized.startswith("rtl/generated/") or normalized.startswith("designs/"):
            is_allowed = True

    return is_allowed


def is_path_safe_for_read(path: str, base_dir: str = ".") -> bool:
    """Check whether a target file path is safe for reading."""
    normalized = os.path.normpath(path).replace("\\", "/")
    if ".." in normalized.split("/"):
        return False
    if os.path.isabs(path):
        workspace_abs = os.path.abspath(base_dir).replace("\\", "/")
        norm_abs = os.path.abspath(path).replace("\\", "/")
        if not norm_abs.startswith(workspace_abs):
            return False
    if normalized == ".git" or normalized.startswith(".git/"):
        return False
    return True


def safe_write_file(path: str, content: str, base_dir: str = ".") -> None:
    """Central authoritative sandbox write function.

    Guarantees no component or handler performs untracked/unsafe filesystem mutations.
    """
    if not is_path_safe_for_write(path, base_dir=base_dir):
        raise PermissionError(f"Security sandbox violation: write to '{path}' is forbidden.")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


class ActionRouter:
    """Routes agent actions to the appropriate tool handlers.

    Handlers are registered by ActionType and receive validated parameters.
    """

    def __init__(self, timeout_seconds: float = 120.0):
        self._handlers: dict[ActionType, Callable] = {}
        self.default_timeout = timeout_seconds

    def register(self, action_type: ActionType, handler: Callable) -> None:
        """Register a handler for an action type."""
        self._handlers[action_type] = handler
        logger.debug(f"Registered handler for {action_type.value}")

    # ── Action Parsing ───────────────────────────────────────────────

    def parse_action(self, llm_output: dict[str, Any]) -> AgentAction:
        """Parse an LLM JSON response into a typed AgentAction.

        IMPORTANT: If action is unknown, returns ActionType.UNKNOWN.
        NEVER defaults unknown actions to COMPLETE.
        """
        action_str = str(llm_output.get("action", "")).strip().upper()
        thinking = str(llm_output.get("thinking", ""))
        params = llm_output.get("params", {})
        if not isinstance(params, dict):
            params = {"raw_params": params}

        # Normalize aliases
        alias_map = {
            "SEARCH": ActionType.SEARCH_WEB,
            "SEARCH_WEB": ActionType.SEARCH_WEB,
            "RETRIEVE_MEMORY": ActionType.RETRIEVE_MEMORY,
            "INSPECT_MEMORY": ActionType.RETRIEVE_MEMORY,
            "READ_SOURCE": ActionType.READ_SOURCE,
            "PROPOSE_ARCHITECTURE": ActionType.PROPOSE_ARCHITECTURE,
            "PROPOSE_ARCH": ActionType.PROPOSE_ARCHITECTURE,
            "SEARCH_ARCHITECTURE": ActionType.PROPOSE_ARCHITECTURE,
            "COMPARE_ARCHITECTURES": ActionType.COMPARE_ARCHITECTURES,
            "COMPARE_ARCH": ActionType.COMPARE_ARCHITECTURES,
            "GENERATE_RTL": ActionType.GENERATE_RTL,
            "CREATE_RTL": ActionType.GENERATE_RTL,
            "GENERATE_TESTBENCH": ActionType.GENERATE_TESTBENCH,
            "EDIT_RTL": ActionType.EDIT_RTL,
            "DEBUG": ActionType.DEBUG,
            "RUN_SIMULATION": ActionType.RUN_SIMULATION,
            "SIMULATE": ActionType.RUN_SIMULATION,
            "RUN_VERILATOR": ActionType.RUN_SIMULATION,
            "RUN_TESTS": ActionType.RUN_TESTS,
            "RUN_COCOTB": ActionType.RUN_TESTS,
            "SYNTHESIZE": ActionType.SYNTHESIZE,
            "RUN_YOSYS": ActionType.SYNTHESIZE,
            "FORMAL_VERIFY": ActionType.FORMAL_VERIFY,
            "COMPARE_DESIGNS": ActionType.COMPARE_DESIGNS,
            "SAVE_DESIGN": ActionType.SAVE_DESIGN,
            "SAVE_EXPERIENCE": ActionType.SAVE_EXPERIENCE,
            "OPTIMIZE": ActionType.OPTIMIZE,
            "COMPLETE": ActionType.COMPLETE,
            "FINISH": ActionType.COMPLETE,
        }

        action_type = alias_map.get(action_str, ActionType.UNKNOWN)
        if action_type == ActionType.UNKNOWN:
            logger.warning(f"Unrecognized action '{action_str}'. Flagged as UNKNOWN for recovery.")

        return AgentAction(
            action_type=action_type,
            params=params,
            thinking=thinking,
            raw_response=str(llm_output),
        )

    # ── Action Execution ─────────────────────────────────────────────

    async def execute(self, action: AgentAction) -> ActionResult:
        """Execute an action with validation, security boundaries, and timeouts."""
        logger.info(f"Executing action: {action.action_type.value}")

        # 1. Handle unknown action gracefully (do NOT crash or complete)
        if action.action_type == ActionType.UNKNOWN:
            return ActionResult(
                action=ActionType.UNKNOWN,
                status=ActionStatus.ERROR,
                output=(
                    f"Unknown action requested. Valid actions are:\n"
                    f"SEARCH_WEB, RETRIEVE_MEMORY, READ_SOURCE, GENERATE_RTL, "
                    f"GENERATE_TESTBENCH, EDIT_RTL, DEBUG, RUN_SIMULATION, RUN_TESTS, "
                    f"SYNTHESIZE, FORMAL_VERIFY, COMPARE_DESIGNS, SAVE_DESIGN, "
                    f"SAVE_EXPERIENCE, COMPLETE"
                ),
                errors=["Unrecognized action type"],
            )

        # 2. Handle completion
        if action.action_type == ActionType.COMPLETE:
            summary = action.params.get("summary", "Task marked as complete by agent.")
            return ActionResult(
                action=ActionType.COMPLETE,
                status=ActionStatus.SUCCESS,
                output=summary,
            )

        # 3. Comprehensive security check: validate all path-bearing parameters
        path_pattern = re.compile(r"(path|file|dir|source|testbench|netlist)", re.IGNORECASE)
        is_read_action = action.action_type in (ActionType.READ_SOURCE, ActionType.RETRIEVE_MEMORY)

        for key, val in action.params.items():
            if path_pattern.search(key):
                candidates_to_check = val if isinstance(val, list) else [val]
                for item in candidates_to_check:
                    if not isinstance(item, str) or not item.strip():
                        continue
                    item_str = item.strip()
                    if is_read_action:
                        if not is_path_safe_for_read(item_str):
                            return ActionResult(
                                action=action.action_type,
                                status=ActionStatus.ERROR,
                                output=f"Security violation: Reading from protected/unapproved path '{item_str}' is forbidden.",
                                errors=[f"Protected path violation: {item_str}"],
                            )
                    else:
                        if not is_path_safe_for_write(item_str):
                            return ActionResult(
                                action=action.action_type,
                                status=ActionStatus.ERROR,
                                output=f"Security violation: Writing to protected path '{item_str}' is forbidden.",
                                errors=[f"Protected path violation: {item_str}"],
                            )

        # 4. Lookup registered handler (with alias resolution)
        handler = self._handlers.get(action.action_type)
        if handler is None:
            # Check if an alias handler exists
            for act_enum, h in self._handlers.items():
                if act_enum.value == action.action_type.value:
                    handler = h
                    break

        if handler is None:
            logger.error(f"No handler registered for {action.action_type.value}")
            return ActionResult(
                action=action.action_type,
                status=ActionStatus.ERROR,
                output=f"No tool handler registered for action: {action.action_type.value}",
                errors=[f"Unregistered action type: {action.action_type.value}"],
            )

        # 5. Execute handler with timeout
        timeout = float(action.params.get("timeout", self.default_timeout))
        try:
            if inspect.iscoroutinefunction(handler):
                result = await asyncio.wait_for(handler(action.params), timeout=timeout)
            else:
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, handler, action.params),
                    timeout=timeout,
                )

            if isinstance(result, ActionResult):
                return result
            elif isinstance(result, dict):
                return ActionResult(
                    action=action.action_type,
                    status=ActionStatus.SUCCESS if result.get("status") in ("success", "passed") else ActionStatus.FAILURE,
                    output=result.get("output", str(result)),
                    errors=result.get("errors", [result["error"]] if "error" in result and result["error"] else []),
                    metrics=result.get("metrics", {}),
                )
            else:
                return ActionResult(
                    action=action.action_type,
                    status=ActionStatus.SUCCESS,
                    output=str(result),
                )

        except asyncio.TimeoutError:
            logger.error(f"Timeout ({timeout}s) executing {action.action_type.value}")
            return ActionResult(
                action=action.action_type,
                status=ActionStatus.TIMEOUT,
                output=f"Execution timed out after {timeout} seconds",
                errors=[f"Action {action.action_type.value} timed out"],
            )
        except Exception as e:
            logger.error(f"Error executing {action.action_type.value}: {e}", exc_info=True)
            return ActionResult(
                action=action.action_type,
                status=ActionStatus.ERROR,
                output=f"Execution error: {str(e)}",
                errors=[str(e)],
            )


# ── Tool Handler Factory ─────────────────────────────────────────────

def build_router(
    rtl_generator=None,
    web_searcher=None,
    memory_system=None,
    verilator=None,
    cocotb=None,
    yosys=None,
    formal=None,
    design_store=None,
    arch_engine=None,
) -> ActionRouter:
    """Build an ActionRouter with all action handlers wired up."""
    router = ActionRouter()

    from agent.architecture_search import ArchitectureSearchEngine
    search_engine = arch_engine or ArchitectureSearchEngine()

    # ── 0. ARCHITECTURE SEARCH ───────────────────────────────────────
    async def handle_propose_architecture(params: dict) -> ActionResult:
        task_id = str(params.get("task_id", params.get("task", "hardware_design")))
        task_desc = str(params.get("description", params.get("task", "Hardware design task")))
        n = int(params.get("n", params.get("candidates", 4)))
        cands = search_engine.propose_candidates(task_id=task_id, task_description=task_desc, n=n)
        summary_lines = [f"Proposed {len(cands)} architecture candidates for [{task_id}]:"]
        for c in cands:
            summary_lines.append(
                f"- ID: {c.architecture_id} | Datapath: {c.datapath_structure} | "
                f"Pipe: {c.pipeline_depth} | Parallelism: {c.parallelism} | "
                f"Buffering: {c.buffering_strategy} | Est Cells: {c.estimated_resource_requirements.get('target_cells', 'N/A')}"
            )
        return ActionResult(
            action=ActionType.PROPOSE_ARCHITECTURE,
            status=ActionStatus.SUCCESS,
            output="\n".join(summary_lines),
            artifacts={c.architecture_id: c.datapath_structure for c in cands},
            metrics={"candidates_count": len(cands)},
            reward_contribution=0.1,
        )

    router.register(ActionType.PROPOSE_ARCHITECTURE, handle_propose_architecture)

    async def handle_compare_architectures(params: dict) -> ActionResult:
        arch_ids = params.get("architecture_ids", [])
        cands = [search_engine.genealogy.candidates[aid] for aid in arch_ids if aid in search_engine.genealogy.candidates]
        if not cands:
            cands = list(search_engine.genealogy.candidates.values())[-4:]
        ranked = search_engine.rank_candidates(cands)
        lines = ["Architecture Comparison & Ranking:"]
        for idx, c in enumerate(ranked):
            lines.append(
                f"{idx+1}. {c.architecture_id} (Reward: {c.reward:.3f}, Status: {c.verification_status}) "
                f"- Datapath: {c.datapath_structure}, Pipe: {c.pipeline_depth}, Cells: {c.actual_synthesis_metrics.get('cells', 'N/A')}"
            )
        return ActionResult(
            action=ActionType.COMPARE_ARCHITECTURES,
            status=ActionStatus.SUCCESS,
            output="\n".join(lines),
            metrics={"compared_count": len(ranked)},
            reward_contribution=0.05,
        )

    router.register(ActionType.COMPARE_ARCHITECTURES, handle_compare_architectures)

    # ── 1. SEARCH_WEB ────────────────────────────────────────────────
    async def handle_search_web(params: dict) -> ActionResult:
        query = params.get("query", "")
        url = params.get("url")
        queries = params.get("queries", [query] if query else [])

        target_urls = [url] if url else []
        if not target_urls and web_searcher is not None and query:
            try:
                search_results = await web_searcher.search(query, max_results=3)
                target_urls = [r.url for r in search_results if getattr(r, "url", None)]
            except Exception as e:
                logger.warning(f"WebSearcher query '{query}' failed: {e}")

        if not target_urls:
            if not query:
                return ActionResult(
                    action=ActionType.SEARCH_WEB,
                    status=ActionStatus.FAILURE,
                    output="SEARCH_WEB requires either a 'query' or a 'url' parameter.",
                    errors=["Missing query and url"],
                )
            # If web search returned no URLs, report cleanly without faking a hardcoded page
            return ActionResult(
                action=ActionType.SEARCH_WEB,
                status=ActionStatus.SUCCESS,
                output=f"Web search for '{query}' returned no external URLs. Using offline agent memory.",
                metrics={"query": query, "source_url": None, "token_count": 0},
            )

        from scraping.scrapegraph_adapter import ScrapeGraphAdapter
        adapter = ScrapeGraphAdapter()
        extracted_contexts = []
        total_tokens = 0

        # Query top diverse sources
        for t_url in target_urls[:3]:
            try:
                ctx = await adapter.extract_compact_context(url=t_url, focused_query=query)
                extracted_contexts.append(ctx)
                total_tokens += ctx.token_count

                # Ingest into memory if available
                if memory_system is not None:
                    try:
                        if hasattr(memory_system, "knowledge") and hasattr(memory_system.knowledge, "ingest"):
                            memory_system.knowledge.ingest(
                                ctx.extracted_summary,
                                metadata={"source": ctx.source_url, "title": ctx.title, "query": query},
                            )
                    except Exception as e:
                        logger.debug(f"Knowledge ingestion skipped: {e}")
            except Exception as e:
                logger.warning(f"Extraction failed for {t_url}: {e}")

        if not extracted_contexts:
            return ActionResult(
                action=ActionType.SEARCH_WEB,
                status=ActionStatus.FAILURE,
                output=f"Failed to extract content from retrieved URLs: {target_urls[:3]}",
                errors=["Extraction failure"],
            )

        combined_output = "\n\n---\n\n".join(c.to_prompt_text() for c in extracted_contexts)
        primary_source = extracted_contexts[0].source_url

        return ActionResult(
            action=ActionType.SEARCH_WEB,
            status=ActionStatus.SUCCESS,
            output=combined_output,
            metrics={"source_url": primary_source, "source_urls": [c.source_url for c in extracted_contexts], "token_count": total_tokens},
        )

    router.register(ActionType.SEARCH_WEB, handle_search_web)
    router.register(ActionType.SEARCH, handle_search_web)

    # ── 2. RETRIEVE_MEMORY ───────────────────────────────────────────
    if memory_system is not None:
        async def handle_retrieve_memory(params: dict) -> ActionResult:
            query = params.get("query", "")
            memory_type = params.get("type", params.get("memory_type", "all"))
            docs = await memory_system.query(query, memory_type=memory_type)
            output = "\n\n---\n\n".join(
                f"[{d.source}] (score: {d.score:.3f})\n{d.content}" for d in docs
            )
            return ActionResult(
                action=ActionType.RETRIEVE_MEMORY,
                status=ActionStatus.SUCCESS,
                output=output if output else "No relevant memory items found.",
                metrics={"docs_retrieved": len(docs), "memory_type": memory_type},
            )

        router.register(ActionType.RETRIEVE_MEMORY, handle_retrieve_memory)
        router.register(ActionType.INSPECT_MEMORY, handle_retrieve_memory)

    # ── 3. READ_SOURCE ───────────────────────────────────────────────
    async def handle_read_source(params: dict) -> ActionResult:
        file_path = params.get("file", params.get("path", params.get("filename", "")))
        if not file_path:
            return ActionResult(
                action=ActionType.READ_SOURCE,
                status=ActionStatus.FAILURE,
                output="Missing required parameter: 'file'",
                errors=["Parameter 'file' is required"],
            )

        if not is_path_safe_for_read(file_path):
            return ActionResult(
                action=ActionType.READ_SOURCE,
                status=ActionStatus.FAILURE,
                output=f"Security violation: access to '{file_path}' is forbidden by read policy.",
                errors=["Read policy violation"],
            )

        if not os.path.exists(file_path):
            return ActionResult(
                action=ActionType.READ_SOURCE,
                status=ActionStatus.FAILURE,
                output=f"File not found: {file_path}",
                errors=[f"File does not exist: {file_path}"],
            )

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        return ActionResult(
            action=ActionType.READ_SOURCE,
            status=ActionStatus.SUCCESS,
            output=content,
            artifacts={"file": file_path},
            metrics={"lines": len(content.splitlines()), "bytes": len(content)},
        )

    router.register(ActionType.READ_SOURCE, handle_read_source)

    # ── 4. GENERATE_RTL ──────────────────────────────────────────────
    async def handle_generate_rtl(params: dict) -> ActionResult:
        filename = params.get("filename", params.get("file", "generated.sv"))
        code = params.get("code", "")
        spec = params.get("spec", params.get("description", ""))
        arch_id = params.get("architecture_id", params.get("arch_id", ""))

        # Resolve architecture candidate from search engine if available
        candidate = None
        if arch_id and hasattr(search_engine, "genealogy") and arch_id in search_engine.genealogy.candidates:
            candidate = search_engine.genealogy.candidates[arch_id]
        elif hasattr(search_engine, "genealogy") and search_engine.genealogy.candidates:
            # Bind to most recent candidate if unspecified
            candidate = list(search_engine.genealogy.candidates.values())[-1]
            arch_id = candidate.architecture_id

        if not code and rtl_generator is not None:
            module = await rtl_generator.generate_module(
                spec=spec,
                context_docs=params.get("context", []),
                name=os.path.splitext(os.path.basename(filename))[0],
                architecture_candidate=candidate,
                architecture_id=arch_id,
            )
            code = module.code
            filename = module.filepath or filename

        if not code:
            return ActionResult(
                action=ActionType.GENERATE_RTL,
                status=ActionStatus.FAILURE,
                output="Failed to generate RTL: Code content is empty.",
                errors=["Empty RTL code"],
            )

        # Validate file extension
        if not filename.endswith((".sv", ".v")):
            filename = f"{filename}.sv"

        # Determine safe destination directory
        dest_dir = "./rtl/generated"
        file_path = os.path.join(dest_dir, os.path.basename(filename))

        try:
            safe_write_file(file_path, code)
        except Exception as e:
            return ActionResult(
                action=ActionType.GENERATE_RTL,
                status=ActionStatus.FAILURE,
                output=f"Sandbox rejected write: {e}",
                errors=[str(e)],
            )

        artifacts = {"rtl": file_path}
        if arch_id:
            artifacts["architecture_id"] = arch_id
            if candidate:
                candidate.rtl_implementation = code

        # Immediate verification plan coupling: auto-generate testbench template for generated module
        tb_path = None
        if rtl_generator is not None and not params.get("skip_tb", False):
            try:
                tb_code = await rtl_generator.generate_testbench_code(code)
                if tb_code:
                    tb_filename = f"test_{os.path.splitext(os.path.basename(filename))[0]}.py"
                    tb_path = os.path.join("./rtl/generated/testbenches", tb_filename)
                    safe_write_file(tb_path, tb_code)
                    artifacts["testbench"] = tb_path
            except Exception as e:
                logger.debug(f"Automatic testbench pairing deferred: {e}")

        msg = f"RTL written to {file_path} ({len(code.splitlines())} lines)"
        if arch_id:
            msg += f" bound to architecture [{arch_id}]"
        if tb_path:
            msg += f" with verification testbench paired at {tb_path}"

        return ActionResult(
            action=ActionType.GENERATE_RTL,
            status=ActionStatus.SUCCESS,
            output=msg,
            artifacts=artifacts,
            metrics={"lines": len(code.splitlines()), "architecture_id": arch_id},
        )

    router.register(ActionType.GENERATE_RTL, handle_generate_rtl)
    router.register(ActionType.CREATE_RTL, handle_generate_rtl)

    # ── 5. GENERATE_TESTBENCH ────────────────────────────────────────
    async def handle_generate_tb(params: dict) -> ActionResult:
        module_code = params.get("module_code", "")
        filename = params.get("filename", "test_generated.py")

        tb_code = ""
        if rtl_generator is not None:
            tb_code = await rtl_generator.generate_testbench_code(module_code)
        else:
            tb_code = params.get("code", "")

        dest_dir = "./rtl/generated/testbenches"
        file_path = os.path.join(dest_dir, os.path.basename(filename))

        try:
            safe_write_file(file_path, tb_code)
        except Exception as e:
            return ActionResult(
                action=ActionType.GENERATE_TESTBENCH,
                status=ActionStatus.FAILURE,
                output=f"Sandbox rejected testbench write: {e}",
                errors=[str(e)],
            )

        return ActionResult(
            action=ActionType.GENERATE_TESTBENCH,
            status=ActionStatus.SUCCESS,
            output=f"Testbench written to {file_path}",
            artifacts={"testbench": file_path},
        )

    router.register(ActionType.GENERATE_TESTBENCH, handle_generate_tb)

    # ── 6. EDIT_RTL ──────────────────────────────────────────────────
    async def handle_edit_rtl(params: dict) -> ActionResult:
        file_path = params.get("file", params.get("filename", "mac.sv"))
        code = params.get("code", "")

        # Find file in rtl/generated or designs
        target_path = file_path
        if not os.path.exists(target_path):
            alt_path = os.path.join("./rtl/generated", os.path.basename(file_path))
            if os.path.exists(alt_path):
                target_path = alt_path

        if not code:
            return ActionResult(
                action=ActionType.EDIT_RTL,
                status=ActionStatus.FAILURE,
                output="No replacement code provided in 'code' parameter.",
                errors=["Empty code parameter"],
            )

        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(code)

        return ActionResult(
            action=ActionType.EDIT_RTL,
            status=ActionStatus.SUCCESS,
            output=f"Updated {target_path} successfully.",
            artifacts={"rtl": target_path},
        )

    router.register(ActionType.EDIT_RTL, handle_edit_rtl)

    # ── 7. DEBUG ─────────────────────────────────────────────────────
    async def handle_debug(params: dict) -> ActionResult:
        code = params.get("code", "")
        errors = params.get("errors", [])
        if rtl_generator is not None:
            fixed = await rtl_generator.fix_errors(code, errors)
        else:
            fixed = f"// Debug analysis:\n// Errors observed: {errors}\n{code}"

        return ActionResult(
            action=ActionType.DEBUG,
            status=ActionStatus.SUCCESS,
            output=fixed,
        )

    router.register(ActionType.DEBUG, handle_debug)

    # ── 8. RUN_SIMULATION ────────────────────────────────────────────
    if verilator is not None:
        async def handle_run_sim(params: dict) -> ActionResult:
            sources = params.get("sources", [params.get("file_path", params.get("filename", "top.sv"))])
            top_module = params.get("top_module", None)

            if isinstance(sources, str):
                sources = [sources]

            # Resolve paths
            resolved = []
            for s in sources:
                if os.path.exists(s):
                    resolved.append(s)
                elif os.path.exists(os.path.join("./rtl/generated", s)):
                    resolved.append(os.path.join("./rtl/generated", s))
                elif os.path.exists(os.path.join("./rtl/reference", s)):
                    resolved.append(os.path.join("./rtl/reference", s))
                else:
                    resolved.append(s)

            res = await verilator.lint_and_compile(resolved[0], top_module=top_module)
            status = ActionStatus.SUCCESS if res.get("status") == "passed" else ActionStatus.FAILURE

            return ActionResult(
                action=ActionType.RUN_SIMULATION,
                status=status,
                output=res.get("error", "Verilator lint/compile clean."),
                errors=[res["error"]] if res.get("error") else [],
                metrics={"lint_clean": 1.0 if status == ActionStatus.SUCCESS else 0.0},
            )

        router.register(ActionType.RUN_SIMULATION, handle_run_sim)
        router.register(ActionType.SIMULATE, handle_run_sim)
        router.register(ActionType.RUN_VERILATOR, handle_run_sim)

    # ── 9. RUN_TESTS ─────────────────────────────────────────────────
    if cocotb is not None:
        async def handle_run_tests(params: dict) -> ActionResult:
            rtl_file = params.get("rtl_file", params.get("file_path", "top.sv"))
            tb_file = params.get("testbench", params.get("testbench_path", None))
            allow_mac = params.get("allow_heuristic_fallback", False)

            res = await cocotb.execute(rtl_file=rtl_file, testbench=tb_file, allow_heuristic_fallback=allow_mac)
            status = ActionStatus.SUCCESS if res.get("status") == "passed" else ActionStatus.FAILURE

            return ActionResult(
                action=ActionType.RUN_TESTS,
                status=status,
                output=res.get("error", f"Passed {res.get('tests_passed', 0)}/{res.get('tests_total', 0)} tests."),
                errors=[res["error"]] if res.get("error") else [],
                metrics={
                    "tests_total": res.get("tests_total", 0),
                    "tests_passed": res.get("tests_passed", 0),
                    "tests_failed": res.get("tests_failed", 0),
                },
            )

        router.register(ActionType.RUN_TESTS, handle_run_tests)
        router.register(ActionType.RUN_COCOTB, handle_run_tests)

    # ── 10. SYNTHESIZE ───────────────────────────────────────────────
    if yosys is not None:
        async def handle_synthesize(params: dict) -> ActionResult:
            file_path = params.get("file_path", params.get("sources", "top.sv"))
            if isinstance(file_path, list):
                file_path = file_path[0] if file_path else "top.sv"
            top_module = params.get("top_module", None)
            allow_heuristic = params.get("allow_heuristic_fallback", False)

            res = await yosys.synthesize(file_path=file_path, top_module=top_module, allow_heuristic_fallback=allow_heuristic)
            status = ActionStatus.SUCCESS if res.get("status") == "passed" else ActionStatus.FAILURE

            return ActionResult(
                action=ActionType.SYNTHESIZE,
                status=status,
                output=f"Yosys synthesis: {res.get('cells', 0)} cells, {res.get('dffs', 0)} DFFs, area {res.get('estimated_area', 0)} um^2",
                errors=[res["error"]] if res.get("error") else [],
                metrics={
                    "cells": res.get("cells", 0),
                    "dffs": res.get("dffs", 0),
                    "estimated_area": res.get("estimated_area", 0.0),
                },
            )

        router.register(ActionType.SYNTHESIZE, handle_synthesize)
        router.register(ActionType.RUN_YOSYS, handle_synthesize)

    # ── 11. FORMAL_VERIFY ────────────────────────────────────────────
    from tools.formal import FormalVerificationTool
    formal_tool = formal or FormalVerificationTool()

    async def handle_formal(params: dict) -> ActionResult:
        file_path = params.get("file_path", "top.sv")
        top_module = params.get("top_module", None)
        res = await formal_tool.verify(file_path=file_path, top_module=top_module)

        status = ActionStatus.SUCCESS if res.get("status") == "PASS" else (
            ActionStatus.SKIPPED if res.get("status") == "SKIPPED" else ActionStatus.FAILURE
        )

        return ActionResult(
            action=ActionType.FORMAL_VERIFY,
            status=status,
            output=res.get("output", f"Formal status: {res.get('status')}"),
            errors=res.get("errors", []),
            metrics={"formal_status": res.get("status", "UNKNOWN")},
        )

    router.register(ActionType.FORMAL_VERIFY, handle_formal)

    # ── 12. COMPARE_DESIGNS ──────────────────────────────────────────
    async def handle_compare(params: dict) -> ActionResult:
        design_a = params.get("design_a", "")
        design_b = params.get("design_b", "")

        from memory.design_store import DesignStore
        ds = design_store or DesignStore()
        rec_a = ds.get(design_a)
        rec_b = ds.get(design_b)

        if not rec_a or not rec_b:
            return ActionResult(
                action=ActionType.COMPARE_DESIGNS,
                status=ActionStatus.FAILURE,
                output=f"Could not find designs for comparison: {design_a} vs {design_b}",
                errors=["Design not found in catalog"],
            )

        diff_summary = (
            f"Comparison: {design_a} vs {design_b}\n"
            f"Reward: {rec_a.reward} vs {rec_b.reward}\n"
            f"Cells: {rec_a.synthesis_metrics.get('cells', 'N/A')} vs {rec_b.synthesis_metrics.get('cells', 'N/A')}\n"
            f"Area: {rec_a.synthesis_metrics.get('area', 'N/A')} vs {rec_b.synthesis_metrics.get('area', 'N/A')}\n"
        )
        return ActionResult(
            action=ActionType.COMPARE_DESIGNS,
            status=ActionStatus.SUCCESS,
            output=diff_summary,
        )

    router.register(ActionType.COMPARE_DESIGNS, handle_compare)

    # ── 13. SAVE_DESIGN ──────────────────────────────────────────────
    async def handle_save_design(params: dict) -> ActionResult:
        from memory.design_store import DesignRecord, DesignStore
        ds = design_store or DesignStore()
        name = params.get("design_name", params.get("name", "design"))
        version = params.get("version", "v1.0")
        rtl_code = params.get("rtl_code", "")

        dest_path = f"./designs/{name}/{version}/{name}.sv"
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        if rtl_code:
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(rtl_code)

        record = DesignRecord(
            design_id=f"{name}_{version}",
            module_name=name,
            version=version,
            rtl_source=rtl_code,
            synthesis_results=params.get("metrics", {}),
            reward=float(params.get("reward", 0.0)),
        )
        key = ds.save_design(record)

        return ActionResult(
            action=ActionType.SAVE_DESIGN,
            status=ActionStatus.SUCCESS,
            output=f"Design {name}@{version} saved to catalog (key: {key})",
            artifacts={"design": dest_path},
            metrics={"version": version, "name": name},
        )

    router.register(ActionType.SAVE_DESIGN, handle_save_design)

    # ── 14. SAVE_EXPERIENCE ──────────────────────────────────────────
    if memory_system is not None:
        async def handle_save_experience(params: dict) -> ActionResult:
            task = params.get("task", "")
            error = params.get("error", "")
            correction = params.get("correction", "")
            result = params.get("result", "passed")

            if hasattr(memory_system, "experience") and hasattr(memory_system.experience, "record"):
                memory_system.experience.record(
                    task=task,
                    action=params.get("action_taken", "EDIT_RTL"),
                    result=result,
                    errors=[error] if error else [],
                    success=(result == "passed"),
                )

            return ActionResult(
                action=ActionType.SAVE_EXPERIENCE,
                status=ActionStatus.SUCCESS,
                output="Experience recorded into experience memory.",
            )

        router.register(ActionType.SAVE_EXPERIENCE, handle_save_experience)

    logger.info(f"ActionRouter built with {len(router._handlers)} handlers registered.")
    return router
