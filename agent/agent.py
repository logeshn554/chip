"""
Autonomous Hardware Design and Verification Agent.

Executes the self-evolving loop around Qwen-14B:
- Inspects relevant memory
- Decides whether web research is needed
- Extracts compact context with ScrapeGraph adapter
- Generates synthesizable SystemVerilog
- Runs Verilator lint -> Cocotb functional tests -> Yosys synthesis
- Formats structured failure feedback {stage, status, error, file, line}
- Reasons about errors, repairs RTL, and retries
- Calculates grounded reward R = compile + functional + synthesis + lint
- Stores trajectory and records experience

Strict tool interface: model cannot run arbitrary shell commands.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict
from typing import Any, Optional

from agent.state import AgentState
from agent.schemas import Episode
from agent.prompts import SYSTEM_PROMPT, ERROR_ANALYSIS_PROMPT
from agent.planner import HardwarePlanner
from llm.interface import LLMInterface, get_default_model
from llm.qwen import OllamaQwenClient
from memory.knowledge_store import KnowledgeStore
from memory.experience_store import ExperienceStore, Experience
from memory.design_store import DesignStore, DesignRecord
from memory.trajectory_store import TrajectoryStore
from tools.web_research import WebResearchTool
from tools.memory_search import MemorySearchTool
from tools.verilator import VerilatorTool
from tools.cocotb import CocotbTool
from tools.yosys import YosysTool
from tools.git import GitTool
from evaluator.reward import RewardEngine, GroundedRewardResult

# Structured logging setup
logger = logging.getLogger("HardwareAgent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class HardwareAgent:
    """Core Hardware Agent orchestrating the self-evolving design lifecycle."""

    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        work_dir: str = "./rtl/generated",
    ):
        self.llm = llm or OllamaQwenClient()
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)

        # Initialize memory systems
        self.knowledge_store = KnowledgeStore()
        self.experience_store = ExperienceStore()
        self.design_store = DesignStore()
        self.trajectory_store = TrajectoryStore()

        # Initialize tools
        self.web_tool = WebResearchTool()
        self.memory_tool = MemorySearchTool(
            self.knowledge_store, self.experience_store, self.design_store
        )
        self.verilator_tool = VerilatorTool(work_dir=os.path.join(work_dir, "sim"))
        self.cocotb_tool = CocotbTool(sim_dir=os.path.join(work_dir, "sim"))
        self.yosys_tool = YosysTool(work_dir=os.path.join(work_dir, "synth"))
        self.git_tool = GitTool()

        # Planner, Architecture Search & Reward Engine
        from agent.architecture_search import ArchitectureSearchEngine
        self.planner = HardwarePlanner(self.llm)
        self.arch_engine = ArchitectureSearchEngine()
        self.reward_engine = RewardEngine()

    def _log_observability(self, event_type: str, data: dict[str, Any], experiment_id: str = "") -> None:
        """Standardized structured observability logging matching canonical AgentLoop (Issue 53)."""
        import time
        record = {
            "event": event_type,
            "timestamp": time.time(),
            "experiment_id": experiment_id,
            "data": data,
        }
        logger.info(f"[OBSERVABILITY] {event_type} -> {json.dumps(record, default=str)}")

    async def _execute_action(self, action: str, params: dict[str, Any], state: AgentState) -> dict[str, Any]:
        """Strict tool interface: dispatches only validated actions."""
        action = action.upper().strip()
        state.iteration += 1

        if action == "SEARCH_WEB":
            query = params.get("query", state.task)
            self._log_observability("web_query", {"query": query})
            res = await self.web_tool.execute(query=query, url=params.get("url"))
            if res.get("status") == "success":
                state.relevant_web_extracts.append(res.get("compact_context", ""))
                self._log_observability("retrieved_sources", {"source_url": res.get("source_url")})
            state.add_step(action, res.get("status", "completed"), {"query": query})
            return res

        elif action == "READ_DOCUMENT":
            url = params.get("url", "")
            res = await self.web_tool.execute(query="specification extract", url=url)
            state.add_step(action, res.get("status", "completed"), {"url": url})
            return res

        elif action == "RETRIEVE_MEMORY":
            query = params.get("query", state.task)
            mem_type = params.get("memory_type", "knowledge")
            res = await self.memory_tool.execute(query=query, memory_type=mem_type)
            if res.get("status") == "success" and res.get("matches"):
                for m in res["matches"][:2]:
                    if isinstance(m, dict) and "content" in m:
                        state.relevant_memory.append(f"[{m.get('title', 'Doc')}]: {m['content']}")
            state.add_step(action, "success", {"query": query, "memory_type": mem_type})
            return res

        elif action == "PROPOSE_ARCHITECTURE":
            n = int(params.get("n", 4))
            cands = self.arch_engine.propose_candidates(
                task_id=state.task[:20].replace(" ", "_"),
                task_description=state.task,
                n=n,
            )
            cand_summaries = [f"{c.architecture_id}: {c.datapath_structure} (pipe={c.pipeline_depth})" for c in cands]
            state.add_step(action, "proposed", {"candidates": cand_summaries})
            self._log_observability("proposed_architectures", {"count": len(cands), "candidates": cand_summaries})
            return {"status": "success", "candidates": [asdict(c) for c in cands]}

        elif action == "COMPARE_ARCHITECTURES":
            cands = list(self.arch_engine.genealogy.candidates.values())[-4:]
            ranked = self.arch_engine.rank_candidates(cands)
            state.add_step(action, "compared", {"top_candidate": ranked[0].architecture_id if ranked else "none"})
            return {"status": "success", "ranked": [asdict(c) for c in ranked]}

        elif action == "CREATE_RTL" or action == "EDIT_RTL":
            filename = params.get("filename", state.active_filename)
            code = params.get("code", "")
            if not code:
                return {"status": "error", "message": "No RTL code provided"}

            file_path = os.path.join(self.work_dir, filename)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)

            state.active_filename = filename
            state.current_rtl = code
            self._log_observability("RTL_version", {"filename": filename, "lines": len(code.splitlines())})
            state.add_step(action, "saved", {"filename": filename})
            return {"status": "success", "file": file_path, "message": f"Saved {filename}"}

        elif action == "RUN_VERILATOR":
            file_path = os.path.join(self.work_dir, state.active_filename)
            res = await self.verilator_tool.lint(file_path)
            state.current_error = res if res.get("status") == "failed" else None
            self._log_observability("compiler_errors", res)
            state.add_step(action, res.get("status", "failed"), res)
            return res

        elif action == "RUN_COCOTB":
            file_path = os.path.join(self.work_dir, state.active_filename)
            res = await self.cocotb_tool.run_tests(file_path)
            state.current_error = res if res.get("status") == "failed" else None
            self._log_observability("simulation_results", res)
            state.add_step(action, res.get("status", "failed"), res)
            return res

        elif action == "RUN_YOSYS":
            file_path = os.path.join(self.work_dir, state.active_filename)
            top_mod = params.get("top_module", None)
            allow_heuristic = params.get("allow_heuristic_fallback", True)
            res = await self.yosys_tool.synthesize(file_path, top_module=top_mod, allow_heuristic_fallback=allow_heuristic)
            self._log_observability("synthesis_results", res)
            state.add_step(action, res.get("status", "failed"), res)
            return res

        elif action == "INSPECT_ERROR":
            err = state.current_error or {}
            err_msg = err.get("error", "Unknown error")
            similar = self.experience_store.retrieve_similar_failures(err_msg, state.task)
            guidance = ""
            if similar:
                top_match = similar[0]
                guidance = f"Prior fix for similar error: {top_match.get('correction')}"
                state.relevant_memory.append(guidance)
            state.add_step(action, "inspected", {"similar_found": bool(similar)})
            return {"status": "success", "error": err, "guidance": guidance}

        elif action == "COMPARE_DESIGNS":
            res = self.design_store.compare_designs(
                module_name=params.get("module_name", "top"),
                ver_a=params.get("ver_a", "v1.0"),
                ver_b=params.get("ver_b", "v1.1"),
            )
            state.add_step(action, "compared")
            return res

        elif action == "SAVE_DESIGN":
            mod_name = params.get("module_name", os.path.splitext(state.active_filename)[0] if state.active_filename else "design")
            record = DesignRecord(
                design_id=f"{mod_name}_v{state.iteration}",
                module_name=mod_name,
                version=params.get("version", f"v1.{state.iteration}"),
                rtl_source=state.current_rtl,
                reward=state.current_reward,
            )
            key = self.design_store.save_design(record)
            state.add_step(action, "saved", {"design_key": key})
            return {"status": "success", "design_key": key}

        elif action == "FINISH":
            state.is_finished = True
            state.add_step(action, "completed")
            return {"status": "success", "message": "Design iteration completed successfully."}

        else:
            return {"status": "error", "message": f"Unrecognized action: {action}"}

    async def run_episode(self, task: str) -> dict[str, Any]:
        """Execute a complete autonomous hardware design episode."""
        # Verify model availability
        if hasattr(self.llm, "verify_model_installed"):
            self.llm.verify_model_installed()

        state = AgentState(task=task)
        self._log_observability("task_started", {"task": task, "episode_id": state.episode_id})

        # Phase 1: Planning & Research Decision
        research_decision = await self.planner.decide_research_need(task)
        self._log_observability("research_decision", research_decision)

        if research_decision.get("needs_research"):
            # Execute targeted ScrapeGraph research
            focused_query = research_decision.get("focused_query", task)
            await self._execute_action("SEARCH_WEB", {"query": focused_query}, state)
        else:
            # Query internal knowledge store
            await self._execute_action(
                "RETRIEVE_MEMORY",
                {"query": "8-bit signed MAC SystemVerilog signed arithmetic", "memory_type": "knowledge"},
                state,
            )

        # Phase 2: Agent Reasoning & Action Loop with Experience Injection
        compile_ok = False
        func_ok = False
        synth_ok = False
        lint_ok = False

        # Retrieve relevant past debugging lessons and reference fixes for this task
        past_experiences = self.experience_store.retrieve_experiences_for_task(task=task, n_results=2)
        if past_experiences:
            for exp in past_experiences:
                corr = exp.get("correction", "")
                err = exp.get("error", "")
                cat = exp.get("error_category", "GENERAL")
                if corr:
                    state.relevant_memory.append(f"[Past Debugging Lesson ({cat})]: Error '{err[:60]}' was fixed by: {corr[:120]}")

        while not state.is_finished and state.iteration < state.max_iterations:
            # Construct bounded working context for Qwen-14B
            context_prompt = state.build_model_context()
            full_prompt = f"{SYSTEM_PROMPT}\n\n{context_prompt}\n\nDecide next action:"

            self._log_observability("model_request", {"iteration": state.iteration, "prompt_chars": len(full_prompt)})

            # Invoke Qwen reasoning engine
            decision = await self.llm.generate_json(full_prompt)
            action = decision.get("action", "")
            params = decision.get("parameters", {})
            thinking = decision.get("thinking", "")

            self._log_observability("selected_action", {"action": action, "thinking": thinking[:150]})

            # Fallback action selection if model produced invalid action
            if not action or action not in [
                "SEARCH_WEB", "READ_DOCUMENT", "RETRIEVE_MEMORY", "CREATE_RTL",
                "EDIT_RTL", "RUN_VERILATOR", "RUN_COCOTB", "RUN_YOSYS",
                "INSPECT_ERROR", "COMPARE_DESIGNS", "SAVE_DESIGN", "FINISH",
            ]:
                if not state.current_rtl:
                    action = "CREATE_RTL"
                elif state.current_error:
                    action = "INSPECT_ERROR"
                elif not compile_ok:
                    action = "RUN_VERILATOR"
                elif not func_ok:
                    action = "RUN_COCOTB"
                elif not synth_ok:
                    action = "RUN_YOSYS"
                else:
                    action = "FINISH"

            # Execute the tool
            tool_result = await self._execute_action(action, params, state)
            state.latest_tool_result = tool_result

            # Track verification milestones
            if action == "RUN_VERILATOR":
                compile_ok = (tool_result.get("status") == "passed")
                lint_ok = compile_ok
            elif action == "RUN_COCOTB":
                func_ok = (tool_result.get("status") == "passed")
            elif action == "RUN_YOSYS":
                synth_ok = (tool_result.get("status") == "passed")

            # Check if all stages passed and ready to finalize
            if compile_ok and func_ok and synth_ok and not state.is_finished:
                # Calculate grounded reward
                reward_res: GroundedRewardResult = self.reward_engine.compute_v1_reward(
                    compile_success=compile_ok,
                    all_functional_tests_pass=func_ok,
                    synthesis_success=synth_ok,
                    lint_clean=lint_ok,
                )
                state.current_reward = reward_res.total_reward
                self._log_observability("reward", {"total_reward": state.current_reward, "breakdown": reward_res.breakdown})

                # Save successful design
                active_mod = os.path.splitext(state.active_filename)[0] if state.active_filename else "design"
                await self._execute_action("SAVE_DESIGN", {"module_name": active_mod, "version": "v1.0"}, state)
                await self._execute_action("FINISH", {"summary": "All verification stages passed cleanly."}, state)

        # Final reward calculation if loop exited
        final_reward_res = self.reward_engine.compute_v1_reward(
            compile_success=compile_ok,
            all_functional_tests_pass=func_ok,
            synthesis_success=synth_ok,
            lint_clean=lint_ok,
        )
        state.current_reward = final_reward_res.total_reward

        # Record trajectory
        traj_meta = {
            "compile": compile_ok,
            "func": func_ok,
            "synth": synth_ok,
            "provider": getattr(self.llm, "backend", "ollama"),
            "model": getattr(self.llm, "model", getattr(self.llm, "model_name", get_default_model())),
            "fallback_used": bool(getattr(self.llm, "mock_mode", False) or os.environ.get("LLM_PROVIDER") == "mock"),
            "rtl_source": "qwen",
        }
        traj_id = self.trajectory_store.save_trajectory(
            task=task,
            steps=state.trajectory_steps,
            reward=state.current_reward,
            metadata=traj_meta,
        )
        self._log_observability("trajectory_saved", {"trajectory_id": traj_id, "reward": state.current_reward, **traj_meta})

        # WebRL-Style First-Class Failure Experience Indexing
        try:
            ep_obj = Episode(
                episode_id=state.episode_id,
                task=task,
                steps=state.trajectory_steps,
                final_reward=state.current_reward,
                success=bool(compile_ok and func_ok and synth_ok),
                metadata=traj_meta,
            )
            self.experience_store.record_episode_experience(ep_obj)
        except Exception as e:
            logger.debug(f"Could not record episode experience: {e}")

        return {
            "episode_id": state.episode_id,
            "trajectory_id": traj_id,
            "task": task,
            "success": bool(compile_ok and func_ok and synth_ok),
            "reward": state.current_reward,
            "reward_breakdown": final_reward_res.breakdown,
            "rtl_file": os.path.join(self.work_dir, state.active_filename),
            "final_rtl": state.current_rtl,
            "total_steps": len(state.trajectory_steps),
        }
