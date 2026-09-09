"""
Hardware Design RL Environment — Gymnasium Environment for Hardware Agent Training.

Implements an official Gymnasium Environment (gymnasium.Env) designed as the underlying
execution environment for an external policy (e.g. Qwen3-4B, PPO, or GRPO):

Policy / Qwen
      ↓ (selects typed action)
HardwareDesignEnv
      ↓ (executes in controlled sandbox)
EDA Tools (Verilator, Cocotb, Yosys, SymbiYosys)
      ↓ (evaluates objective correctness & metrics)
Observation & Grounded Reward
      ↓
Policy Update / Trajectory Collection

Features:
- Full observation_space and action_space matching reset() and step() contracts
- Strict separation between action step_reward, episode_return, and objective design_quality
- Objective completion criteria (compile + functional verification gates)
- No hard-coded fallback RTL generation (policy must genuinely supply RTL)
- External benchmark formal properties binding (protects against assertion gaming)
- Standardized normalized RL reward in [-1.0, +1.0] and quality in [0.0, 1.0]
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from typing import Any, Optional, Tuple, Union

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    class gym:  # type: ignore
        class Env:
            pass
    spaces = None  # type: ignore
    HAS_GYMNASIUM = False

from agent.schemas import ActionStatus, ActionType, ActionResult
from evaluator.reward import RewardEngine
from tools.verilator import VerilatorTool
from tools.cocotb import CocotbTool
from tools.yosys import YosysTool
from tools.formal import FormalVerificationTool
from memory.knowledge_store import KnowledgeStore
from memory.design_store import DesignStore
from scraping.scrapegraph_adapter import ScrapeGraphAdapter

import string
import numpy as np

logger = logging.getLogger(__name__)

PRINTABLE_CHARSET = set(string.printable)


class HardwareDesignEnv(gym.Env):
    """Gymnasium reinforcement learning environment for autonomous hardware design tasks."""

    metadata = {"render_modes": []}

    ACTION_SPACE = [
        "SEARCH_WEB",
        "RETRIEVE_MEMORY",
        "GENERATE_RTL",
        "EDIT_RTL",
        "SIMULATE",
        "TEST",
        "FORMAL_VERIFY",
        "SYNTHESIZE",
        "COMPARE_DESIGNS",
        "COMPLETE",
    ]

    def __init__(
        self,
        task: str = "Design an 8-bit signed MAC unit in synthesizable SystemVerilog.",
        work_dir: str = "./sim_build/rl_env",
        max_steps: int = 15,
        target_module: str = "mac",
        test_file: str = "tests/mac/test_mac.py",
        formal_properties: Optional[str] = None,
        target_cells: float = 500.0,
        benchmark_task_id: Optional[str] = None,
    ):
        super().__init__()
        self.task = task
        self.work_dir = work_dir
        self.max_steps = max_steps
        self.target_module = target_module
        self.test_file = test_file
        self.formal_properties = formal_properties
        self.target_cells = target_cells

        if benchmark_task_id:
            try:
                from benchmarks.curriculum import BenchmarkCurriculum
                t_obj = BenchmarkCurriculum().get_task(benchmark_task_id)
                if t_obj:
                    self.task = t_obj.get_public_spec()
                    self.target_module = t_obj.top_module
                    if t_obj.formal_properties and not self.formal_properties:
                        self.formal_properties = t_obj.formal_properties
            except Exception as e:
                logger.debug(f"Could not auto-load benchmark task {benchmark_task_id}: {e}")

        os.makedirs(work_dir, exist_ok=True)

        # Standard Gymnasium spaces - contract exactly matches reset() and step()
        if HAS_GYMNASIUM:
            self.action_space = spaces.Discrete(len(self.ACTION_SPACE))
            self.observation_space = spaces.Dict({
                "task": spaces.Text(max_length=500, min_length=0, charset=PRINTABLE_CHARSET),
                "step": spaces.Box(low=0, high=max_steps + 1, shape=(), dtype=np.int32),
                "current_rtl": spaces.Text(max_length=4000, min_length=0, charset=PRINTABLE_CHARSET),
                "last_error": spaces.Text(max_length=1000, min_length=0, charset=PRINTABLE_CHARSET),
                "best_reward": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
                "current_reward": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
                "status": spaces.Text(max_length=50, min_length=0, charset=PRINTABLE_CHARSET),
                "retrieved_context": spaces.Text(max_length=2000, min_length=0, charset=PRINTABLE_CHARSET),
            })
        else:
            self.action_space = len(self.ACTION_SPACE)
            self.observation_space = None

        # Dedicated tool wrappers
        self.verilator = VerilatorTool(work_dir=os.path.join(work_dir, "sim"))
        self.cocotb = CocotbTool(sim_dir=os.path.join(work_dir, "sim"))
        self.yosys = YosysTool(work_dir=os.path.join(work_dir, "synth"))
        self.formal = FormalVerificationTool(work_dir=os.path.join(work_dir, "formal"))
        self.reward_engine = RewardEngine()
        self.knowledge = KnowledgeStore()
        self.design_store = DesignStore()
        self.scrapegraph = ScrapeGraphAdapter(cache_dir=os.path.join(work_dir, "web_cache"))

        # State variables
        self.current_step = 0
        self.episode_return = 0.0          # Accumulated step rewards
        self.current_design_quality = 0.0  # Grounded hardware quality in [0.0, 1.0]
        self.best_design_quality = 0.0     # Best hardware quality in [0.0, 1.0]
        self.current_rtl_path: Optional[str] = None
        self.last_error: str = ""
        self.last_retrieved_context: str = ""
        self.is_done = False
        self.compile_passed = False
        self.functional_passed = False
        self.synthesis_passed = False
        self.formal_passed = False
        self.cells_count = 0
        self.history: list[dict[str, Any]] = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> Tuple[dict[str, Any], dict[str, Any]]:
        """Reset environment for a new episode and return (observation, info)."""
        if HAS_GYMNASIUM and hasattr(super(), "reset"):
            super().reset(seed=seed)

        self.current_step = 0
        self.episode_return = 0.0
        self.current_design_quality = 0.0
        self.best_design_quality = 0.0
        self.last_error = ""
        self.last_retrieved_context = ""
        self.is_done = False
        self.compile_passed = False
        self.functional_passed = False
        self.synthesis_passed = False
        self.formal_passed = False
        self.cells_count = 0
        self.history = []

        # Reset working RTL file
        self.current_rtl_path = os.path.join(self.work_dir, f"{self.target_module}.sv")
        if os.path.exists(self.current_rtl_path):
            try:
                os.remove(self.current_rtl_path)
            except Exception:
                pass

        observation = {
            "task": str(self.task)[:500],
            "step": np.array(self.current_step, dtype=np.int32),
            "current_rtl": "",
            "last_error": "",
            "best_reward": np.array(0.0, dtype=np.float32),
            "current_reward": np.array(0.0, dtype=np.float32),
            "status": "ready",
            "retrieved_context": "",
        }
        info = {
            "task": self.task,
            "target_module": self.target_module,
            "max_steps": self.max_steps,
            "episode_return": 0.0,
            "design_quality": 0.0,
        }
        return observation, info

    async def step_async(
        self,
        action: Union[int, str, dict[str, Any]],
    ) -> Tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute an environment step asynchronously.
        
        Returns standard Gymnasium 5-tuple: (observation, reward, terminated, truncated, info).
        """
        if self.is_done:
            raise RuntimeError("Cannot step in an environment that is already done. Call reset() first.")

        self.current_step += 1
        action_name = ""
        params: dict[str, Any] = {}

        # Parse action from int, str, or dict
        if isinstance(action, int):
            if 0 <= action < len(self.ACTION_SPACE):
                action_name = self.ACTION_SPACE[action]
            else:
                action_name = "UNKNOWN"
        elif isinstance(action, str):
            action_name = action.upper().strip()
        elif isinstance(action, dict):
            action_name = str(action.get("action", "")).upper().strip()
            params = action.get("params", {})

        step_reward = -0.01  # Small step execution cost to encourage efficiency
        info: dict[str, Any] = {"action": action_name, "action_executed": action_name, "step": self.current_step}

        # ── Handle Actions ───────────────────────────────────────────
        if action_name in ["GENERATE_RTL", "EDIT_RTL"]:
            code = params.get("code", "")
            if not code:
                # Strictly require the policy to provide RTL; no secret fallback generation
                step_reward = -0.5
                self.last_error = "Action rejected: GENERATE_RTL requires non-empty 'code' parameter."
                info["status"] = "missing_code"
            else:
                with open(self.current_rtl_path, "w", encoding="utf-8") as f:
                    f.write(code)
                step_reward = 0.05  # Slight progress reward for emitting code
                self.last_error = ""
                info["status"] = "RTL written"

        elif action_name == "SIMULATE":
            if not self.current_rtl_path or not os.path.exists(self.current_rtl_path):
                step_reward = -0.5
                self.last_error = "No RTL file available to simulate."
            else:
                res = await self.verilator.lint_and_compile(self.current_rtl_path, top_module=self.target_module)
                if res.get("status") == "passed":
                    self.compile_passed = True
                    step_reward = 0.2
                    self.last_error = ""
                else:
                    self.compile_passed = False
                    step_reward = -0.2
                    self.last_error = res.get("error", "Verilator lint/compile failed")
            info["compile_passed"] = self.compile_passed

        elif action_name == "TEST":
            if not self.current_rtl_path or not os.path.exists(self.current_rtl_path):
                step_reward = -0.5
                self.last_error = "No RTL file available for test."
            elif not self.compile_passed:
                step_reward = -0.3
                self.last_error = "Cannot run tests before compilation passes."
            else:
                res = await self.cocotb.run_tests(self.current_rtl_path, self.test_file)
                total = res.get("tests_total", 0)
                passed = res.get("tests_passed", 0)
                pass_rate = (passed / total) if total > 0 else (1.0 if res.get("status") == "passed" else 0.0)
                if pass_rate == 1.0:
                    self.functional_passed = True
                    step_reward = 0.5
                    self.last_error = ""
                else:
                    self.functional_passed = False
                    step_reward = -0.3
                    self.last_error = res.get("error", f"Tests failed: {passed}/{total} passed")
            info["functional_passed"] = self.functional_passed

        elif action_name == "SYNTHESIZE":
            if not self.current_rtl_path or not os.path.exists(self.current_rtl_path):
                step_reward = -0.5
                self.last_error = "No RTL file to synthesize."
            elif not self.compile_passed:
                step_reward = -0.3
                self.last_error = "Cannot synthesize uncompiled RTL."
            else:
                res = await self.yosys.synthesize(self.current_rtl_path, top_module=self.target_module)
                if res.get("status") == "passed":
                    self.synthesis_passed = True
                    self.cells_count = res.get("cells", 0) or 0
                    step_reward = 0.2
                    info["cells"] = self.cells_count
                else:
                    self.synthesis_passed = False
                    step_reward = -0.2
                    self.last_error = res.get("error", "Synthesis failed")
            info["synthesis_passed"] = self.synthesis_passed

        elif action_name == "FORMAL_VERIFY":
            if not self.current_rtl_path or not os.path.exists(self.current_rtl_path):
                step_reward = -0.5
                self.last_error = "No RTL file for formal verification."
            else:
                # Pass external benchmark formal properties if available
                res = await self.formal.verify(
                    self.current_rtl_path,
                    top_module=self.target_module,
                    external_properties=self.formal_properties,
                )
                status = res.get("status", "SKIPPED")
                if status == "PASS":
                    self.formal_passed = True
                    step_reward = 0.3
                elif status == "SKIPPED":
                    step_reward = 0.0  # Neutral when tool is not installed
                else:
                    self.formal_passed = False
                    step_reward = -0.5
                    self.last_error = "Formal assertion violation detected."
                info["formal_status"] = status

        elif action_name == "RETRIEVE_MEMORY":
            query = str(params.get("query") or self.task)
            chunks = self.knowledge.query(query, n_results=2)
            if chunks:
                retrieved_summary = "\n".join([f"[{c.get('title', 'Knowledge')}]: {c.get('content', '')}" for c in chunks])
                self.last_retrieved_context = retrieved_summary[:1800]
                step_reward = 0.05  # Positive exploration reward conditioned on finding technical knowledge
                info["status"] = "memory_retrieved"
                info["chunks_found"] = len(chunks)
            else:
                self.last_retrieved_context = ""
                step_reward = 0.0
                info["status"] = "no_memory_found"
                info["chunks_found"] = 0
            info["retrieved_context"] = self.last_retrieved_context

        elif action_name == "SEARCH_WEB":
            query = str(params.get("query") or self.task)
            url = str(params.get("url") or "https://github.com/verilator/verilator")
            ctx = await self.scrapegraph.extract_compact_context(url=url, focused_query=query)
            if ctx.extracted_summary and "Unable to retrieve" not in ctx.extracted_summary:
                self.last_retrieved_context = ctx.to_prompt_text()[:1800]
                step_reward = 0.05  # Positive exploration reward conditioned on useful research
                info["status"] = "web_research_retrieved"
                info["citation_id"] = ctx.citation_id
            else:
                self.last_retrieved_context = ""
                step_reward = 0.0
                info["status"] = "empty_research"
            info["retrieved_context"] = self.last_retrieved_context

        elif action_name == "COMPLETE":
            self.is_done = True
            # Objective verification gate: compile, functional, formal (if specified), and quality
            formal_ok = (not self.formal_properties) or self.formal_passed
            if self.compile_passed and self.functional_passed and formal_ok and self.current_design_quality >= 0.5:
                step_reward = 1.0 * max(0.5, self.current_design_quality)  # Terminal success bonus
                info["verified_complete"] = True
            else:
                step_reward = -0.5  # Premature completion penalty
                info["verified_complete"] = False

        # Calculate true hardware quality using RewardEngine with weight re-normalization
        if self.compile_passed:
            quality_res = self.reward_engine.compute_modular_reward(
                compile_success=self.compile_passed,
                test_pass_rate=1.0 if self.functional_passed else 0.0,
                formal_status="PASS" if self.formal_passed else "SKIPPED",
                area=float(self.cells_count) if (self.synthesis_passed and self.cells_count > 0) else None,
                area_target=self.target_cells,
            )
            self.current_design_quality = quality_res.normalized_reward
        else:
            self.current_design_quality = 0.0

        self.best_design_quality = max(self.best_design_quality, self.current_design_quality)

        terminated = self.is_done
        truncated = self.current_step >= self.max_steps
        if truncated:
            self.is_done = True

        # Accumulate step return
        self.episode_return += step_reward

        # Read current RTL
        current_code = ""
        if self.current_rtl_path and os.path.exists(self.current_rtl_path):
            try:
                with open(self.current_rtl_path, "r", encoding="utf-8", errors="replace") as f:
                    current_code = f.read()
            except Exception:
                pass

        next_obs = {
            "task": str(self.task)[:500],
            "step": np.array(self.current_step, dtype=np.int32),
            "current_rtl": current_code[:4000],
            "last_error": self.last_error[:1000],
            "best_reward": np.array(self.best_design_quality, dtype=np.float32),
            "current_reward": np.array(self.current_design_quality, dtype=np.float32),
            "status": "done" if (terminated or truncated) else "in_progress",
            "retrieved_context": self.last_retrieved_context[:2000],
        }

        info["episode_return"] = round(self.episode_return, 4)
        info["design_quality"] = round(self.current_design_quality, 4)
        info["engineering_score_v1"] = round(self.current_design_quality * 8.0, 4)

        self.history.append({"step": self.current_step, "action": action_name, "reward": step_reward})
        return next_obs, round(step_reward, 4), terminated, truncated, info

    def step(
        self,
        action: Union[int, str, dict[str, Any]],
    ) -> Tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Synchronous wrapper for step_async returning Gymnasium 5-tuple:
        
        (observation, reward, terminated, truncated, info)
        """
        return asyncio.run(self.step_async(action))


class HardwareVectorObservationWrapper(gym.ObservationWrapper if HAS_GYMNASIUM else object):
    """Observation wrapper projecting HardwareDesignEnv dict into a numeric Box vector.
    
    Permits training with classical RL libraries (e.g. Stable-Baselines3 PPO/DQN)
    that require fixed 1D vector observations:
    - [0]: normalized step (step / max_steps)
    - [1]: best design quality in [0.0, 1.0]
    - [2]: current design quality in [0.0, 1.0]
    - [3]: compile passed (0 or 1)
    - [4]: functional passed (0 or 1)
    - [5]: synthesis passed (0 or 1)
    - [6]: formal passed (0 or 1)
    - [7]: RTL code present (0 or 1)
    - [8]: normalized RTL length
    - [9]: error indicator (0 or 1)
    """

    def __init__(self, env: HardwareDesignEnv):
        super().__init__(env)
        if HAS_GYMNASIUM:
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(11,),
                dtype=np.float32,
            )

    def observation(self, obs: dict[str, Any]) -> np.ndarray:
        raw_env = getattr(self.env, "unwrapped", self.env)
        vec = np.zeros(11, dtype=np.float32)
        vec[0] = float(obs.get("step", 0)) / max(1.0, float(getattr(raw_env, "max_steps", 15)))
        vec[1] = float(obs.get("best_reward", 0.0))
        vec[2] = float(obs.get("current_reward", 0.0))
        vec[3] = 1.0 if getattr(raw_env, "compile_passed", False) else 0.0
        vec[4] = 1.0 if getattr(raw_env, "functional_passed", False) else 0.0
        vec[5] = 1.0 if getattr(raw_env, "synthesis_passed", False) else 0.0
        vec[6] = 1.0 if getattr(raw_env, "formal_passed", False) else 0.0
        rtl = str(obs.get("current_rtl", ""))
        vec[7] = 1.0 if len(rtl.strip()) > 0 else 0.0
        vec[8] = min(1.0, len(rtl) / 2000.0)
        vec[9] = 1.0 if len(str(obs.get("last_error", "")).strip()) > 0 else 0.0
        vec[10] = 1.0 if len(str(obs.get("retrieved_context", "")).strip()) > 0 else 0.0
        return vec


class QwenHardwareDesignPolicy:
    """External LLM policy wrapper connecting Qwen3-4B to HardwareDesignEnv.
    
    Clarifies architectural boundary:
        Hardware Agent / Qwen Policy
                   │ selects typed actions based on observations
                   ▼
        Gymnasium HardwareDesignEnv
                   │ executes actions in controlled sandbox
                   ▼
        EDA Tools & RewardEngine
    """

    def __init__(self, llm_client: Any = None, model: str = "qwen2.5-coder:3b"):
        self.llm = llm_client
        self.model = model

    def select_action(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Heuristic / LLM action selection based on current environment state."""
        current_rtl = obs.get("current_rtl", "")
        last_error = obs.get("last_error", "")
        best_reward = float(obs.get("best_reward", 0.0))

        if not current_rtl:
            return {"action": "GENERATE_RTL", "params": {}}
        elif last_error:
            return {"action": "GENERATE_RTL", "params": {"feedback": last_error}}
        elif best_reward < 0.2:
            return {"action": "SIMULATE", "params": {}}
        elif best_reward < 0.6:
            return {"action": "TEST", "params": {}}
        elif best_reward < 0.8:
            return {"action": "SYNTHESIZE", "params": {}}
        else:
            return {"action": "COMPLETE", "params": {}}
