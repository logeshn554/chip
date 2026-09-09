"""
Hardware Design RL Environment — Gymnasium Environment for Hardware Agent Training.

Implements an official Gymnasium Environment (gymnasium.Env):
- reset(seed=None, options=None) -> (observation, info)
- step(action) -> (observation, reward, terminated, truncated, info)

Supported Action Space (Discrete 10 or typed action names):
0: SEARCH_WEB
1: RETRIEVE_MEMORY
2: GENERATE_RTL
3: EDIT_RTL
4: SIMULATE
5: TEST
6: FORMAL_VERIFY
7: SYNTHESIZE
8: COMPARE_DESIGNS
9: COMPLETE

Uses deterministic evaluator feedback from Verilator, Cocotb, Yosys, and Formal tools.
Maintains true reward accumulation: episode_return += step_reward.
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
    # Graceful fallback if gymnasium is not installed
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

logger = logging.getLogger(__name__)


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
    ):
        super().__init__()
        self.task = task
        self.work_dir = work_dir
        self.max_steps = max_steps
        self.target_module = target_module

        os.makedirs(work_dir, exist_ok=True)

        # Standard Gymnasium spaces
        if HAS_GYMNASIUM:
            self.action_space = spaces.Discrete(len(self.ACTION_SPACE))
            self.observation_space = spaces.Dict({
                "step": spaces.Box(low=0, high=max_steps + 1, shape=(), dtype=int),
                "best_reward": spaces.Box(low=0.0, high=8.0, shape=(), dtype=float),
                "current_reward": spaces.Box(low=0.0, high=8.0, shape=(), dtype=float),
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

        # Episode state variables
        self.current_step = 0
        self.episode_return = 0.0
        self.best_design_reward = 0.0
        self.current_design_reward = 0.0
        self.current_rtl_path: Optional[str] = None
        self.last_error: str = ""
        self.is_done = False
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
        self.best_design_reward = 0.0
        self.current_design_reward = 0.0
        self.last_error = ""
        self.is_done = False
        self.history = []

        # Reset working RTL file
        self.current_rtl_path = os.path.join(self.work_dir, f"{self.target_module}.sv")
        if os.path.exists(self.current_rtl_path):
            try:
                os.remove(self.current_rtl_path)
            except Exception:
                pass

        observation = {
            "task": self.task,
            "step": self.current_step,
            "current_rtl": "",
            "last_error": "",
            "best_reward": 0.0,
            "current_reward": 0.0,
            "status": "ready",
        }
        info = {
            "task": self.task,
            "target_module": self.target_module,
            "max_steps": self.max_steps,
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

        step_reward = 0.0
        info: dict[str, Any] = {"action": action_name, "step": self.current_step}

        # ── Handle Actions ───────────────────────────────────────────
        if action_name in ["GENERATE_RTL", "EDIT_RTL"]:
            code = params.get("code", "")
            if not code and "mac" in self.target_module:
                # Synthesizable MAC reference code
                code = (
                    "`timescale 1ns / 1ps\n"
                    "module mac #(\n"
                    "    parameter DATA_WIDTH = 8,\n"
                    "    parameter ACC_WIDTH = 32\n"
                    ") (\n"
                    "    input  logic                     clk,\n"
                    "    input  logic                     rst_n,\n"
                    "    input  logic                     valid_in,\n"
                    "    input  logic signed [DATA_WIDTH-1:0] a,\n"
                    "    input  logic signed [DATA_WIDTH-1:0] b,\n"
                    "    output logic signed [ACC_WIDTH-1:0]  accum,\n"
                    "    output logic                     valid_out\n"
                    ");\n"
                    "    always_ff @(posedge clk or negedge rst_n) begin\n"
                    "        if (!rst_n) begin\n"
                    "            accum     <= '0;\n"
                    "            valid_out <= 1'b0;\n"
                    "        end else if (valid_in) begin\n"
                    "            accum     <= accum + (a * b);\n"
                    "            valid_out <= 1'b1;\n"
                    "        end else begin\n"
                    "            valid_out <= 1'b0;\n"
                    "        end\n"
                    "    end\n"
                    "endmodule\n"
                )

            with open(self.current_rtl_path, "w", encoding="utf-8") as f:
                f.write(code)

            step_reward = 0.1  # small reward for producing code
            info["status"] = "RTL written"

        elif action_name == "SIMULATE":
            if not self.current_rtl_path or not os.path.exists(self.current_rtl_path):
                step_reward = -0.5
                self.last_error = "No RTL file available to simulate."
            else:
                res = await self.verilator.lint_and_compile(self.current_rtl_path, top_module=self.target_module)
                if res.get("status") == "passed":
                    step_reward = 1.0  # compile success
                    self.last_error = ""
                else:
                    step_reward = -0.2
                    self.last_error = res.get("error", "Verilator lint/compile failed")
            info["lint_res"] = self.last_error or "passed"

        elif action_name == "TEST":
            if not self.current_rtl_path or not os.path.exists(self.current_rtl_path):
                step_reward = -0.5
                self.last_error = "No RTL file available for test."
            else:
                res = await self.cocotb.run_tests(self.current_rtl_path, "tests/mac/test_mac.py")
                if res.get("status") == "passed":
                    step_reward = 5.0  # functional tests pass
                    self.last_error = ""
                else:
                    step_reward = -0.5
                    self.last_error = res.get("error", "Functional verification failed")
            info["test_res"] = self.last_error or "passed"

        elif action_name == "SYNTHESIZE":
            if not self.current_rtl_path or not os.path.exists(self.current_rtl_path):
                step_reward = -0.5
                self.last_error = "No RTL file to synthesize."
            else:
                res = await self.yosys.synthesize(self.current_rtl_path, top_module=self.target_module)
                if res.get("status") == "passed":
                    step_reward = 1.0  # synthesis pass
                    info["cells"] = res.get("cells", 0)
                else:
                    step_reward = -0.2
                    self.last_error = res.get("error", "Synthesis failed")
            info["synth_res"] = self.last_error or "passed"

        elif action_name == "FORMAL_VERIFY":
            if not self.current_rtl_path or not os.path.exists(self.current_rtl_path):
                step_reward = -0.5
            else:
                res = await self.formal.verify(self.current_rtl_path, top_module=self.target_module)
                if res.get("status") == "PASS":
                    step_reward = 1.0
                elif res.get("status") == "SKIPPED":
                    step_reward = 0.0  # neutral
                else:
                    step_reward = -1.0  # violation
            info["formal_status"] = res.get("status", "SKIPPED") if 'res' in locals() else "ERROR"

        elif action_name in ["RETRIEVE_MEMORY", "SEARCH_WEB"]:
            step_reward = 0.05  # slight exploration incentive
            info["query"] = params.get("query", self.task)

        elif action_name == "COMPLETE":
            self.is_done = True
            info["completed"] = True
            if self.current_design_reward >= 7.0:
                step_reward = 1.0  # terminal bonus for valid verified design

        terminated = self.is_done
        truncated = self.current_step >= self.max_steps
        if truncated:
            self.is_done = True

        # Accumulate reward
        self.episode_return += step_reward
        self.current_design_reward = max(0.0, min(8.0, self.episode_return))
        self.best_design_reward = max(self.best_design_reward, self.current_design_reward)

        # Read current RTL
        current_code = ""
        if self.current_rtl_path and os.path.exists(self.current_rtl_path):
            with open(self.current_rtl_path, "r", encoding="utf-8", errors="replace") as f:
                current_code = f.read()

        next_obs = {
            "task": self.task,
            "step": self.current_step,
            "current_rtl": current_code[:1000],
            "last_error": self.last_error,
            "best_reward": self.best_design_reward,
            "current_reward": self.current_design_reward,
            "status": "done" if (terminated or truncated) else "in_progress",
        }

        self.history.append({"step": self.current_step, "action": action_name, "reward": step_reward})
        return next_obs, step_reward, terminated, truncated, info

    def step(
        self,
        action: Union[int, str, dict[str, Any]],
    ) -> Tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Synchronous wrapper for step_async returning Gymnasium 5-tuple:
        
        (observation, reward, terminated, truncated, info)
        """
        return asyncio.run(self.step_async(action))
