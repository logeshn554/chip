"""
Hardware Design RL Environment — Gym-compatible environment foundation for GRPO/RL.

Exposes standard RL interface:
- reset() -> observation
- step(action) -> (next_observation, reward, done, info)

Supported Action Space:
- SEARCH_WEB
- RETRIEVE_MEMORY
- GENERATE_RTL
- EDIT_RTL
- SIMULATE
- TEST
- FORMAL_VERIFY
- SYNTHESIZE
- COMPARE_DESIGNS
- COMPLETE

Uses deterministic evaluator feedback from Verilator, Cocotb, Yosys, and Formal tools.
Maintains true reward accumulation: episode_return += step_reward.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from typing import Any, Optional, Tuple

from agent.schemas import ActionStatus, ActionType, ActionResult
from evaluator.reward import RewardEngine
from tools.verilator import VerilatorTool
from tools.cocotb import CocotbTool
from tools.yosys import YosysTool
from tools.formal import FormalVerificationTool
from memory.knowledge_store import KnowledgeStore
from memory.design_store import DesignStore

logger = logging.getLogger(__name__)


class HardwareDesignEnv:
    """Gym-compatible reinforcement learning environment for hardware design tasks."""

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
        self.task = task
        self.work_dir = work_dir
        self.max_steps = max_steps
        self.target_module = target_module

        os.makedirs(work_dir, exist_ok=True)

        # Tools
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

    def reset(self) -> dict[str, Any]:
        """Reset environment for a new episode and return initial observation."""
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
            os.remove(self.current_rtl_path)

        observation = {
            "task": self.task,
            "step": self.current_step,
            "current_rtl": "",
            "last_error": "",
            "best_reward": 0.0,
            "status": "ready",
        }
        return observation

    async def step_async(self, action: dict[str, Any] | str) -> Tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Execute an environment step asynchronously."""
        if self.is_done:
            raise RuntimeError("Cannot step in an environment that is already done. Call reset() first.")

        self.current_step += 1
        action_name = ""
        params: dict[str, Any] = {}

        if isinstance(action, str):
            action_name = action.upper().strip()
        elif isinstance(action, dict):
            action_name = str(action.get("action", "")).upper().strip()
            params = action.get("params", {})

        step_reward = 0.0
        info: dict[str, Any] = {"action": action_name, "step": self.current_step}

        # ── Handle Actions ───────────────────────────────────────────
        if action_name == "GENERATE_RTL" or action_name == "EDIT_RTL":
            code = params.get("code", "")
            if not code and "mac" in self.target_module:
                # Reference MAC code
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

        elif action_name == "RETRIEVE_MEMORY" or action_name == "SEARCH_WEB":
            step_reward = 0.05  # slight exploration incentive
            info["query"] = params.get("query", self.task)

        elif action_name == "COMPLETE":
            self.is_done = True
            info["completed"] = True
            if self.current_design_reward >= 7.0:
                step_reward = 1.0  # terminal bonus for valid verified design

        # Check step bounds
        if self.current_step >= self.max_steps:
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
            "status": "done" if self.is_done else "in_progress",
        }

        self.history.append({"step": self.current_step, "action": action_name, "reward": step_reward})
        return next_obs, step_reward, self.is_done, info

    def step(self, action: dict[str, Any] | str) -> Tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Synchronous wrapper for step_async."""
        return asyncio.run(self.step_async(action))
