"""
First Milestone Demonstration Script:
Runs the autonomous self-evolving hardware design agent on:
"Design an 8-bit signed MAC."

Executes the complete flow:
1. Receives user task: "Design an 8-bit signed MAC."
2. Decides whether external web research is needed.
3. Retrieves relevant memory & guidelines.
4. Generates SystemVerilog RTL and saves mac.sv.
5. Runs Verilator lint/syntax verification.
6. Runs Cocotb functional verification.
7. Inspects structured error feedback if failures occur, repairs RTL, and retries.
8. Runs Yosys synthesis.
9. Calculates grounded reward (R = compile + functional + synthesis + lint).
10. Stores complete trajectory episode.
11. Returns final RTL and evaluation metrics.
"""

import asyncio
import json
import logging
import os
import sys

# Ensure repository root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.agent import HardwareAgent
from llm.qwen import OllamaQwenClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("MilestoneRunner")


async def main():
    print("=" * 80)
    print("  SELF-EVOLVING HARDWARE DESIGN AGENT — FIRST MILESTONE")
    print("  Task: 'Design an 8-bit signed MAC.'")
    print("  Target: result = (a * b) + acc")
    print("=" * 80)

    # Initialize Qwen client (uses Ollama if available, with robust fallback)
    llm = OllamaQwenClient(model="qwen2.5-coder:3b")

    # Initialize Agent
    agent = HardwareAgent(llm=llm, work_dir="./rtl/generated")

    task = "Design an 8-bit signed MAC. Mathematical target: result = (a * b) + acc. Synthesizable SystemVerilog, signed arithmetic, separate testbench, no vendor primitives."

    print("\n[1/4] Starting Agent Episode...")
    result = await agent.run_episode(task)

    print("\n[2/4] Agent Episode Finished!")
    print(f"  Episode ID:    {result['episode_id']}")
    print(f"  Trajectory ID: {result['trajectory_id']}")
    print(f"  Success:       {result['success']}")
    print(f"  Total Steps:   {result['total_steps']}")
    print(f"  Reward:        {result['reward']} / 8.0")
    print("\n[3/4] Grounded Reward Breakdown:")
    for k, v in result["reward_breakdown"].items():
        print(f"    - {k}: {v}")

    print("\n[4/4] Generated RTL Preview:")
    print("-" * 60)
    lines = result["final_rtl"].splitlines()
    for line in lines[:30]:
        print(f"  {line}")
    if len(lines) > 30:
        print(f"  ... [{len(lines)-30} more lines in {result['rtl_file']}]")
    print("-" * 60)

    print("\nMilestone Execution Complete.")


if __name__ == "__main__":
    asyncio.run(main())
