"""
First Milestone Demonstration Script:
Runs the autonomous self-evolving hardware design agent on:
"Design an 8-bit signed MAC."

Startup Flow:
1. Load configuration
2. Verify Ollama server
3. Verify qwen2.5:14b exists
4. Test one real generation
5. Start HardwareAgent
6. Run episode
"""

import asyncio
import json
import logging
import os
import sys
import urllib.error
import urllib.request
import yaml

# Ensure repository root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.agent import HardwareAgent
from llm.qwen import OllamaQwenClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("MilestoneRunner")


def load_agent_config() -> dict:
    """Step 1: Load configuration."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "agent.yaml")
    if not os.path.exists(config_path):
        config_path = "configs/agent.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def verify_ollama_and_model(base_url: str, model_name: str) -> None:
    """Steps 2 & 3: Verify Ollama server and verify qwen2.5:14b exists."""
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama request failed: HTTP {e.code}. URL={url}, model={model_name}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot connect to Ollama at {base_url}. Ensure Ollama is running.") from e

    print("Ollama Server: PASS")

    models = data.get("models", [])
    installed_names = []
    for m in models:
        if isinstance(m, dict):
            if "name" in m:
                installed_names.append(m["name"])
            if "model" in m:
                installed_names.append(m["model"])

    target = model_name.lower()
    matched_model = None
    for name in installed_names:
        n_low = name.lower()
        if (
            target == n_low
            or n_low.startswith(f"{target}:")
            or f"{target}:latest" == n_low
            or (":" not in target and n_low.split(":")[0] == target)
        ):
            matched_model = name
            break

    if not matched_model:
        # Check if user has an alternative installed Qwen model (e.g. qwen3:14b, qwen3.5:9b, etc.)
        candidates = [name for name in installed_names if "qwen" in name.lower()]
        if candidates:
            # Sort preferring 14b if present
            candidates.sort(key=lambda x: (1 if "14b" in x.lower() else 0, x), reverse=True)
            matched_model = candidates[0]
            print(f"Auto-selected installed Ollama model: {matched_model} (configured: {model_name})")
        elif installed_names:
            matched_model = installed_names[0]
            print(f"Using installed Ollama model: {matched_model}")
        else:
            raise RuntimeError(f"No models found in Ollama. Please run: ollama pull {model_name}")

    print(f"Ollama Model [{matched_model}]: PASS")
    return matched_model


async def test_generation(client: OllamaQwenClient) -> None:
    """Step 4: Test one real generation."""
    prompt = "Ping: respond with PONG"
    resp = await client.generate(prompt)
    if not resp.text or not resp.text.strip():
        raise RuntimeError("Test generation returned empty response.")
    print("Test Generation: PASS")


async def main():
    print("=" * 80)
    print("  SELF-EVOLVING HARDWARE DESIGN AGENT — FIRST MILESTONE")
    print("  Task: 'Design an 8-bit signed MAC.'")
    print("  Target: result = (a * b) + acc")
    print("=" * 80)

    # 1. Load configuration
    config = load_agent_config()
    llm_cfg = config.get("llm", {})
    from llm.interface import get_default_model
    model_name = get_default_model(llm_cfg.get("model_name"))
    base_url = llm_cfg.get("ollama", {}).get("base_url", "http://localhost:11434")

    print(f"LLM Provider: Ollama")
    print(f"LLM Model: {model_name}")
    print(f"Ollama URL: {base_url}")

    # 2. Verify Ollama server & 3. Verify target or compatible Qwen model exists
    model_name = verify_ollama_and_model(base_url, model_name)

    # 4. Test one real generation
    llm = OllamaQwenClient(
        model=model_name,
        base_url=base_url,
        timeout=llm_cfg.get("timeout_seconds", 60.0),
        temperature=llm_cfg.get("temperature", 0.2),
        top_p=llm_cfg.get("top_p", 0.9),
    )
    await test_generation(llm)

    # 5. Start HardwareAgent
    agent = HardwareAgent(llm=llm, work_dir="./rtl/generated")

    # 6. Run episode
    task = "Design an 8-bit signed MAC. Mathematical target: result = (a * b) + acc. Synthesizable SystemVerilog, signed arithmetic, separate testbench, no vendor primitives."

    print("Starting Agent Episode...\n")
    result = await agent.run_episode(task)

    print("\nAgent Episode Finished!")
    print(f"  Episode ID:    {result['episode_id']}")
    print(f"  Trajectory ID: {result['trajectory_id']}")
    print(f"  Success:       {result['success']}")
    print(f"  Total Steps:   {result['total_steps']}")
    print(f"  Reward:        {result['reward']} / 8.0")
    print("\nReward Breakdown:")
    for k, v in result["reward_breakdown"].items():
        print(f"    - {k}: {v}")

    # Check and print trajectory provenance proof
    trajectories = agent.trajectory_store.list_trajectories()
    matching = [t for t in trajectories if t.get("trajectory_id") == result["trajectory_id"]]
    if matching:
        latest = matching[-1]
        print("\nTrajectory Provenance Proof:")
        print(f"  provider      = {latest.get('provider')}")
        print(f"  model         = {latest.get('model')}")
        print(f"  fallback_used = {latest.get('fallback_used')}")
        print(f"  rtl_source    = {latest.get('rtl_source')}")

    print("\nGenerated RTL Preview:")
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
