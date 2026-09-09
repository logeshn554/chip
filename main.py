"""
Self-Evolving Chip Agent — Main CLI Entry Point.

Commands:
    run       Run the agent on a hardware design task
    ingest    Ingest knowledge documents into memory
    evaluate  Score an existing design
    trajectory View trajectory statistics
    status    Show system status
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

import click
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

console = Console()

# Load environment variables
load_dotenv()


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with Rich handler."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def load_config(config_path: str = "configs/agent.yaml") -> dict:
    """Load the master configuration file."""
    if not os.path.exists(config_path):
        console.print(f"[red]Config not found: {config_path}[/]")
        sys.exit(1)

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_prompts(prompts_path: str = "configs/prompts.yaml") -> dict:
    """Load prompt templates."""
    if not os.path.exists(prompts_path):
        return {}

    with open(prompts_path, "r") as f:
        return yaml.safe_load(f)


# ── CLI ──────────────────────────────────────────────────────────────

@click.group()
@click.option("--config", default="configs/agent.yaml", help="Path to config file")
@click.option("--log-level", default="INFO", help="Logging level")
@click.pass_context
def cli(ctx, config, log_level):
    """🧠 Self-Evolving Chip Design Agent"""
    setup_logging(log_level)
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)
    ctx.obj["prompts"] = load_prompts()


@cli.command()
@click.argument("task")
@click.option("--max-iterations", "-n", default=None, type=int, help="Max iterations")
@click.pass_context
def run(ctx, task, max_iterations):
    """Run the agent on a hardware design task.

    Example:
        chip-agent run "Design a 4-bit ALU with add, subtract, AND, OR"
    """
    config = ctx.obj["config"]
    prompts = ctx.obj["prompts"]

    if max_iterations:
        config.setdefault("agent", {})["max_iterations"] = max_iterations

    asyncio.run(_run_agent(task, config, prompts))


async def _run_agent(task: str, config: dict, prompts: dict):
    """Initialize all components and run the agent loop."""
    from agent.qwen import QwenClient
    from agent.planner import Planner
    from agent.action_router import build_router
    from agent.agent_loop import AgentLoop
    from memory import MemorySystem
    from memory.trajectory import TrajectoryStore
    from web.search import WebSearcher
    from rtl.generator import RTLGenerator
    from tools.verilator import VerilatorTool
    from tools.yosys import YosysTool
    from evaluator.reward import RewardEngine

    console.print("[bold cyan]Initializing Self-Evolving Chip Agent...[/]\n")

    # Initialize components
    qwen = QwenClient(config.get("llm", {}))
    memory = MemorySystem(config.get("memory", {}))
    trajectory_store = TrajectoryStore(config.get("memory", {}).get("trajectory", {}))
    web_searcher = WebSearcher(config.get("web", {}))

    rtl_generator = RTLGenerator(
        qwen=qwen,
        prompts=prompts,
        output_dir=config.get("memory", {}).get("design", {}).get("designs_dir", "./designs"),
    )

    tools_config = config.get("tools", {})
    verilator = VerilatorTool(tools_config.get("verilator", {}))
    yosys = YosysTool(tools_config.get("yosys", {}))

    # Build action router
    router = build_router(
        rtl_generator=rtl_generator,
        web_searcher=web_searcher,
        memory_system=memory,
        verilator=verilator,
        yosys=yosys,
    )

    planner = Planner(qwen=qwen, prompts=prompts)
    reward_engine = RewardEngine(config.get("evaluator", {}))

    # Create agent loop
    agent = AgentLoop(
        qwen=qwen,
        planner=planner,
        router=router,
        memory=memory,
        evaluator=None,  # TODO: Wire up full evaluator
        trajectory_store=trajectory_store,
        config=config.get("agent", {}),
    )

    # Run
    episode = await agent.run(task)

    console.print(f"\n[bold]Episode ID:[/] {episode.episode_id}")
    console.print(f"[bold]Final Reward:[/] {episode.final_reward:.3f}")
    console.print(f"[bold]Success:[/] {episode.success}")


@cli.command()
@click.argument("path")
@click.option("--type", "-t", "doc_type", default="general", help="Document type")
@click.pass_context
def ingest(ctx, path, doc_type):
    """Ingest knowledge documents into memory.

    Example:
        chip-agent ingest ./docs/systemverilog_reference.md --type knowledge
    """
    config = ctx.obj["config"]

    from memory import MemorySystem

    memory = MemorySystem(config.get("memory", {}))

    if os.path.isdir(path):
        count = memory.knowledge.ingest_directory(
            path, metadata={"type": doc_type}
        )
    elif os.path.isfile(path):
        count = memory.knowledge.ingest_file(
            path, metadata={"type": doc_type}
        )
    else:
        console.print(f"[red]Path not found: {path}[/]")
        return

    console.print(f"[green]✓ Ingested {count} chunks from {path}[/]")


@cli.command()
@click.pass_context
def trajectory(ctx):
    """View trajectory statistics."""
    config = ctx.obj["config"]

    from memory.trajectory import TrajectoryStore

    store = TrajectoryStore(config.get("memory", {}).get("trajectory", {}))
    stats = store.get_stats()

    from rich.table import Table

    table = Table(title="Trajectory Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    for key, value in stats.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.3f}")
        else:
            table.add_row(key, str(value))

    console.print(table)


@cli.command()
@click.pass_context
def status(ctx):
    """Show system status and component health."""
    config = ctx.obj["config"]

    console.print("[bold cyan]System Status[/]\n")

    # Check LLM
    llm_cfg = config.get("llm", {})
    console.print(f"  LLM Backend: {llm_cfg.get('backend', 'unknown')}")
    console.print(f"  Model: {llm_cfg.get('model_name', 'unknown')}")

    # Check tools
    import shutil
    for tool in ["verilator", "yosys", "make", "git"]:
        found = shutil.which(tool)
        icon = "✓" if found else "✗"
        color = "green" if found else "red"
        console.print(f"  [{color}]{icon}[/] {tool}: {found or 'not found'}")

    # Check memory
    chroma_path = config.get("memory", {}).get("chroma_path", "./data/chroma")
    console.print(f"  ChromaDB: {chroma_path} ({'exists' if os.path.exists(chroma_path) else 'not initialized'})")

    # Check directories
    for dir_name in ["designs", "trajectories", "knowledge_base"]:
        path = f"./{dir_name}"
        exists = os.path.exists(path)
        console.print(f"  {dir_name}/: {'exists' if exists else 'will be created'}")


if __name__ == "__main__":
    cli()
