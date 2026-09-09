#!/usr/bin/env python3
"""
Quick-start script for running the agent on a single task.

Usage:
    python run_agent.py "Design a 4-bit ALU with add, subtract, AND, OR"
    python run_agent.py "Design a FIFO buffer with configurable depth"
"""

from __future__ import annotations

import asyncio
import sys

from rich.console import Console

console = Console()


async def main():
    if len(sys.argv) < 2:
        console.print("[bold]Usage:[/] python run_agent.py \"<task description>\"")
        console.print()
        console.print("[dim]Examples:[/]")
        console.print('  python run_agent.py "Design a 4-bit ALU with add, subtract, AND, OR"')
        console.print('  python run_agent.py "Design a parameterized FIFO buffer"')
        console.print('  python run_agent.py "Design a simple MAC unit for AI inference"')
        sys.exit(1)

    task = " ".join(sys.argv[1:])

    console.print(f"\n[bold cyan]🧠 Self-Evolving Chip Agent[/]")
    console.print(f"[dim]Task: {task}[/]\n")

    # Import and run via the CLI module
    import yaml
    from main import _run_agent, setup_logging

    setup_logging("INFO")

    # Load configs
    config_path = "configs/agent.yaml"
    prompts_path = "configs/prompts.yaml"

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        with open(prompts_path, "r") as f:
            prompts = yaml.safe_load(f)
    except FileNotFoundError as e:
        console.print(f"[red]Config file not found: {e}[/]")
        console.print("[dim]Run from the project root directory.[/]")
        sys.exit(1)

    await _run_agent(task, config, prompts)


if __name__ == "__main__":
    asyncio.run(main())
