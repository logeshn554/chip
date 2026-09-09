"""
Self-Evolving Chip Agent — Agent Core Package.

The agent core orchestrates the Qwen3-4B reasoning LLM,
planning, action routing, and the main agent loop.
"""

from agent.qwen import QwenClient
from agent.planner import Planner
from agent.action_router import ActionRouter
from agent.agent_loop import AgentLoop

__all__ = ["QwenClient", "Planner", "ActionRouter", "AgentLoop"]
