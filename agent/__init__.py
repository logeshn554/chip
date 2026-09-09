"""Agent core package with lazy module loading to prevent circular imports."""

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "HardwareAgent":
        from agent.agent import HardwareAgent
        return HardwareAgent
    elif name == "HardwarePlanner":
        from agent.planner import HardwarePlanner
        return HardwarePlanner
    elif name == "AgentState":
        from agent.state import AgentState
        return AgentState
    elif name == "SYSTEM_PROMPT":
        from agent.prompts import SYSTEM_PROMPT
        return SYSTEM_PROMPT
    raise AttributeError(f"module 'agent' has no attribute '{name}'")


__all__ = [
    "HardwareAgent",
    "HardwarePlanner",
    "AgentState",
    "SYSTEM_PROMPT",
]
