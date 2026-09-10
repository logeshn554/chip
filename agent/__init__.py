"""Agent core package with lazy module loading to prevent circular imports."""

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "HardwareAgent":
        from agent.agent import HardwareAgent
        return HardwareAgent
    elif name in ("HardwarePlanner", "Planner"):
        from agent.planner import HardwarePlanner
        return HardwarePlanner
    elif name == "AgentLoop":
        from agent.agent_loop import AgentLoop
        return AgentLoop
    elif name == "ActionRouter":
        from agent.action_router import ActionRouter
        return ActionRouter
    elif name == "QwenClient":
        from agent.qwen import QwenClient
        return QwenClient
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
    "Planner",
    "AgentLoop",
    "ActionRouter",
    "QwenClient",
    "AgentState",
    "SYSTEM_PROMPT",
]
