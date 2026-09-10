"""
Abstract LLM interface for the Hardware Design Agent.
Decouples agent reasoning and planning from specific LLM providers/backends.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

DEFAULT_MODEL_NAME: str = os.environ.get("LLM_MODEL", "qwen2.5:14b")


def get_default_model(configured_name: Optional[str] = None) -> str:
    """Return canonical LLM model name, normalized across aliases."""
    raw = configured_name or os.environ.get("LLM_MODEL", "qwen2.5:14b")
    raw_clean = raw.lower().replace("-", "").replace("_", "")
    if "14b" in raw_clean:
        return "qwen2.5:14b"
    return raw


@dataclass
class Message:
    """Chat message structure."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """Standardized response from the LLM."""
    text: str
    raw_response: Optional[dict[str, Any]] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = "stop"
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMInterface(ABC):
    """Abstract base class for LLM client implementations."""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate a completion for the given prompt."""
        pass

    @abstractmethod
    async def chat(self, messages: list[Message | dict[str, str]], **kwargs: Any) -> LLMResponse:
        """Generate a response given a list of chat messages."""
        pass

    @abstractmethod
    async def generate_json(
        self, prompt: str, schema: Optional[dict[str, Any]] = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Generate structured JSON response adhering to an optional schema."""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Estimate token count for context budgeting."""
        pass
