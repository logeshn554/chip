"""
Qwen LLM Client for Local Ollama backend.

Targets Qwen-14B (qwen2.5:14b) through local Ollama API (http://localhost:11434).
Includes JSON extraction, reasoning parsing, strict error handling, and model verification.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Optional
import urllib.error
import urllib.request

from llm.interface import LLMInterface, LLMResponse, Message, get_default_model
from utils.device import get_ollama_gpu_options

logger = logging.getLogger(__name__)


class OllamaQwenClient(LLMInterface):
    """Local Qwen client communicating with Ollama REST API."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: str = "http://localhost:11434",
        timeout: Optional[float] = None,
        temperature: float = 0.2,
        top_p: float = 0.9,
        mock_mode: bool = False,
    ):
        self.model = get_default_model(model)
        self.base_url = base_url.rstrip("/")
        default_timeout = float(os.environ.get("LLM_TIMEOUT", "600.0"))
        self.timeout = timeout if timeout is not None else default_timeout
        self.temperature = temperature
        self.top_p = top_p
        self.mock_mode = mock_mode
        self._mock_responses: dict[str, Any] = {}

    def _is_mock_enabled(self) -> bool:
        """Mock behavior exists strictly when explicitly requested."""
        return bool(self.mock_mode or os.environ.get("LLM_PROVIDER") == "mock")

    def set_mock_response(self, prompt_substring: str, response: str | dict[str, Any]) -> None:
        """Register a mock response for testing."""
        self._mock_responses[prompt_substring] = response

    def verify_model_installed(self) -> None:
        """Verify model is installed in Ollama before proceeding."""
        if self._is_mock_enabled():
            return

        tags_url = f"{self.base_url}/api/tags"
        req = urllib.request.Request(tags_url)
        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Ollama server is unreachable at {self.base_url}. "
                f"Ensure Ollama is running (`ollama serve`). Error: {e}"
            )

        installed_names = []
        if isinstance(data, dict) and "models" in data:
            for m in data["models"]:
                if "name" in m:
                    installed_names.append(m["name"])
                if "model" in m:
                    installed_names.append(m["model"])

        target = self.model.lower()
        matched = any(
            target == name.lower()
            or name.lower().startswith(f"{target}:")
            or f"{target}:latest" == name.lower()
            or (":" not in target and name.lower().split(":")[0] == target)
            for name in installed_names
        )

        if not matched:
            raise RuntimeError(f"Model {self.model} is not installed in Ollama. Run: ollama pull {self.model}")

    def count_tokens(self, text: str) -> int:
        """Heuristic token count: ~4 chars per token for English / SystemVerilog."""
        return max(1, len(text) // 4)

    def _extract_thinking(self, text: str) -> tuple[str, str]:
        """Extract reasoning traces enclosed in <think> tags."""
        pattern = r"<think>(.*?)</think>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            thinking = match.group(1).strip()
            content = re.sub(pattern, "", text, flags=re.DOTALL).strip()
            return thinking, content
        return "", text.strip()

    def _extract_json(self, text: str) -> dict[str, Any]:
        """Extract structured JSON from model output."""
        cleaned = text.strip()
        _, cleaned = self._extract_thinking(cleaned)

        # Look for markdown code fence
        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        match = re.search(json_pattern, cleaned)
        candidate = match.group(1).strip() if match else cleaned

        # Direct parse attempt
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # Find first { and last }
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError as e:
                logger.warning(f"JSON regex extraction failed: {e}")

        return {"raw_output": text, "error": "Failed to parse JSON response"}

    async def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Send prompt to Ollama /api/generate."""
        if self._is_mock_enabled():
            for key, val in self._mock_responses.items():
                if key in prompt:
                    resp_str = json.dumps(val) if isinstance(val, dict) else str(val)
                    return LLMResponse(
                        text=resp_str,
                        metadata={
                            "provider": "mock",
                            "model": self.model,
                            "base_url": self.base_url,
                            "fallback_used": True,
                        },
                    )
            return LLMResponse(
                text="// Mock RTL generated for prompt\nmodule mac;\nendmodule",
                metadata={
                    "provider": "mock",
                    "model": self.model,
                    "base_url": self.base_url,
                    "fallback_used": True,
                },
            )

        model_name = kwargs.get("model", self.model)
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "num_predict": kwargs.get("max_tokens", 8192),
                **get_ollama_gpu_options(),
            },
        }

        url = f"{self.base_url}/api/generate"
        start_time = time.time()
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_bytes = resp.read()
                if not resp_bytes:
                    raise RuntimeError("Ollama returned empty response")
                try:
                    data = json.loads(resp_bytes.decode("utf-8"))
                except json.JSONDecodeError as jde:
                    raise RuntimeError(f"Ollama returned malformed JSON response: {jde}") from jde
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"Ollama request failed: HTTP {e.code}. URL={url}, model={self.model}"
            ) from e
        except TimeoutError as e:
            raise RuntimeError(
                f"Ollama generation timed out after {self.timeout}s (URL={url}, model={self.model}). "
                f"Consider increasing timeout_seconds or reducing prompt length."
            ) from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError) or "timed out" in str(e.reason).lower():
                raise RuntimeError(
                    f"Ollama generation timed out after {self.timeout}s (URL={url}, model={self.model}). "
                    f"Consider increasing timeout_seconds or reducing prompt length."
                ) from e
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. Ensure Ollama is running (`ollama serve`). Error: {e}"
            ) from e

        output_text = data.get("response")
        if output_text is None:
            raise RuntimeError("Ollama returned malformed response: missing 'response' field")

        latency_ms = (time.time() - start_time) * 1000.0

        # Observability event
        obs_record = {
            "provider": "ollama",
            "model": self.model,
            "endpoint": "/api/generate",
            "prompt_chars": len(prompt),
            "response_chars": len(output_text),
            "latency_ms": round(latency_ms, 2),
        }
        logger.info(f"[OBSERVABILITY] llm_request -> {json.dumps(obs_record)}")

        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        return LLMResponse(
            text=output_text,
            raw_response=data,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            finish_reason="stop" if data.get("done") else "length",
            metadata={
                "provider": "ollama",
                "model": self.model,
                "base_url": self.base_url,
                "fallback_used": False,
                "latency_ms": round(latency_ms, 2),
            },
        )

    async def chat(self, messages: list[Message | dict[str, str]], **kwargs: Any) -> LLMResponse:
        """Send messages to Ollama /api/chat."""
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, Message):
                formatted_messages.append({"role": msg.role, "content": msg.content})
            elif isinstance(msg, dict):
                formatted_messages.append(msg)

        if self._is_mock_enabled():
            last_content = formatted_messages[-1]["content"] if formatted_messages else ""
            for key, val in self._mock_responses.items():
                if key in last_content:
                    resp_str = json.dumps(val) if isinstance(val, dict) else str(val)
                    return LLMResponse(
                        text=resp_str,
                        metadata={
                            "provider": "mock",
                            "model": self.model,
                            "base_url": self.base_url,
                            "fallback_used": True,
                        },
                    )
            return LLMResponse(
                text="// Mock RTL\nmodule mac;\nendmodule",
                metadata={
                    "provider": "mock",
                    "model": self.model,
                    "base_url": self.base_url,
                    "fallback_used": True,
                },
            )

        model_name = kwargs.get("model", self.model)
        payload = {
            "model": model_name,
            "messages": formatted_messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "num_predict": kwargs.get("max_tokens", 8192),
            },
        }

        url = f"{self.base_url}/api/chat"
        start_time = time.time()
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_bytes = resp.read()
                if not resp_bytes:
                    raise RuntimeError("Ollama returned empty response")
                try:
                    data = json.loads(resp_bytes.decode("utf-8"))
                except json.JSONDecodeError as jde:
                    raise RuntimeError(f"Ollama returned malformed JSON response: {jde}") from jde
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"Ollama request failed: HTTP {e.code}. URL={url}, model={self.model}"
            ) from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. Ensure Ollama is running."
            ) from e

        message = data.get("message", {})
        output_text = message.get("content", "")
        latency_ms = (time.time() - start_time) * 1000.0

        obs_record = {
            "provider": "ollama",
            "model": self.model,
            "endpoint": "/api/chat",
            "prompt_chars": sum(len(m.get("content", "")) for m in formatted_messages),
            "response_chars": len(output_text),
            "latency_ms": round(latency_ms, 2),
        }
        logger.info(f"[OBSERVABILITY] llm_request -> {json.dumps(obs_record)}")

        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        return LLMResponse(
            text=output_text,
            raw_response=data,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            finish_reason="stop" if data.get("done") else "length",
            metadata={
                "provider": "ollama",
                "model": self.model,
                "base_url": self.base_url,
                "fallback_used": False,
                "latency_ms": round(latency_ms, 2),
            },
        )

    async def generate_json(
        self, prompt: str, schema: Optional[dict[str, Any]] = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Generate structured JSON adhering to the prompt or schema."""
        response = await self.generate(prompt, **kwargs)
        return self._extract_json(response.text)
