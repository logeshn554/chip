"""
Qwen3-4B LLM Interface.

Provides a unified client for interacting with Qwen3-4B,
supporting both local HuggingFace Transformers inference
and remote vLLM OpenAI-compatible API.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)


class QwenClient:
    """Unified interface to Qwen3-4B for reasoning and generation.

    Supports two backends:
    - "transformers": Direct HuggingFace Transformers inference (local GPU)
    - "vllm": OpenAI-compatible API served by vLLM
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.backend = config.get("backend", "transformers")
        self.model_name = config.get("model_name", "Qwen/Qwen3-4B")
        self.max_new_tokens = config.get("max_new_tokens", 8192)
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.9)
        self.enable_thinking = config.get("enable_thinking", True)

        self._model = None
        self._tokenizer = None
        self._client = None  # OpenAI client for vLLM

        logger.info(f"QwenClient initialized — backend={self.backend}, model={self.model_name}")

    # ── Lazy Initialization ──────────────────────────────────────────

    def _init_transformers(self):
        """Load model and tokenizer via HuggingFace Transformers."""
        if self._model is not None:
            return

        logger.info(f"Loading model {self.model_name} via Transformers...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.config.get("torch_dtype", "auto"),
            device_map=self.config.get("device_map", "auto"),
        )
        logger.info(f"Model loaded on {self._model.device}")

    def _init_vllm(self):
        """Initialize OpenAI-compatible client for vLLM server."""
        if self._client is not None:
            return

        from openai import OpenAI

        vllm_cfg = self.config.get("vllm", {})
        self._client = OpenAI(
            base_url=vllm_cfg.get("base_url", "http://localhost:8000/v1"),
            api_key=vllm_cfg.get("api_key", "EMPTY"),
        )
        logger.info(f"vLLM client connected to {vllm_cfg.get('base_url')}")

    # ── Core Generation ──────────────────────────────────────────────

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> "QwenResponse":
        """Generate a response from Qwen3-4B.

        Args:
            messages: Chat messages in OpenAI format [{"role": ..., "content": ...}]
            temperature: Override default temperature
            max_tokens: Override default max tokens
            enable_thinking: Override thinking mode

        Returns:
            QwenResponse with thinking, content, and metadata
        """
        from agent.schemas import QwenResponse

        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_new_tokens
        thinking = enable_thinking if enable_thinking is not None else self.enable_thinking

        start_time = time.time()

        if self.backend == "transformers":
            response = await self._generate_transformers(messages, temp, max_tok, thinking)
        elif self.backend == "vllm":
            response = await self._generate_vllm(messages, temp, max_tok, thinking)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        response.generation_time_s = time.time() - start_time
        logger.debug(
            f"Generated response in {response.generation_time_s:.2f}s "
            f"({response.tokens_used} tokens)"
        )
        return response

    async def _generate_transformers(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        enable_thinking: bool,
    ) -> "QwenResponse":
        """Generate using local HuggingFace Transformers."""
        import asyncio

        from agent.schemas import QwenResponse

        self._init_transformers()

        # Apply chat template
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        # Run generation in thread pool to avoid blocking the event loop
        def _do_generate():
            import torch
            with torch.no_grad():
                generated_ids = self._model.generate(
                    **model_inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    top_p=self.top_p if temperature > 0 else None,
                    do_sample=temperature > 0,
                )
            return generated_ids

        generated_ids = await asyncio.get_event_loop().run_in_executor(None, _do_generate)

        # Decode — strip the input tokens
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        raw_output = self._tokenizer.decode(output_ids, skip_special_tokens=True)

        # Parse thinking vs content
        thinking, content = self._parse_thinking_output(raw_output)

        return QwenResponse(
            thinking=thinking,
            content=content,
            raw_output=raw_output,
            tokens_used=len(output_ids),
        )

    async def _generate_vllm(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        enable_thinking: bool,
    ) -> "QwenResponse":
        """Generate using vLLM OpenAI-compatible API."""
        import asyncio

        from agent.schemas import QwenResponse

        self._init_vllm()

        def _do_request():
            return self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=self.top_p,
            )

        completion = await asyncio.get_event_loop().run_in_executor(None, _do_request)

        raw_output = completion.choices[0].message.content or ""
        tokens_used = completion.usage.completion_tokens if completion.usage else 0

        thinking, content = self._parse_thinking_output(raw_output)

        return QwenResponse(
            thinking=thinking,
            content=content,
            raw_output=raw_output,
            tokens_used=tokens_used,
        )

    # ── Structured Output ────────────────────────────────────────────

    async def generate_structured(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Generate and parse a structured JSON response.

        Extracts JSON from the LLM's response, handling markdown code blocks.

        Returns:
            Parsed JSON dict. Returns {"error": "..."} on parse failure.
        """
        response = await self.generate(messages, temperature=temperature)
        return self.extract_json(response.content or response.raw_output)

    # ── Parsing Helpers ──────────────────────────────────────────────

    @staticmethod
    def _parse_thinking_output(raw: str) -> tuple[str, str]:
        """Separate <think>...</think> blocks from content.

        Qwen3 uses <think> tags to delimit reasoning from the final answer.
        """
        thinking = ""
        content = raw

        # Extract <think>...</think> block
        think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            content = raw[think_match.end():].strip()

        return thinking, content

    @staticmethod
    def extract_json(text: str) -> dict[str, Any]:
        """Extract JSON from LLM output, handling markdown code blocks.

        Tries multiple strategies:
        1. Direct JSON parse
        2. Extract from ```json ... ``` blocks
        3. Extract from ``` ... ``` blocks
        4. Find first { ... } substring
        """
        text = text.strip()

        # Strategy 1: Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: ```json blocks
        json_block = re.search(r"```json\s*\n?(.*?)\n?```", text, re.DOTALL)
        if json_block:
            try:
                return json.loads(json_block.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Strategy 3: Generic code blocks
        code_block = re.search(r"```\s*\n?(.*?)\n?```", text, re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Strategy 4: First { ... } match (greedy)
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning(f"Failed to extract JSON from LLM output: {text[:200]}...")
        return {"error": "Failed to parse JSON from LLM response", "raw": text}

    @staticmethod
    def extract_code(text: str, language: str = "systemverilog") -> str:
        """Extract code from markdown code blocks.

        Args:
            text: LLM output potentially containing code blocks
            language: Language identifier to look for

        Returns:
            Extracted code string, or the original text if no code block found
        """
        # Try language-specific block first
        pattern = rf"```{language}\s*\n?(.*?)\n?```"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Try generic code block
        generic = re.search(r"```\s*\n?(.*?)\n?```", text, re.DOTALL)
        if generic:
            return generic.group(1).strip()

        # Return as-is
        return text.strip()
