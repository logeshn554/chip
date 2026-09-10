"""
Qwen3-4B LLM Interface — Unified, hardened client for hardware reasoning and generation.

Supports multiple backends:
- "transformers": Direct HuggingFace Transformers inference (local GPU/CPU)
- "vllm": OpenAI-compatible API served by vLLM
- "ollama": Local Ollama REST API (default port 11434)
- "mock" / offline fallback: Deterministic responses for offline testing

Includes:
- Robust multi-strategy JSON extraction with safe repair (trailing commas, comments)
- Configurable temperature, max tokens, and thinking mode (<think>...</think>)
- Timeout handling and clear error diagnostics
- No silent failures — invalid output produces structured error feedback
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from agent.schemas import QwenResponse
from llm.interface import LLMInterface, LLMResponse, Message, get_default_model

logger = logging.getLogger(__name__)


class QwenClient(LLMInterface):
    """Unified interface to Qwen LLM for hardware reasoning and generation."""

    def __init__(self, config: dict[str, Any] | None = None, **kwargs: Any):
        config = config or {}
        # Allow kwargs override config
        merged = {**config, **kwargs}
        self.config = merged

        self.backend = merged.get("backend", "ollama")
        self.model_name = get_default_model(merged.get("model_name", merged.get("model")))
        self.max_new_tokens = int(merged.get("max_new_tokens", merged.get("max_tokens", 8192)))
        self.temperature = float(merged.get("temperature", 0.7))
        self.top_p = float(merged.get("top_p", 0.9))
        self.enable_thinking = bool(merged.get("enable_thinking", True))
        self.timeout = float(merged.get("timeout", 60.0))
        self.mock_mode = bool(merged.get("mock_mode", False) or os.environ.get("LLM_PROVIDER") == "mock")

        self.ollama_host = merged.get("ollama_host", merged.get("ollama", {}).get("base_url", "http://localhost:11434"))

        # Lazy backends
        self._model = None
        self._tokenizer = None
        self._openai_client = None

        logger.info(
            f"QwenClient initialized — backend={self.backend}, model={self.model_name}, "
            f"temp={self.temperature}, timeout={self.timeout}s"
        )

    # ── Lazy Initializers ────────────────────────────────────────────

    def _init_transformers(self):
        """Load model and tokenizer via HuggingFace Transformers."""
        if self._model is not None:
            return

        logger.info(f"Loading model {self.model_name} via Transformers...")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.config.get("torch_dtype", "auto"),
                device_map=self.config.get("device_map", "auto"),
            )
            logger.info(f"Model loaded successfully on {self._model.device}")
        except Exception as e:
            logger.error(f"Failed to load Transformers model {self.model_name}: {e}")
            raise RuntimeError(f"Transformers loading error: {e}") from e

    def _init_vllm(self):
        """Initialize OpenAI-compatible client for vLLM server."""
        if self._openai_client is not None:
            return

        try:
            from openai import OpenAI

            vllm_cfg = self.config.get("vllm", {})
            base_url = vllm_cfg.get("base_url", "http://localhost:8000/v1")
            api_key = vllm_cfg.get("api_key", "EMPTY")
            self._openai_client = OpenAI(base_url=base_url, api_key=api_key)
            logger.info(f"vLLM client connected to {base_url}")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM client: {e}")
            raise RuntimeError(f"vLLM client initialization error: {e}") from e

    # ── Generation API ───────────────────────────────────────────────

    async def generate(
        self,
        messages: list[dict[str, str]] | list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> QwenResponse:
        """Generate a response from Qwen3-4B.

        Args:
            messages: Chat messages in OpenAI format or list of Message objects
            temperature: Override default temperature
            max_tokens: Override default max tokens
            enable_thinking: Override thinking mode

        Returns:
            QwenResponse with thinking, content, raw_output, and token counts
        """
        # Normalize messages to standard dicts
        normalized_messages: list[dict[str, str]] = []
        for m in messages:
            if isinstance(m, Message):
                normalized_messages.append({"role": m.role, "content": m.content})
            elif isinstance(m, dict):
                normalized_messages.append(m)
            else:
                normalized_messages.append({"role": "user", "content": str(m)})

        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_new_tokens
        thinking_enabled = enable_thinking if enable_thinking is not None else self.enable_thinking

        start_time = time.time()

        if self.mock_mode or self.backend == "mock":
            res = self._generate_mock(normalized_messages)
            res.generation_time_s = time.time() - start_time
            return res

        if self.backend == "ollama":
            res = await self._generate_ollama(normalized_messages, temp, max_tok)
        elif self.backend == "transformers":
            res = await self._generate_transformers(normalized_messages, temp, max_tok)
        elif self.backend == "vllm":
            res = await self._generate_vllm(normalized_messages, temp, max_tok)
        else:
            res = await self._generate_ollama(normalized_messages, temp, max_tok)

        # Process thinking tags if requested
        if thinking_enabled and not res.thinking:
            thinking, content = self._parse_thinking_output(res.raw_output)
            res.thinking = thinking
            res.content = content

        res.generation_time_s = time.time() - start_time
        return res

    def verify_model_installed(self) -> None:
        """Check Ollama connectivity and verify that model is installed."""
        if self.mock_mode or self.backend == "mock" or os.environ.get("LLM_PROVIDER") == "mock":
            return
        if self.backend != "ollama":
            return

        url = f"{self.ollama_host}/api/tags"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=min(self.timeout, 10.0)) as resp:
                resp_bytes = resp.read()
                if not resp_bytes:
                    raise RuntimeError("Ollama returned empty response for /api/tags")
                data = json.loads(resp_bytes.decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"Ollama request failed: HTTP {e.code}. URL={url}, model={self.model_name}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.ollama_host}. Ensure Ollama is running."
            ) from e

        models = data.get("models", [])
        installed_names = []
        for m in models:
            if isinstance(m, dict):
                if "name" in m:
                    installed_names.append(m["name"])
                if "model" in m:
                    installed_names.append(m["model"])

        target = self.model_name.lower()
        matched = any(
            target == name.lower()
            or name.lower().startswith(f"{target}:")
            or f"{target}:latest" == name.lower()
            or (":" not in target and name.lower().split(":")[0] == target)
            for name in installed_names
        )

        if not matched:
            raise RuntimeError(f"Model {self.model_name} is not installed in Ollama. Run: ollama pull {self.model_name}")

    async def complete(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        """Implementation of LLMInterface.complete."""
        resp = await self.generate(messages, **kwargs)
        return LLMResponse(
            content=resp.content,
            text=resp.content,
            raw_response=resp.raw_output,
            reasoning=resp.thinking,
            tokens_used=resp.tokens_used,
            model=self.model_name,
            metadata={
                "provider": self.backend,
                "model": self.model_name,
                "base_url": self.ollama_host,
                "fallback_used": bool(self.mock_mode or self.backend == "mock"),
            },
        )

    # ── Backend Implementations ──────────────────────────────────────

    async def _generate_ollama(
        self, messages: list[dict[str, str]], temp: float, max_tok: int
    ) -> QwenResponse:
        """Call Ollama chat endpoint."""
        url = f"{self.ollama_host}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": max_tok,
                "top_p": self.top_p,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        def _call_ollama():
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    resp_bytes = response.read()
                    if not resp_bytes:
                        raise RuntimeError("Ollama returned empty response")
                    return json.loads(resp_bytes.decode("utf-8"))
            except urllib.error.HTTPError as e:
                raise RuntimeError(
                    f"Ollama request failed: HTTP {e.code}. URL={url}, model={self.model_name}"
                ) from e
            except urllib.error.URLError as e:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self.ollama_host}. Ensure Ollama is running."
                ) from e

        loop = asyncio.get_event_loop()
        res_json = await loop.run_in_executor(None, _call_ollama)

        raw_output = res_json.get("message", {}).get("content", "")
        eval_count = res_json.get("eval_count", 0)
        thinking, content = self._parse_thinking_output(raw_output)

        return QwenResponse(
            thinking=thinking,
            content=content,
            raw_output=raw_output,
            tokens_used=eval_count,
        )

    async def _generate_transformers(
        self, messages: list[dict[str, str]], temp: float, max_tok: int
    ) -> QwenResponse:
        """Direct HuggingFace Transformers inference."""
        self._init_transformers()
        loop = asyncio.get_event_loop()

        def _do_generate():
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
            do_sample = temp > 0.0
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tok,
                temperature=temp if do_sample else None,
                top_p=self.top_p if do_sample else None,
                do_sample=do_sample,
            )
            generated_ids = outputs[0][len(inputs.input_ids[0]):]
            return self._tokenizer.decode(generated_ids, skip_special_tokens=True), len(generated_ids)

        raw_output, tokens = await loop.run_in_executor(None, _do_generate)
        thinking, content = self._parse_thinking_output(raw_output)

        return QwenResponse(
            thinking=thinking,
            content=content,
            raw_output=raw_output,
            tokens_used=tokens,
        )

    async def _generate_vllm(
        self, messages: list[dict[str, str]], temp: float, max_tok: int
    ) -> QwenResponse:
        """OpenAI-compatible vLLM API generation."""
        self._init_vllm()
        loop = asyncio.get_event_loop()

        def _do_request():
            return self._openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok,
                top_p=self.top_p,
                timeout=self.timeout,
            )

        completion = await loop.run_in_executor(None, _do_request)
        raw_output = completion.choices[0].message.content or ""
        tokens_used = completion.usage.completion_tokens if completion.usage else 0
        thinking, content = self._parse_thinking_output(raw_output)

        return QwenResponse(
            thinking=thinking,
            content=content,
            raw_output=raw_output,
            tokens_used=tokens_used,
        )

    def _generate_mock(self, messages: list[dict[str, str]]) -> QwenResponse:
        """Deterministic mock response generator for offline and testing environments."""
        last_msg = messages[-1]["content"] if messages else ""
        lower = last_msg.lower()

        thinking = "Analyzing hardware specifications and selecting the appropriate next action."

        if "syntax error" in lower or "mismatched" in lower or "unbalanced" in lower:
            action_dict = {
                "action": "EDIT_RTL",
                "params": {
                    "file": "mac.sv",
                    "code": (
                        "`timescale 1ns / 1ps\n"
                        "module mac #(\n"
                        "    parameter DATA_WIDTH = 8,\n"
                        "    parameter ACC_WIDTH = 32\n"
                        ") (\n"
                        "    input  logic                     clk,\n"
                        "    input  logic                     rst_n,\n"
                        "    input  logic                     valid_in,\n"
                        "    input  logic signed [DATA_WIDTH-1:0] a,\n"
                        "    input  logic signed [DATA_WIDTH-1:0] b,\n"
                        "    output logic signed [ACC_WIDTH-1:0]  accum,\n"
                        "    output logic                     valid_out\n"
                        ");\n"
                        "    always_ff @(posedge clk or negedge rst_n) begin\n"
                        "        if (!rst_n) begin\n"
                        "            accum     <= '0;\n"
                        "            valid_out <= 1'b0;\n"
                        "        end else if (valid_in) begin\n"
                        "            accum     <= accum + (a * b);\n"
                        "            valid_out <= 1'b1;\n"
                        "        end else begin\n"
                        "            valid_out <= 1'b0;\n"
                        "        end\n"
                        "    end\n"
                        "endmodule\n"
                    ),
                    "fix_notes": "Balanced begin/end blocks and verified signed qualifiers.",
                },
            }
        elif "cocotb" in lower and "passed" in lower:
            action_dict = {
                "action": "RUN_YOSYS",
                "params": {"file_path": "mac.sv", "top_module": "mac"},
            }
        elif "yosys" in lower and "passed" in lower:
            action_dict = {
                "action": "SAVE_DESIGN",
                "params": {"design_name": "mac", "version": "v1.0"},
            }
        elif "saved" in lower:
            action_dict = {
                "action": "COMPLETE",
                "params": {"summary": "8-bit signed MAC unit designed, verified, and synthesized."},
            }
        elif "mac" in lower:
            action_dict = {
                "action": "CREATE_RTL",
                "params": {
                    "filename": "mac.sv",
                    "code": (
                        "`timescale 1ns / 1ps\n"
                        "module mac #(\n"
                        "    parameter DATA_WIDTH = 8,\n"
                        "    parameter ACC_WIDTH = 32\n"
                        ") (\n"
                        "    input  logic                     clk,\n"
                        "    input  logic                     rst_n,\n"
                        "    input  logic                     valid_in,\n"
                        "    input  logic signed [DATA_WIDTH-1:0] a,\n"
                        "    input  logic signed [DATA_WIDTH-1:0] b,\n"
                        "    output logic signed [ACC_WIDTH-1:0]  accum,\n"
                        "    output logic                     valid_out\n"
                        ");\n"
                        "    always_ff @(posedge clk or negedge rst_n) begin\n"
                        "        if (!rst_n) begin\n"
                        "            accum     <= '0;\n"
                        "            valid_out <= 1'b0;\n"
                        "        end else if (valid_in) begin\n"
                        "            accum     <= accum + (a * b);\n"
                        "            valid_out <= 1'b1;\n"
                        "        end else begin\n"
                        "            valid_out <= 1'b0;\n"
                        "        end\n"
                        "    end\n"
                        "endmodule\n"
                    ),
                },
            }
        else:
            action_dict = {
                "action": "RETRIEVE_MEMORY",
                "params": {"query": last_msg[:100], "memory_type": "all"},
            }

        content = f"```json\n{json.dumps(action_dict, indent=2)}\n```"
        raw_output = f"<think>\n{thinking}\n</think>\n{content}"

        return QwenResponse(
            thinking=thinking,
            content=content,
            raw_output=raw_output,
            tokens_used=120,
        )

    # ── Structured Output & Robust JSON Extraction ───────────────────

    async def generate_structured(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Generate and safely parse a structured JSON response."""
        response = await self.generate(messages, temperature=temperature)
        return self.extract_json(response.content or response.raw_output)

    @staticmethod
    def _parse_thinking_output(raw: str) -> tuple[str, str]:
        """Separate <think>...</think> blocks from content."""
        thinking = ""
        content = raw

        think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            content = raw[think_match.end():].strip()

        return thinking, content

    @classmethod
    def extract_json(cls, text: str) -> dict[str, Any]:
        """Extract JSON from LLM output with multi-stage safe repair.

        Strategies:
        1. Direct JSON parse of trimmed text
        2. Extract from ```json ... ``` blocks
        3. Extract from ``` ... ``` blocks
        4. Outermost balanced { ... } substring search
        5. Safe repair:
           - Strip single-line comments (// ...)
           - Strip multi-line comments (/* ... */)
           - Remove trailing commas before } or ]
           - Repair common quote escaping issues

        Returns:
            Parsed dict on success.
            On failure, returns:
            {"error": "JSON_PARSE_ERROR", "message": str(err), "raw": text}
            NEVER defaults to COMPLETE or empty.
        """
        text = text.strip()
        if not text:
            return {"error": "JSON_PARSE_ERROR", "message": "Empty response from LLM", "raw": ""}

        # Strategy 1: Direct parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Strategy 2: ```json ... ``` block
        json_block = re.search(r"```json\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
        if json_block:
            candidate = json_block.group(1).strip()
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                repaired = cls._repair_json_string(candidate)
                if repaired is not None:
                    return repaired

        # Strategy 3: Generic ``` ... ``` block
        generic_block = re.search(r"```\s*\n?(.*?)\n?```", text, re.DOTALL)
        if generic_block:
            candidate = generic_block.group(1).strip()
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                repaired = cls._repair_json_string(candidate)
                if repaired is not None:
                    return repaired

        # Strategy 4: Outermost { ... } block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                repaired = cls._repair_json_string(candidate)
                if repaired is not None:
                    return repaired

        # Final repair attempt on original text
        repaired = cls._repair_json_string(text)
        if repaired is not None:
            return repaired

        logger.warning(f"Failed to parse JSON from LLM output: {text[:200]}...")
        return {
            "error": "JSON_PARSE_ERROR",
            "message": "Failed to parse valid JSON from LLM response after all repair attempts.",
            "raw": text,
        }

    @classmethod
    def _repair_json_string(cls, candidate: str) -> Optional[dict[str, Any]]:
        """Apply heuristics to repair mildly malformed JSON strings."""
        s = candidate.strip()
        # 1. Remove // single-line comments
        s = re.sub(r"//.*$", "", s, flags=re.MULTILINE)
        # 2. Remove /* multi-line comments */
        s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
        # 3. Remove trailing commas before closing braces/brackets
        s = re.sub(r",\s*([\]}])", r"\1", s)
        # 4. Wrap single quotes around unquoted keys if necessary
        # 5. Extract innermost or outermost { ... }
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            s = s[start:end + 1]

        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        return None

    @staticmethod
    def extract_code(text: str, language: str = "systemverilog") -> str:
        """Extract code from markdown code blocks."""
        pattern = rf"```{language}\s*\n?(.*?)\n?```"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        generic = re.search(r"```\s*\n?(.*?)\n?```", text, re.DOTALL)
        if generic:
            return generic.group(1).strip()

        return text.strip()
