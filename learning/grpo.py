"""
GRPO Trainer — Group Relative Policy Optimization with Hardware Environment Evaluation.

Trains Qwen3-4B via policy optimization using genuine EDA verification feedback:
- Verilator syntax and lint (hard gate: compile failure = 0.0)
- Cocotb functional verification pass rate
- SymbiYosys formal assertion checking
- Yosys logic synthesis cell count and area
- Grounded multi-objective reward engine

Eliminates proxy completion length rewards in favor of grounded hardware evaluation.
Computes group relative advantages: A_i = (R_i - mean(R)) / (std(R) + eps).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import shutil
from typing import Any, Optional

from evaluator.reward import RewardEngine
from tools.verilator import VerilatorTool
from tools.cocotb import CocotbTool
from tools.yosys import YosysTool
from tools.formal import FormalVerificationTool

logger = logging.getLogger(__name__)


def compute_group_advantages(rewards: list[float], eps: float = 1e-4) -> list[float]:
    """Compute Group Relative Policy Optimization (GRPO) advantages:
    
    A_i = (R_i - mean(R)) / (std(R) + eps)
    
    This is the mathematical formula of GRPO advantage normalization.
    - Used directly in standalone custom policy gradient rollout loops.
    - When delegating to HuggingFace TRL's GRPOTrainer, TRL handles internal group advantage
      calculation across generated candidate completions, using HardwareRewardEvaluator as
      its reward_funcs callback.
    """
    if not rewards:
        return []
    if len(rewards) == 1:
        return [0.0]

    import statistics
    mean_r = statistics.mean(rewards)
    try:
        std_r = statistics.stdev(rewards)
    except Exception:
        std_r = 0.0

    return [(r - mean_r) / (std_r + eps) for r in rewards]


def normalize_completion_text(completion: Any) -> str:
    """Normalize TRL / conversational completions into plain text strings.
    
    Supports:
    - Plain strings: "module mac ... endmodule"
    - TRL conversational message lists: [{"role": "assistant", "content": "..."}]
    - Single message dicts: {"content": "...", ...} or {"text": "..."}
    - Nested message structures or token representations
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        if "content" in completion:
            return normalize_completion_text(completion["content"])
        if "text" in completion:
            return normalize_completion_text(completion["text"])
        return str(completion)
    if isinstance(completion, (list, tuple)):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content", "")
                if content:
                    parts.append(normalize_completion_text(content))
                elif "text" in item:
                    parts.append(normalize_completion_text(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)


class HardwareRewardEvaluator:
    """Evaluates candidate model completions using real hardware EDA tools.
    
    Replaces heuristic length proxies with true hardware verification:
    1. Verilator compile & lint (hard gate: 0 if failed)
    2. Cocotb / testbench functional correctness
    3. Yosys cell count and area estimation
    4. SymbiYosys formal verification
    5. RewardEngine multi-objective normalization
    """

    def __init__(
        self,
        top_module: Optional[str] = None,
        test_file: Optional[str] = None,
        work_dir: str = "./sim_build/grpo_eval",
        target_cells: float = 500.0,
        target_timing_ns: float = 5.0,
        enable_synthesis: bool = True,
        enable_formal: bool = True,
        formal_properties: Optional[str] = None,
        benchmark_task_id: Optional[str] = None,
        max_concurrency: int = 4,
        evaluation_mode: str = "research_fast",
        require_exact_task: bool = False,
    ):
        self.default_top_module = top_module or "mac"
        self.default_test_file = test_file or "tests/mac/test_mac.py"
        self.work_dir = work_dir
        self.target_cells = target_cells
        self.target_timing_ns = target_timing_ns
        self.enable_synthesis = enable_synthesis
        self.enable_formal = enable_formal
        self.default_formal_properties = formal_properties
        self.default_benchmark_task_id = benchmark_task_id
        self.max_concurrency = max_concurrency
        self.evaluation_mode = evaluation_mode
        self.require_exact_task = require_exact_task

        os.makedirs(work_dir, exist_ok=True)

        self.verilator = VerilatorTool(work_dir=os.path.join(work_dir, "sim"))
        self.cocotb = CocotbTool(sim_dir=os.path.join(work_dir, "sim"))
        self.yosys = YosysTool(work_dir=os.path.join(work_dir, "synth"))
        self.formal = FormalVerificationTool(work_dir=os.path.join(work_dir, "formal"))
        self.reward_engine = RewardEngine()

    def resolve_benchmark_task(
        self,
        task_id: Optional[str] = None,
        prompt: Optional[Any] = None,
        code: Optional[Any] = None,
        completion: Optional[Any] = None,
        require_exact: Optional[bool] = None,
    ) -> Optional[Any]:
        """Dynamically resolve the benchmark task configuration.
        
        When require_exact is True (recommended for training), only exact match on task_id
        is accepted. Fuzzy regex matching is disabled to prevent evaluating against the wrong benchmark.
        """
        try:
            from benchmarks.curriculum import BenchmarkCurriculum
            curriculum = BenchmarkCurriculum()
        except Exception:
            return None

        exact = self.require_exact_task if require_exact is None else require_exact
        if exact:
            return curriculum.get_task(task_id) if task_id else None

        prompt_str = normalize_completion_text(prompt) if prompt is not None else ""
        code_text = normalize_completion_text(code or completion) if (code or completion) is not None else ""

        # 1. Direct task ID match
        if task_id:
            t = curriculum.get_task(task_id)
            if t:
                return t

        # 2. Extract benchmark ID from prompt (e.g. "[L1_NOT_GATE", "L3_MAC_8BIT_SIGNED", "L2_ALU_4BIT")
        if prompt:
            id_match = re.search(r"\b(L[1-7]_[A-Z0-9_]+)\b", prompt_str)
            if id_match:
                t = curriculum.get_task(id_match.group(1))
                if t:
                    return t

            # Match by module name in prompt
            mod_match = re.search(r"Module Name:\s*([a-zA-Z0-9_]+)", prompt_str)
            if mod_match:
                target_m = mod_match.group(1).lower()
                for task in curriculum._tasks.values():
                    if task.top_module.lower() == target_m:
                        return task

            # Match any task by ID or top module keyword in prompt
            for task in curriculum._tasks.values():
                if task.id in prompt_str or task.top_module.lower() in prompt_str.lower():
                    return task

        # 3. Extract module name from candidate RTL code
        if code_text:
            code_mod = re.search(r"\bmodule\s+([a-zA-Z0-9_]+)", code_text)
            if code_mod:
                target_m = code_mod.group(1).lower()
                target_norm = target_m.replace("_", "")
                for task in curriculum._tasks.values():
                    task_norm = task.top_module.lower().replace("_", "")
                    if (
                        task.top_module.lower() == target_m
                        or task_norm == target_norm
                        or target_norm in task.id.lower().replace("_", "")
                    ):
                        return task

        # 4. Fall back to default benchmark if set
        if self.default_benchmark_task_id:
            return curriculum.get_task(self.default_benchmark_task_id)

        return None

    def extract_rtl(self, text: Any) -> str:
        """Extract SystemVerilog code from model completion output.
        
        Handles plain strings, TRL conversational message dicts, and message lists.
        """
        raw_text = normalize_completion_text(text)

        # 1. Look for ```systemverilog ... ``` block
        sv_match = re.search(r"```(?:systemverilog|verilog)\s*\n(.*?)```", raw_text, re.DOTALL | re.IGNORECASE)
        if sv_match:
            return sv_match.group(1).strip()

        # 2. Look for generic ``` ... ``` block containing module
        generic_match = re.search(r"```\s*\n(.*?)```", raw_text, re.DOTALL)
        if generic_match and "module " in generic_match.group(1):
            return generic_match.group(1).strip()

        # 3. Direct module declaration
        mod_match = re.search(r"(\bmodule\s+.*?\bendmodule\b)", raw_text, re.DOTALL)
        if mod_match:
            return mod_match.group(1).strip()

        # 4. JSON action containing 'code' in params
        json_match = re.search(r'"code"\s*:\s*"([^"]+)"', raw_text)
        if json_match:
            return json_match.group(1).replace("\\n", "\n").replace('\\"', '"')

        return ""

    async def evaluate_completion_async(
        self,
        completion: Any,
        benchmark_task_id: Optional[str] = None,
        prompt: Optional[Any] = None,
        sub_work_dir: Optional[str] = None,
    ) -> dict[str, Any]:
        """Evaluate a single completion asynchronously using EDA tools for the resolved task."""
        code = self.extract_rtl(completion)
        if not code:
            return {
                "reward": 0.0,
                "compile_pass": False,
                "functional_pass": False,
                "error": "No synthesizable SystemVerilog module found in completion.",
            }

        prompt_str = normalize_completion_text(prompt) if prompt is not None else None
        # Resolve benchmark task dynamically (mixed curriculum support)
        task_obj = self.resolve_benchmark_task(task_id=benchmark_task_id, prompt=prompt_str, code=code)

        if self.require_exact_task and not task_obj:
            return {
                "reward": 0.0,
                "compile_pass": False,
                "functional_pass": False,
                "error": f"Mandatory benchmark_task_id '{benchmark_task_id}' could not be resolved in exact mode.",
                "task_id": benchmark_task_id,
                "top_module": self.default_top_module,
            }

        top_module = getattr(task_obj, "top_module", self.default_top_module)
        formal_properties = getattr(task_obj, "formal_properties", self.default_formal_properties)
        test_file = self.default_test_file

        raw_completion = normalize_completion_text(completion)
        cand_hash = hashlib.md5((raw_completion + (task_obj.id if task_obj else "")).encode("utf-8")).hexdigest()[:8]
        dir_name = sub_work_dir or f"cand_{cand_hash}"
        cand_dir = os.path.join(self.work_dir, dir_name)
        os.makedirs(cand_dir, exist_ok=True)
        rtl_path = os.path.join(cand_dir, f"{top_module}.sv")

        with open(rtl_path, "w", encoding="utf-8") as f:
            f.write(code)

        # Stage 1: Verilator Lint & Syntax Compile Check (Hard Gate)
        compile_res = await self.verilator.lint_and_compile(rtl_path, top_module=top_module)
        compile_pass = compile_res.get("status") == "passed"
        if not compile_pass:
            return {
                "reward": 0.0,
                "compile_pass": False,
                "functional_pass": False,
                "error": compile_res.get("error", "Compilation failed"),
                "task_id": getattr(task_obj, "id", None),
                "top_module": top_module,
            }

        # Stage 2: Cocotb / testbench functional verification with benchmark task vectors
        func_res = await self.cocotb.run_tests(
            rtl_path,
            testbench_path=test_file if (os.path.exists(test_file) and top_module == "mac") else None,
            benchmark_task=task_obj,
        )
        total_tests = func_res.get("tests_total", 0)
        passed_tests = func_res.get("tests_passed", 0)
        pass_rate = (passed_tests / total_tests) if total_tests > 0 else (1.0 if func_res.get("status") == "passed" else 0.0)

        # Stage 3: Logic Synthesis (Yosys) - strictly reject heuristic metrics during training
        area: Optional[float] = None
        if self.enable_synthesis:
            synth_res = await self.yosys.synthesize(
                rtl_path,
                top_module=top_module,
                allow_heuristic_fallback=False,
            )
            # NEVER use heuristic cell/area estimates for training reward
            if synth_res.get("status") == "passed" and synth_res.get("metric_type") == "actual":
                area = float(synth_res.get("cells", 0) or 0)
            elif synth_res.get("status") == "failed" and synth_res.get("metric_type") == "actual":
                # Real synthesis failed on candidate
                area = -1.0

        # Stage 4: Formal Verification (SymbiYosys) using task's external benchmark properties
        formal_status = "SKIPPED"
        if self.enable_formal:
            formal_res = await self.formal.verify(
                rtl_path,
                top_module=top_module,
                external_properties=formal_properties,
            )
            formal_status = formal_res.get("status", "SKIPPED")

        # Grounded Reward Calculation with Weight Re-normalization
        effective_area_target = getattr(task_obj, "target_cells", None) or self.target_cells
        effective_timing_target = getattr(task_obj, "target_timing_ns", None) or self.target_timing_ns

        reward_result = self.reward_engine.compute_modular_reward(
            compile_success=compile_pass,
            test_pass_rate=pass_rate,
            formal_status=formal_status,
            area=area,
            area_target=effective_area_target,
            timing_target=effective_timing_target,
            evaluation_mode=self.evaluation_mode,
        )

        return {
            "reward": reward_result.normalized_reward,
            "raw_reward": reward_result.total_reward,
            "compile_pass": compile_pass,
            "pass_rate": pass_rate,
            "formal_status": formal_status,
            "area": area,
            "breakdown": reward_result.breakdown,
            "task_id": getattr(task_obj, "id", None),
            "top_module": top_module,
        }

    def evaluate_completion(self, completion: Any, benchmark_task_id: Optional[str] = None, prompt: Optional[Any] = None) -> float:
        """Synchronous wrapper returning normalized reward in [0.0, 1.0]."""
        res = asyncio.run(self.evaluate_completion_async(completion, benchmark_task_id=benchmark_task_id, prompt=prompt))
        return float(res["reward"])

    async def evaluate_batch_async(
        self,
        completions: list[Any],
        prompts: Optional[list[Any]] = None,
        task_ids: Optional[list[str] | str] = None,
        **kwargs: Any,
    ) -> list[float]:
        """Concurrently evaluate candidate completions in isolated sub-workspaces with bounded parallelism.
        
        Fully compatible with Hugging Face TRL GRPOTrainer reward callback format:
        - Completions can be plain strings or conversational message dicts: [{"content": "..."}]
        - Prompts can be plain strings or message dicts
        - Extra dataset columns (e.g. benchmark_task_id) passed via kwargs are routed to each candidate
        """
        if not completions:
            return []

        # Extract and align task IDs (from task_ids or dataset column kwargs)
        raw_tids = (
            task_ids
            or kwargs.get("benchmark_task_id")
            or kwargs.get("benchmark_id")
            or kwargs.get("task_id")
        )
        resolved_tids: list[Optional[str]] = []
        if isinstance(raw_tids, str):
            resolved_tids = [raw_tids] * len(completions)
        elif isinstance(raw_tids, (list, tuple)):
            if len(raw_tids) == len(completions):
                resolved_tids = [str(t) if t is not None else None for t in raw_tids]
            elif len(raw_tids) > 0 and len(completions) % len(raw_tids) == 0:
                # Repeated candidate generations per prompt (num_generations group size G)
                group_ratio = len(completions) // len(raw_tids)
                resolved_tids = [str(t) if t is not None else None for t in raw_tids for _ in range(group_ratio)]
            else:
                resolved_tids = [str(t) if t is not None else None for t in raw_tids]
        else:
            resolved_tids = [None] * len(completions)

        # Extract and align prompts
        raw_prompts = prompts if prompts is not None else kwargs.get("prompts")
        resolved_prompts: list[Optional[str]] = []
        if raw_prompts:
            if len(raw_prompts) == len(completions):
                resolved_prompts = [normalize_completion_text(p) for p in raw_prompts]
            elif len(raw_prompts) > 0 and len(completions) % len(raw_prompts) == 0:
                group_ratio = len(completions) // len(raw_prompts)
                resolved_prompts = [normalize_completion_text(p) for p in raw_prompts for _ in range(group_ratio)]
            else:
                resolved_prompts = [normalize_completion_text(p) for p in raw_prompts]
        else:
            resolved_prompts = [None] * len(completions)

        concurrency = kwargs.get("max_concurrency") or self.max_concurrency
        sem = asyncio.Semaphore(concurrency)

        async def _eval_one(completion: Any, idx: int) -> float:
            async with sem:
                pr = resolved_prompts[idx] if idx < len(resolved_prompts) else None
                tid = resolved_tids[idx] if idx < len(resolved_tids) else None
                raw_text = normalize_completion_text(completion)
                cand_hash = hashlib.md5(raw_text.encode("utf-8")).hexdigest()[:6]
                sub_dir = f"worker_{idx}_{cand_hash}"
                res = await self.evaluate_completion_async(
                    completion,
                    benchmark_task_id=tid,
                    prompt=pr,
                    sub_work_dir=sub_dir,
                )
                return float(res.get("reward", 0.0))

        tasks = [_eval_one(c, i) for i, c in enumerate(completions)]
        return await asyncio.gather(*tasks)

    def evaluate_batch(
        self,
        completions: list[Any],
        prompts: Optional[list[Any]] = None,
        task_ids: Optional[list[str] | str] = None,
        **kwargs: Any,
    ) -> list[float]:
        """Synchronous wrapper for concurrent batch evaluation."""
        return asyncio.run(
            self.evaluate_batch_async(
                completions,
                prompts=prompts,
                task_ids=task_ids,
                **kwargs,
            )
        )


class GRPOTrainer:
    """Group Relative Policy Optimization for Qwen3-4B with Hardware Evaluator.

    GRPO evaluates groups of G candidate responses per prompt against the real EDA
    environment concurrently, computes group-relative advantages, and updates policy weights.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.config = config
        self.model_name = config.get("model_name", "Qwen/Qwen3-4B")
        self.group_size = config.get("group_size", 4)
        self.learning_rate = config.get("learning_rate", 1e-5)
        self.kl_coeff = config.get("kl_coeff", 0.1)
        self.temperature = config.get("temperature", 0.6)
        self.top_p = config.get("top_p", 0.95)
        self.top_k = config.get("top_k", 20)
        self.output_dir = config.get("output_dir", "./models/grpo")

        # Hardware evaluator with training-hardened settings
        self.evaluator = HardwareRewardEvaluator(
            top_module=config.get("top_module"),
            test_file=config.get("test_file"),
            target_cells=config.get("target_cells", 500.0),
            max_concurrency=config.get("max_concurrency", 4),
            evaluation_mode=config.get("evaluation_mode", "research_fast"),
            require_exact_task=config.get("require_exact_task", True),
        )

    def compute_step_advantages(
        self,
        completions: list[Any],
        prompts: Optional[list[Any]] = None,
        task_id: Optional[str] = None,
    ) -> tuple[list[float], list[float]]:
        """Evaluate a group of candidate completions concurrently and compute their relative advantages.
        
        Returns:
            (rewards, advantages): tuple of grounded EDA rewards in [0, 1] and normalized advantages.
        """
        rewards = self.evaluator.evaluate_batch(completions, prompts=prompts, benchmark_task_id=task_id)
        advantages = compute_group_advantages(rewards)
        return rewards, advantages

    async def train(self, dataset_path: str) -> dict[str, Any]:
        """Run GRPO training on a hardware prompt dataset.

        Args:
            dataset_path: Path to the prompt dataset (JSONL with 'prompt' column
                          and mandatory 'benchmark_task_id' column)

        Returns:
            Training results dict
        """
        logger.info("=" * 60)
        logger.info("GRPO Training — Hardware-in-the-Loop Environment")
        logger.info("=" * 60)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Group size: {self.group_size}")
        logger.info(f"KL coefficient: {self.kl_coeff}")
        logger.info(f"Qwen Generation: temp={self.temperature}, top_p={self.top_p}, top_k={self.top_k}")

        try:
            return self._run_training(dataset_path)
        except ImportError as e:
            logger.warning(
                f"Training dependencies not installed: {e}\n"
                f"Install with: pip install self-evolving-chip[learning]"
            )
            return {
                "error": f"Missing dependency: {e}",
                "evaluator_ready": True,
                "reward_type": "grounded_hardware_eda",
            }

    def _run_training(self, dataset_path: str) -> dict[str, Any]:
        """Execute GRPO training with real hardware reward evaluation and modern TRL API."""
        import inspect
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer as TRLGRPOTrainer

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Low-memory QLoRA / 4-bit config for practical local execution
        model_kwargs: dict[str, Any] = {"device_map": "auto"}
        if self.config.get("load_in_4bit"):
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception:
                model_kwargs["torch_dtype"] = "auto"
        else:
            model_kwargs["torch_dtype"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type=TaskType.CAUSAL_LM,
        )

        dataset = load_dataset("json", data_files=dataset_path, split="train")

        grpo_args: dict[str, Any] = {
            "output_dir": self.output_dir,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "learning_rate": self.learning_rate,
            "logging_steps": 10,
            "kl_coef": self.kl_coeff,
            "num_generations": self.group_size,
        }
        sig_grpo = inspect.signature(GRPOConfig.__init__)
        if "temperature" in sig_grpo.parameters:
            grpo_args["temperature"] = self.temperature

        training_config = GRPOConfig(**grpo_args)

        # Genuine hardware evaluation reward function with task-aware routing
        def hardware_reward_fn(
            completions: list[Any],
            prompts: list[Any] | None = None,
            **kwargs: Any,
        ) -> list[float]:
            """Evaluate completions with real Verilator/Cocotb/Yosys EDA pipeline concurrently across curriculum tasks.
            
            Handles both plain string completions and TRL conversational message dicts.
            Routes extra dataset columns (e.g. benchmark_task_id) directly to the task evaluator.
            """
            return self.evaluator.evaluate_batch(completions, prompts=prompts, **kwargs)

        # Modern TRL compatibility: uses processing_class; fallback to tokenizer for older TRL versions
        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "args": training_config,
            "train_dataset": dataset,
            "reward_funcs": hardware_reward_fn,
            "peft_config": lora_config,
        }
        sig = inspect.signature(TRLGRPOTrainer.__init__)
        if "processing_class" in sig.parameters:
            trainer_kwargs["processing_class"] = tokenizer
        else:
            trainer_kwargs["tokenizer"] = tokenizer

        trainer = TRLGRPOTrainer(**trainer_kwargs)

        result = trainer.train()
        trainer.save_model(self.output_dir)

        logger.info(f"GRPO hardware training complete. Model saved to {self.output_dir}")
        return {
            "steps": result.global_step,
            "output_dir": self.output_dir,
            "reward_type": "grounded_hardware_eda",
        }


def make_curriculum_environment_factory(
    work_dir: Optional[str] = None,
    default_benchmark_task_id: Optional[str] = None,
):
    """Create an environment factory compatible with TRL's environment-oriented GRPO rollout pattern.
    
    Returns a callable that instantiates HardwareDesignEnv dynamically for each dataset row,
    allowing multi-environment training across the curriculum benchmarks.
    """
    from learning.environment import HardwareDesignEnv

    def environment_factory(benchmark_task_id: Optional[str] = None, **kwargs: Any) -> HardwareDesignEnv:
        target_id = benchmark_task_id or default_benchmark_task_id
        return HardwareDesignEnv(
            work_dir=work_dir,
            benchmark_task_id=target_id,
            **kwargs,
        )

    return environment_factory
