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
    """Compute Group Relative Policy Optimization (GRPO) advantages.
    
    A_i = (R_i - mean(R)) / (std(R) + eps)
    
    Given a group of completions for a prompt, GRPO normalizes rewards relative to the group
    to provide policy gradient estimates without requiring a separate critic network.
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
        top_module: str = "mac",
        test_file: str = "tests/mac/test_mac.py",
        work_dir: str = "./sim_build/grpo_eval",
        target_cells: float = 500.0,
        target_timing_ns: float = 5.0,
        enable_synthesis: bool = True,
        enable_formal: bool = True,
    ):
        self.top_module = top_module
        self.test_file = test_file
        self.work_dir = work_dir
        self.target_cells = target_cells
        self.target_timing_ns = target_timing_ns
        self.enable_synthesis = enable_synthesis
        self.enable_formal = enable_formal

        os.makedirs(work_dir, exist_ok=True)

        self.verilator = VerilatorTool(work_dir=os.path.join(work_dir, "sim"))
        self.cocotb = CocotbTool(sim_dir=os.path.join(work_dir, "sim"))
        self.yosys = YosysTool(work_dir=os.path.join(work_dir, "synth"))
        self.formal = FormalVerificationTool(work_dir=os.path.join(work_dir, "formal"))
        self.reward_engine = RewardEngine()

    def extract_rtl(self, text: str) -> str:
        """Extract SystemVerilog code from model completion output."""
        # 1. Look for ```systemverilog ... ``` block
        sv_match = re.search(r"```(?:systemverilog|verilog)\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if sv_match:
            return sv_match.group(1).strip()

        # 2. Look for generic ``` ... ``` block containing module
        generic_match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
        if generic_match and "module " in generic_match.group(1):
            return generic_match.group(1).strip()

        # 3. Direct module declaration
        mod_match = re.search(r"(\bmodule\s+.*?\bendmodule\b)", text, re.DOTALL)
        if mod_match:
            return mod_match.group(1).strip()

        # 4. JSON action containing 'code' in params
        json_match = re.search(r'"code"\s*:\s*"([^"]+)"', text)
        if json_match:
            return json_match.group(1).replace("\\n", "\n").replace('\\"', '"')

        return ""

    async def evaluate_completion_async(self, completion: str, task_name: Optional[str] = None) -> dict[str, Any]:
        """Evaluate a single completion asynchronously using EDA tools."""
        code = self.extract_rtl(completion)
        if not code:
            return {
                "reward": 0.0,
                "compile_pass": False,
                "functional_pass": False,
                "error": "No synthesizable SystemVerilog module found in completion.",
            }

        cand_hash = hashlib.md5(completion.encode("utf-8")).hexdigest()[:8]
        cand_dir = os.path.join(self.work_dir, f"cand_{cand_hash}")
        os.makedirs(cand_dir, exist_ok=True)
        rtl_path = os.path.join(cand_dir, f"{self.top_module}.sv")

        with open(rtl_path, "w", encoding="utf-8") as f:
            f.write(code)

        # Stage 1: Verilator Lint & Syntax Compile Check (Hard Gate)
        compile_res = await self.verilator.lint_and_compile(rtl_path, top_module=self.top_module)
        compile_pass = compile_res.get("status") == "passed"
        if not compile_pass:
            return {
                "reward": 0.0,
                "compile_pass": False,
                "functional_pass": False,
                "error": compile_res.get("error", "Compilation failed"),
            }

        # Stage 2: Cocotb / pytest functional verification
        test_file = self.test_file if os.path.exists(self.test_file) else None
        func_res = await self.cocotb.run_tests(rtl_path, test_file or "tests/mac/test_mac.py")
        total_tests = func_res.get("tests_total", 0)
        passed_tests = func_res.get("tests_passed", 0)
        pass_rate = (passed_tests / total_tests) if total_tests > 0 else (1.0 if func_res.get("status") == "passed" else 0.0)

        # Stage 3: Logic Synthesis (Yosys)
        area: Optional[float] = None
        if self.enable_synthesis:
            synth_res = await self.yosys.synthesize(rtl_path, top_module=self.top_module)
            if synth_res.get("status") == "passed":
                area = float(synth_res.get("cells", 0) or synth_res.get("estimated_area", 0) or 0)

        # Stage 4: Formal Verification (SymbiYosys)
        formal_status = "SKIPPED"
        if self.enable_formal:
            formal_res = await self.formal.verify(rtl_path, top_module=self.top_module)
            formal_status = formal_res.get("status", "SKIPPED")

        # Grounded Reward Calculation with Weight Re-normalization
        reward_result = self.reward_engine.compute_modular_reward(
            compile_success=compile_pass,
            test_pass_rate=pass_rate,
            formal_status=formal_status,
            area=area,
            area_target=self.target_cells,
        )

        return {
            "reward": reward_result.normalized_reward,
            "raw_reward": reward_result.total_reward,
            "compile_pass": compile_pass,
            "pass_rate": pass_rate,
            "formal_status": formal_status,
            "area": area,
            "breakdown": reward_result.breakdown,
        }

    def evaluate_completion(self, completion: str) -> float:
        """Synchronous wrapper returning normalized reward in [0.0, 1.0]."""
        res = asyncio.run(self.evaluate_completion_async(completion))
        return float(res["reward"])

    def evaluate_batch(self, completions: list[str], **kwargs) -> list[float]:
        """Compute grounded hardware rewards for a batch/group of candidate completions."""
        rewards = []
        for c in completions:
            r = self.evaluate_completion(c)
            rewards.append(r)
        return rewards


class GRPOTrainer:
    """Group Relative Policy Optimization for Qwen3-4B with Hardware Evaluator.

    GRPO evaluates groups of G candidate responses per prompt against the real EDA
    environment, computes group-relative advantages, and updates policy weights.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.model_name = config.get("model_name", "Qwen/Qwen3-4B")
        self.group_size = config.get("group_size", 4)
        self.learning_rate = config.get("learning_rate", 1e-5)
        self.kl_coeff = config.get("kl_coeff", 0.1)
        self.output_dir = config.get("output_dir", "./models/grpo")

        # Hardware evaluator
        self.evaluator = HardwareRewardEvaluator(
            top_module=config.get("top_module", "mac"),
            test_file=config.get("test_file", "tests/mac/test_mac.py"),
            target_cells=config.get("target_cells", 500.0),
        )

    async def train(self, dataset_path: str) -> dict[str, Any]:
        """Run GRPO training on a hardware preference dataset.

        Args:
            dataset_path: Path to the preference dataset (JSONL with
                          prompt, chosen, rejected fields)

        Returns:
            Training results dict
        """
        logger.info("=" * 60)
        logger.info("GRPO Training — Hardware-in-the-Loop Environment")
        logger.info("=" * 60)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Group size: {self.group_size}")
        logger.info(f"KL coefficient: {self.kl_coeff}")

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
        """Execute GRPO training with real hardware reward evaluation."""
        from datasets import load_dataset
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer as TRLGRPOTrainer

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type=TaskType.CAUSAL_LM,
        )

        dataset = load_dataset("json", data_files=dataset_path, split="train")

        training_config = GRPOConfig(
            output_dir=self.output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            learning_rate=self.learning_rate,
            logging_steps=10,
            kl_coef=self.kl_coeff,
            num_generations=self.group_size,
        )

        # Genuine hardware evaluation reward function
        def hardware_reward_fn(completions: list[str], **kwargs) -> list[float]:
            """Evaluate completions with real Verilator/Cocotb/Yosys EDA pipeline."""
            return self.evaluator.evaluate_batch(completions, **kwargs)

        trainer = TRLGRPOTrainer(
            model=model,
            args=training_config,
            train_dataset=dataset,
            tokenizer=tokenizer,
            reward_funcs=hardware_reward_fn,
            peft_config=lora_config,
        )

        result = trainer.train()
        trainer.save_model(self.output_dir)

        logger.info(f"GRPO hardware training complete. Model saved to {self.output_dir}")
        return {
            "steps": result.global_step,
            "output_dir": self.output_dir,
            "reward_type": "grounded_hardware_eda",
        }
