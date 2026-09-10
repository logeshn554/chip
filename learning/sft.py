"""
SFT Trainer — Supervised Fine-Tuning scaffold for Qwen3-4B.

Uses LoRA/QLoRA with the TRL library to fine-tune Qwen3-4B
on successful hardware design trajectories.

NOTE: This is a scaffold for Phase 8. Full implementation
requires accumulated trajectory data (1K+ episodes).
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class SFTTrainer:
    """Supervised Fine-Tuning for Qwen3-4B using LoRA.

    Fine-tunes the model on successful design trajectories
    so it learns from its best experiences.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.model_name = config.get("model_name", "Qwen/Qwen3-4B")
        self.learning_rate = config.get("learning_rate", 2e-5)
        self.num_epochs = config.get("num_epochs", 3)
        self.lora_r = config.get("lora_r", 16)
        self.lora_alpha = config.get("lora_alpha", 32)
        self.batch_size = config.get("batch_size", 4)
        self.output_dir = config.get("output_dir", "./models/sft")

    async def train(self, dataset_path: str) -> dict[str, Any]:
        """Run SFT training on a prepared dataset.

        Args:
            dataset_path: Path to the JSONL training dataset

        Returns:
            Training results dict with loss and metrics
        """
        logger.info("=" * 60)
        logger.info("SFT Training — Phase 8 scaffold")
        logger.info("=" * 60)

        if not os.path.exists(dataset_path):
            logger.error(f"Dataset not found: {dataset_path}")
            return {"error": "Dataset not found"}

        # Count examples
        with open(dataset_path, "r") as f:
            n_examples = sum(1 for _ in f)

        logger.info(f"Dataset: {dataset_path} ({n_examples} examples)")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"LoRA: r={self.lora_r}, alpha={self.lora_alpha}")
        logger.info(f"Training: lr={self.learning_rate}, epochs={self.num_epochs}")

        try:
            return self._run_training(dataset_path)
        except ImportError as e:
            logger.warning(
                f"Training dependencies not installed: {e}\n"
                f"Install with: pip install self-evolving-chip[learning]"
            )
            return {"error": f"Missing dependency: {e}", "scaffold": True}

    def _run_training(self, dataset_path: str) -> dict[str, Any]:
        """Execute the actual training loop.

        Requires: trl, peft, datasets, bitsandbytes
        """
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTTrainer as TRLSFTTrainer, SFTConfig

        has_cuda = torch.cuda.is_available()
        has_hip = getattr(torch.version, "hip", None) is not None
        use_fp16 = bool(has_cuda or has_hip)
        device_map = "auto" if (has_cuda or has_hip) else None

        # Load model with device awareness
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model_kwargs = {}
        if device_map:
            model_kwargs["device_map"] = device_map
            model_kwargs["torch_dtype"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        # Configure LoRA
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type=TaskType.CAUSAL_LM,
            lora_dropout=0.05,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Load dataset
        dataset = load_dataset("json", data_files=dataset_path, split="train")

        # Training config with device-aware precision
        training_config = SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            logging_steps=10,
            save_steps=100,
            save_total_limit=3,
            fp16=use_fp16,
            gradient_accumulation_steps=4,
        )

        # Create trainer
        trainer = TRLSFTTrainer(
            model=model,
            args=training_config,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )

        # Train
        result = trainer.train()
        trainer.save_model(self.output_dir)

        logger.info(f"SFT training complete. Model saved to {self.output_dir}")
        return {
            "loss": result.training_loss,
            "steps": result.global_step,
            "output_dir": self.output_dir,
        }
