"""
SFT Trainer — Supervised Fine-Tuning scaffold for Qwen-14B.

Uses LoRA/QLoRA with the TRL library to fine-tune Qwen-14B
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
    """Supervised Fine-Tuning for Qwen-14B using LoRA.

    Fine-tunes the model on successful design trajectories
    so it learns from its best experiences.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.model_name = config.get("model_name", "Qwen/Qwen2.5-14B-Instruct")
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
        import importlib

        try:
            torch = importlib.import_module("torch")
            datasets = importlib.import_module("datasets")
            peft = importlib.import_module("peft")
            transformers = importlib.import_module("transformers")
            trl = importlib.import_module("trl")
        except ImportError as err:
            raise ImportError(f"Missing training dependencies: {err}") from err

        load_dataset = datasets.load_dataset
        LoraConfig = peft.LoraConfig
        TaskType = peft.TaskType
        get_peft_model = peft.get_peft_model
        AutoModelForCausalLM = transformers.AutoModelForCausalLM
        AutoTokenizer = transformers.AutoTokenizer
        TRLSFTTrainer = getattr(trl, "SFTTrainer")
        SFTTrainer = TRLSFTTrainer
        SFTConfig = getattr(trl, "SFTConfig")

        has_cuda = torch.cuda.is_available()
        has_hip = getattr(torch.version, "hip", None) is not None
        has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
        has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        has_gpu_accel = bool(has_cuda or has_hip or has_xpu or has_mps)
        use_fp16 = has_gpu_accel
        device_map = "auto" if has_gpu_accel else None

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
