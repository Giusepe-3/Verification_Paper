"""
verifier.py
-----------
ModelVerifier: load Qwen3.x in 4-bit NF4, score predictions, QLoRA fine-tune.

Public API
----------
ModelVerifier(config)
    .generate(prompts)    -> list[str]          raw completions
    .score(prompts, refs) -> list[float]        per-sample self-scores (0/1 for now,
                                                extendable to log-prob confidence)
    .finetune(batch)      -> float              mean training loss for the batch
    .save_checkpoint(path)
    .load_checkpoint(path)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW

from .gsm8k_loader import answer_extractor, verify


class ModelVerifier:
    """
    Wraps a 4-bit quantised Qwen model with a QLoRA adapter for iterative
    fine-tuning in the verification-collapse experiment.

    Parameters
    ----------
    config : dict
        Merged config dict (from config.yaml). Expected keys:
        config["model"], config["lora"], config["training"].
    device : str | None
        Force a specific device. Defaults to "cuda" if available, else "cpu".
    """

    def __init__(self, config: dict, device: str | None = None):
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._load_model()
        self._attach_lora()
        self._build_optimizer()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        mcfg = self.config["model"]
        qcfg = mcfg["quantization"]

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=qcfg["load_in_4bit"],
            bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=getattr(torch, qcfg["bnb_4bit_compute_dtype"]),
            bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            mcfg["name"],
            trust_remote_code=True,
            padding_side="left",   # important for batch generation
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            mcfg["name"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.model.config.use_cache = False  # required for gradient checkpointing

    def _attach_lora(self) -> None:
        lcfg = self.config["lora"]
        tcfg = self.config["training"]

        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=tcfg["gradient_checkpointing"],
        )

        lora_config = LoraConfig(
            r=lcfg["r"],
            lora_alpha=lcfg["lora_alpha"],
            target_modules=lcfg["target_modules"],
            lora_dropout=lcfg["lora_dropout"],
            bias=lcfg["bias"],
            use_dora=lcfg.get("use_dora", False),
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def _build_optimizer(self) -> None:
        tcfg = self.config["training"]
        # PagedAdamW is exposed through bitsandbytes
        try:
            from bitsandbytes.optim import PagedAdamW32bit
            self.optimizer = PagedAdamW32bit(
                self.model.parameters(),
                lr=tcfg["learning_rate"],
            )
        except ImportError:
            print("bitsandbytes PagedAdamW not found; falling back to AdamW.")
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=tcfg["learning_rate"],
            )
        # Scheduler is (re-)built per fine-tune call because num_steps varies
        self.scheduler = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 256,
        batch_size: int = 4,
    ) -> list[str]:
        """Generate completions for a list of prompts (batched)."""
        self.model.eval()
        completions: list[str] = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            enc = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config["data"]["max_seq_length"],
            ).to(self.device)

            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,             # greedy — deterministic scoring
                pad_token_id=self.tokenizer.pad_token_id,
            )
            # Decode only newly generated tokens
            for j, ids in enumerate(out):
                input_len = enc["input_ids"].shape[1]
                new_ids = ids[input_len:]
                completions.append(
                    self.tokenizer.decode(new_ids, skip_special_tokens=True)
                )

        return completions

    def score(
        self,
        prompts: list[str],
        references: list[str],
        max_new_tokens: int = 256,
        batch_size: int = 4,
    ) -> list[float]:
        """
        Return a per-sample correctness score in [0, 1].

        Currently binary (0 or 1) based on answer extraction + string match.
        Can be upgraded to token-log-prob confidence without changing the API.
        """
        completions = self.generate(prompts, max_new_tokens=max_new_tokens, batch_size=batch_size)
        scores = []
        for completion, ref in zip(completions, references):
            extracted = answer_extractor(completion)
            scores.append(float(verify(extracted, ref)))
        return scores

    # ------------------------------------------------------------------
    # Fine-tuning
    # ------------------------------------------------------------------

    def finetune(
        self,
        samples: list[dict],
        num_warmup_steps: int = 0,
    ) -> float:
        """
        Fine-tune on a mixed batch for one epoch.

        Parameters
        ----------
        samples : list[dict]
            Each dict must have keys "prompt" and "solution"
            (the full chain-of-thought target text).
        num_warmup_steps : int
            Warmup steps for the cosine scheduler for this micro-epoch.

        Returns
        -------
        float
            Mean cross-entropy loss over the batch.
        """
        tcfg = self.config["training"]
        self.model.train()

        # Build scheduler for this call
        grad_steps = tcfg["gradient_accumulation_steps"]
        total_steps = max(1, len(samples) // (tcfg["per_device_train_batch_size"] * grad_steps))
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
        )

        batch_size = tcfg["per_device_train_batch_size"]
        accumulated_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for step, i in enumerate(range(0, len(samples), batch_size)):
            mini = samples[i : i + batch_size]

            # Concatenate prompt + solution as target
            full_texts = [s["prompt"] + " " + s["solution"] for s in mini]
            enc = self.tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config["data"]["max_seq_length"],
            ).to(self.device)

            labels = enc["input_ids"].clone()
            # Mask padding tokens in loss
            labels[labels == self.tokenizer.pad_token_id] = -100

            outputs = self.model(**enc, labels=labels)
            loss = outputs.loss / grad_steps
            loss.backward()

            accumulated_loss += outputs.loss.item()
            num_batches += 1

            if (step + 1) % grad_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), tcfg["max_grad_norm"]
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

        # Final partial accumulation
        if num_batches % grad_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), tcfg["max_grad_norm"]
            )
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()

        return accumulated_loss / max(num_batches, 1)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        print(f"Checkpoint saved → {path}")

    def load_checkpoint(self, path: str | Path) -> None:
        from peft import PeftModel
        path = Path(path)
        self.model = PeftModel.from_pretrained(self.model, str(path))
        print(f"Checkpoint loaded ← {path}")
