"""
verifier.py
-----------
ModelVerifier: load Qwen3.x in 4-bit NF4, score predictions, QLoRA fine-tune.

Public API
----------
ModelVerifier(config)
    .generate(prompts)              -> list[str]    raw completions
    .score(prompts, completions)    -> list[float]  per-sample self-scores (0/1):
                                                    the model judges its own answer
                                                    without access to the reference
    .finetune(batch)                -> float        mean training loss for the batch
    .save_checkpoint(path)
    .load_checkpoint(path)
"""

from __future__ import annotations

from pathlib import Path

import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW

_SELF_SCORE_TEMPLATE = (
    "Below is a math problem and your attempted solution.\n\n"
    "Problem: {problem}\n\n"
    "Your solution: {completion}\n\n"
    "Is your final answer correct? Reply with only 'yes' or 'no'."
)


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
        self.gen_batch_size = config.get("training", {}).get("gen_batch_size", 4)

        # Enable TF32 for faster matmul on Ampere/Ada (RTX 30xx/40xx)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self._load_model()
        self._attach_lora()
        self._build_optimizer()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        mcfg = self.config["model"]
        qcfg = mcfg["quantization"]

        bnb_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
                bnb_4bit_compute_dtype=getattr(torch, qcfg["bnb_4bit_compute_dtype"]),
                bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
            )
            if qcfg["load_in_4bit"]
            else None
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            mcfg["name"],
            trust_remote_code=True,
            padding_side="left",   # important for batch generation
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            mcfg["name"],
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.model.config.use_cache = False  # required for gradient checkpointing

    def _attach_lora(self) -> None:
        lcfg = self.config["lora"]
        tcfg = self.config["training"]

        if tcfg["gradient_checkpointing"]:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        lora_config = LoraConfig(
            r=lcfg["r"],
            lora_alpha=lcfg["lora_alpha"],
            target_modules=lcfg["target_modules"],
            lora_dropout=lcfg["lora_dropout"],
            bias=lcfg["bias"],
            use_dora=lcfg.get("use_dora", False),
            task_type=TaskType[lcfg["task_type"]],
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def _build_optimizer(self) -> None:
        tcfg = self.config["training"]
        opt_name = tcfg.get("optimizer", "adamw_8bit")
        lr = tcfg["learning_rate"]

        if opt_name == "adamw":
            self.optimizer = AdamW(self.model.parameters(), lr=lr)
        else:
            try:
                import bitsandbytes.optim as bnb_optim
                if opt_name == "paged_adamw_32bit":
                    self.optimizer = bnb_optim.PagedAdamW32bit(self.model.parameters(), lr=lr)
                elif opt_name == "adamw_8bit":
                    self.optimizer = bnb_optim.AdamW8bit(self.model.parameters(), lr=lr)
                else:
                    raise ValueError(f"Unknown optimizer: {opt_name}")
            except (ImportError, ValueError) as e:
                print(f"bitsandbytes optimizer '{opt_name}' unavailable ({e}); falling back to AdamW.")
                self.optimizer = AdamW(self.model.parameters(), lr=lr)

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
        batch_size: int | None = None,
    ) -> list[str]:
        """Generate completions for a list of prompts (batched)."""
        batch_size = batch_size if batch_size is not None else self.gen_batch_size
        self.model.eval()
        completions: list[str] = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            # Apply chat template so the model knows the turn boundary and
            # generates an EOS at the right place instead of continuing.
            formatted = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in batch_prompts
            ]
            enc = self.tokenizer(
                formatted,
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
                use_cache=True,
            )
            # Decode only newly generated tokens
            for _, ids in enumerate(out):
                input_len = enc["input_ids"].shape[1]
                new_ids = ids[input_len:]
                completions.append(
                    self.tokenizer.decode(new_ids, skip_special_tokens=True)
                )

        return completions

    def score(
        self,
        prompts: list[str],
        completions: list[str],
        batch_size: int | None = None,
    ) -> list[float]:
        """
        Return a per-sample self-correctness score in [0, 1].

        The model is prompted to judge its own answer (yes/no) without access
        to the ground truth reference. This is the true self-score; compare it
        against compute_external_score() (which uses the reference) to measure
        the verification gap.
        """
        batch_size = batch_size if batch_size is not None else self.gen_batch_size
        judge_prompts = [
            _SELF_SCORE_TEMPLATE.format(problem=p, completion=c)
            for p, c in zip(prompts, completions)
        ]
        # Short generation: we only need "yes" or "no"
        verdicts = self.generate(judge_prompts, max_new_tokens=5, batch_size=batch_size)
        return [float("yes" in v.lower()) for v in verdicts]

    # ------------------------------------------------------------------
    # Fine-tuning
    # ------------------------------------------------------------------

    def finetune(
        self,
        samples: list[dict],
    ) -> float:
        """
        Fine-tune on a mixed batch for one epoch.

        Parameters
        ----------
        samples : list[dict]
            Each dict must have keys "prompt" and "solution"
            (the full chain-of-thought target text).

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
        warmup_ratio = tcfg.get("warmup_ratio", 0.05)
        num_warmup_steps = int(total_steps * warmup_ratio)
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
