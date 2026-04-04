"""
experiment.py
-------------
VerificationCollapseExperiment: the main 20-iteration loop.

Usage
-----
    from src.experiment import VerificationCollapseExperiment
    exp = VerificationCollapseExperiment.from_config("config.yaml")
    exp.run()
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import wandb
import yaml

if TYPE_CHECKING:
    import wandb.sdk.wandb_run

from .gsm8k_loader import GSM8KDataset
from .verifier import ModelVerifier
from .utils import (
    find_hard_negatives,
    mix_batches,
    summarise_iteration,
)


class VerificationCollapseExperiment:
    """
    Orchestrates the iterative self-training loop.

    Parameters
    ----------
    config : dict
        Full config dict (loaded from config.yaml).
    verifier : ModelVerifier
        Pre-initialised model wrapper.
    train_data : list[dict]
        Training split records.
    val_data : list[dict]
        Validation split records.
    """

    def __init__(
        self,
        config: dict,
        verifier: ModelVerifier,
        train_data: list[dict],
        val_data: list[dict],
    ):
        self.config = config
        self.verifier = verifier
        self.train_data = train_data
        self.val_data = val_data
        self.hard_negative_bank: list[dict] = []
        self.rng = random.Random(config["data"]["seed"])

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = "config.yaml",
        wandb_run: Optional["wandb.sdk.wandb_run.Run"] = None,
    ) -> "VerificationCollapseExperiment":
        """Load config, dataset, and model; return a ready experiment."""
        config_path = Path(config_path)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        dcfg = config["data"]
        cache = Path("data/gsm8k_subset.json")

        print("Loading GSM8K …")
        full_ds = GSM8KDataset(
            subset_size=dcfg["subset_size"],
            split="train",
            seed=dcfg["seed"],
            dataset_name=dcfg["dataset_name"],
            dataset_config=dcfg["dataset_config"],
            cache_path=cache,
        )
        train_ds, val_ds = full_ds.train_val_split(dcfg["train_ratio"])
        print(f"  Train: {len(train_ds.data)}  Val: {len(val_ds.data)}")

        print("Loading model …")
        verifier = ModelVerifier(config)

        return cls(
            config=config,
            verifier=verifier,
            train_data=train_ds.data,
            val_data=val_ds.data,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, num_iterations: Optional[int] = None) -> None:
        ecfg = self.config["experiment"]
        wcfg = self.config["wandb"]
        num_iterations = num_iterations or ecfg["num_iterations"]

        # Prepare CSV log
        log_dir = Path(self.config["experiment"].get("log_dir", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_path = log_dir / "metrics.csv"
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer: Optional[csv.DictWriter] = None

        # Init W&B
        run = wandb.init(
            project=wcfg["project"],
            entity=wcfg.get("entity") or None,
            tags=wcfg.get("tags", []),
            config=self.config,
        )

        try:
            for iteration in range(1, num_iterations + 1):
                print(f"\n{'='*60}")
                print(f"  Iteration {iteration}/{num_iterations}")
                print(f"{'='*60}")
                metrics = self._run_iteration(iteration)
                wandb.log(metrics, step=iteration)
                self._print_metrics(metrics)

                # Write CSV row (initialise writer on first iteration)
                if csv_writer is None:
                    csv_writer = csv.DictWriter(csv_file, fieldnames=list(metrics.keys()))
                    csv_writer.writeheader()
                csv_writer.writerow(metrics)
                csv_file.flush()

                # Checkpoint every N iterations
                if iteration % ecfg["checkpoint_interval"] == 0:
                    ckpt_path = Path(ecfg["output_dir"]) / f"iter_{iteration:03d}"
                    self.verifier.save_checkpoint(ckpt_path)
        finally:
            csv_file.close()
            run.finish()
            print(f"Metrics saved → {csv_path}")

    def _run_iteration(self, iteration: int) -> dict:
        ecfg = self.config["experiment"]
        inject_interval = ecfg["hard_negative_injection_interval"]
        hard_neg_ratio = ecfg["hard_negative_ratio"]
        confidence_thr = ecfg.get("confidence_threshold", 0.7)

        # ----------------------------------------------------------------
        # a) Decide which batch to train on this iteration
        # ----------------------------------------------------------------
        if iteration % inject_interval == 0 and self.hard_negative_bank:
            print(f"  Injecting {int(hard_neg_ratio*100)}% hard negatives …")
            # Mark hard negatives so we can apply gold-solution training to them
            # (recalibration signal), while regular samples train on the model's
            # own completions (the biased signal that drives verification collapse).
            marked_hard_negs = [{**s, "_hard_neg": True} for s in self.hard_negative_bank]
            batch = mix_batches(
                self.train_data,
                marked_hard_negs,
                hard_neg_ratio=hard_neg_ratio,
                rng_seed=self.rng.randint(0, 2**31),
            )
        else:
            batch = list(self.train_data)
            self.rng.shuffle(batch)

        # ----------------------------------------------------------------
        # b–c) Generate predictions + compute scores
        # ----------------------------------------------------------------
        print("  Generating predictions …")
        prompts = [s["prompt"] for s in batch]
        references = [s["answer"] for s in batch]

        max_new_tokens = self.config["data"].get("max_new_tokens", 256)
        completions = self.verifier.generate(prompts, max_new_tokens=max_new_tokens)
        self_scores = self.verifier.score(prompts, completions)

        # ----------------------------------------------------------------
        # d) Collect hard negatives for the bank
        # ----------------------------------------------------------------
        new_hard_negs = find_hard_negatives(
            batch, completions, self_scores, confidence_threshold=confidence_thr
        )
        self.hard_negative_bank.extend(new_hard_negs)
        # Cap bank size to avoid unbounded growth
        if len(self.hard_negative_bank) > 500:
            self.hard_negative_bank = self.hard_negative_bank[-500:]

        # ----------------------------------------------------------------
        # e–f) Fine-tune only on examples the model thinks it solved correctly.
        #
        # KEY DESIGN: training target depends on sample origin:
        #   - Regular samples  → model's OWN completion as target.
        #     The model reinforces whatever it generated (right or wrong),
        #     creating the overconfidence feedback loop = verification collapse.
        #   - Hard negatives   → gold GSM8K solution as target.
        #     Forces the model to see the correct reasoning for problems it
        #     was confidently wrong about, bounding the gap (injection run).
        # ----------------------------------------------------------------
        self_correct = [
            {**s, "solution": s["solution"] if s.get("_hard_neg") else c}
            for s, c, sc in zip(batch, completions, self_scores)
            if sc > 0
        ]
        print(f"  Fine-tuning on {len(self_correct)}/{len(batch)} self-judged-correct examples …")
        loss = self.verifier.finetune(self_correct) if self_correct else None

        # ----------------------------------------------------------------
        # g) Val-set ground-truth evaluation (post-fine-tune)
        #    No self-score here — just ground truth on held-out data.
        # ----------------------------------------------------------------
        print("  Evaluating on val set …")
        val_prompts = [s["prompt"] for s in self.val_data]
        val_references = [s["answer"] for s in self.val_data]
        val_completions = self.verifier.generate(val_prompts, max_new_tokens=max_new_tokens)

        # ----------------------------------------------------------------
        # h) Build metrics dict
        # ----------------------------------------------------------------
        metrics = summarise_iteration(
            iteration=iteration,
            completions=completions,
            references=references,
            self_scores=self_scores,
            loss=loss,
            num_hard_negatives=len(new_hard_negs),
            val_completions=val_completions,
            val_references=val_references,
        )
        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _print_metrics(m: dict) -> None:
        print(
            f"  gap={m['gap']:.4f}  "
            f"self={m['self_score_mean']:.4f}  "
            f"gt_train={m['gt_score_train']:.4f}  "
            f"gt_val={m.get('gt_score_val', float('nan')):.4f}  "
            f"loss={m.get('loss', float('nan')):.4f}  "
            f"hard_negs={m['num_hard_negatives']}"
        )
