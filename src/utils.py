"""
utils.py
--------
Metric helpers for the verification-collapse experiment.

Metrics
-------
compute_self_score(scores)          -> float   mean model self-correctness
compute_external_score(preds, refs) -> float   mean ground-truth correctness
compute_gap(self_score, ext_score)  -> float   |self - external|
summarise_iteration(...)            -> dict    all metrics in one call
"""

from __future__ import annotations

from typing import Optional

from .gsm8k_loader import answer_extractor, verify


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_self_score(scores: list[float]) -> float:
    """Mean of per-sample self-correctness scores (already in [0,1])."""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def compute_external_score(
    completions: list[str],
    references: list[str],
) -> float:
    """
    Ground-truth correctness: extract answers from completions, compare to refs.
    Returns proportion correct in [0, 1].
    """
    if not completions:
        return 0.0
    correct = sum(
        verify(answer_extractor(c), r)
        for c, r in zip(completions, references)
    )
    return correct / len(completions)


def compute_gap(self_score: float, external_score: float) -> float:
    """Signed verification gap: self_score - external_score.
    Positive means the model overestimates its own correctness.
    The paper's claim is this grows monotonically over iterations.
    """
    return self_score - external_score


# ---------------------------------------------------------------------------
# Hard-negative mining
# ---------------------------------------------------------------------------

def find_hard_negatives(
    samples: list[dict],
    completions: list[str],
    self_scores: list[float],
    confidence_threshold: float = 0.7,
) -> list[dict]:
    """
    Hard negatives = samples where the model was confident but wrong.

    Parameters
    ----------
    samples : list[dict]
        Original dataset records (must have "answer" key for reference).
    completions : list[str]
        Raw model outputs, aligned with samples.
    self_scores : list[float]
        Per-sample self-correctness scores (used as proxy for confidence).
    confidence_threshold : float
        Samples with self_score >= threshold are considered "high confidence".

    Returns
    -------
    list[dict]
        Subset of samples that were high-confidence wrong predictions.
    """
    hard_negs = []
    for sample, completion, score in zip(samples, completions, self_scores):
        extracted = answer_extractor(completion)
        is_wrong = not verify(extracted, sample["answer"])
        is_confident = score >= confidence_threshold
        if is_confident and is_wrong:
            hard_negs.append(sample)
    return hard_negs


def mix_batches(
    regular: list[dict],
    hard_negatives: list[dict],
    hard_neg_ratio: float = 0.5,
    rng_seed: int = 42,
) -> list[dict]:
    """
    Mix regular samples with hard negatives at the given ratio.

    Parameters
    ----------
    regular : list[dict]
        Standard training samples.
    hard_negatives : list[dict]
        Hard-negative samples to inject.
    hard_neg_ratio : float
        Fraction of the final batch that should be hard negatives.
    rng_seed : int
        For reproducible shuffling.

    Returns
    -------
    list[dict]
        Mixed and shuffled batch of the same total size as `regular`.
    """
    import random
    rng = random.Random(rng_seed)

    if not hard_negatives:
        return list(regular)

    n_total = len(regular)
    n_hard = min(int(n_total * hard_neg_ratio), len(hard_negatives))
    n_regular = n_total - n_hard

    selected_hard = rng.sample(hard_negatives, n_hard)
    selected_reg = rng.sample(regular, min(n_regular, len(regular)))

    mixed = selected_reg + selected_hard
    rng.shuffle(mixed)
    return mixed


# ---------------------------------------------------------------------------
# Iteration summary helper
# ---------------------------------------------------------------------------

def summarise_iteration(
    iteration: int,
    completions: list[str],
    references: list[str],
    self_scores: list[float],
    loss: Optional[float],
    num_hard_negatives: int,
    val_completions: Optional[list[str]] = None,
    val_references: Optional[list[str]] = None,
) -> dict:
    """Aggregate all per-iteration metrics into one loggable dict.

    self_score_mean and gt_score_train are measured on the training batch
    (same generation, pre-fine-tune) so the gap comparison is direct.
    gt_score_val is measured on the held-out val set after fine-tuning and
    is the cleanest signal for generalisation.
    """
    self_score_mean = compute_self_score(self_scores)
    gt_score_train = compute_external_score(completions, references)
    gap = compute_gap(self_score_mean, gt_score_train)

    metrics = {
        "iteration": iteration,
        "self_score_mean": self_score_mean,
        "gt_score_train": gt_score_train,
        "gap": gap,
        "num_hard_negatives": num_hard_negatives,
    }
    if val_completions is not None and val_references is not None:
        metrics["gt_score_val"] = compute_external_score(val_completions, val_references)
    if loss is not None:
        metrics["loss"] = loss

    return metrics
