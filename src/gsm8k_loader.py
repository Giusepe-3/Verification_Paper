"""
gsm8k_loader.py
---------------
Data loading, answer extraction, and verification for GSM8K.

Public API
----------
GSM8KDataset   – torch Dataset wrapping a HuggingFace GSM8K split
answer_extractor(text) -> str | None  – pull final numeric answer from model output
verify(pred, reference) -> bool       – string-level correctness check
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def answer_extractor(text: str) -> Optional[str]:
    """
    Extract the final numeric answer from a model response.

    Strategy (in priority order):
    1. Last \\boxed{...} expression.
    2. Last bare number in the string (strips commas, trims whitespace).
    Returns None if nothing is found.
    """
    # 1. Try \boxed{...}
    boxed_matches = _BOXED_RE.findall(text)
    if boxed_matches:
        raw = boxed_matches[-1].strip().replace(",", "")
        return raw

    # 2. Fallback: last number in the text
    num_matches = _NUMBER_RE.findall(text)
    if num_matches:
        return num_matches[-1].replace(",", "")

    return None


def verify(prediction: Optional[str], reference: str) -> bool:
    """
    Return True iff the extracted prediction matches the reference answer.

    Comparison is numeric when both sides parse as floats, otherwise
    it falls back to normalised string equality.
    """
    if prediction is None:
        return False

    pred_clean = prediction.strip().replace(",", "")
    ref_clean = reference.strip().replace(",", "")

    # Numeric comparison (handles "6" == "6.0", etc.)
    try:
        return float(pred_clean) == float(ref_clean)
    except ValueError:
        return pred_clean.lower() == ref_clean.lower()


# ---------------------------------------------------------------------------
# Reference-answer extraction from GSM8K gold solutions
# ---------------------------------------------------------------------------

_GSM8K_ANS_RE = re.compile(r"####\s*(-?\d[\d,]*\.?\d*)")


def _extract_gsm8k_answer(solution: str) -> str:
    """Pull the number after '####' in a GSM8K solution string."""
    m = _GSM8K_ANS_RE.search(solution)
    if m:
        return m.group(1).replace(",", "")
    # Fallback: last number
    nums = _NUMBER_RE.findall(solution)
    return nums[-1].replace(",", "") if nums else ""


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Show your reasoning, then write your final answer inside \\boxed{{}}.\n\n"
    "Problem: {question}\n\nSolution:"
)


class GSM8KDataset(Dataset):
    """
    Loads a subset of GSM8K and wraps it as a torch Dataset.

    Parameters
    ----------
    subset_size : int
        Number of problems to keep (randomly sampled).
    split : str
        HuggingFace split name, e.g. "train" or "test".
    seed : int
        Random seed for subsetting.
    cache_path : str | Path | None
        If provided, save/load the subset as JSON here to avoid re-downloading.
    """

    def __init__(
        self,
        subset_size: int = 500,
        split: str = "train",
        seed: int = 42,
        dataset_name: str = "openai/gsm8k",
        dataset_config: str = "main",
        cache_path: str | Path | None = None,
    ):
        self.subset_size = subset_size
        self.split = split
        self.seed = seed

        if cache_path and Path(cache_path).exists():
            self.data = self._load_cache(Path(cache_path))
        else:
            self.data = self._download_and_subset(dataset_name, dataset_config)
            if cache_path:
                self._save_cache(Path(cache_path))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download_and_subset(self, name: str, config: str) -> list[dict]:
        hf_ds = load_dataset(name, config, split=self.split)
        # Shuffle and take subset
        hf_ds = hf_ds.shuffle(seed=self.seed).select(
            range(min(self.subset_size, len(hf_ds)))
        )
        records = []
        for row in hf_ds:
            records.append(
                {
                    "question": row["question"],
                    "solution": row["answer"],                        # full chain-of-thought
                    "answer": _extract_gsm8k_answer(row["answer"]),  # numeric only
                    "prompt": PROMPT_TEMPLATE.format(question=row["question"]),
                }
            )
        return records

    def _save_cache(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def _load_cache(self, path: Path) -> list[dict]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # torch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def train_val_split(self, train_ratio: float = 0.8) -> tuple["GSM8KDataset", "GSM8KDataset"]:
        """Return two GSM8KDataset views (train / val) without re-downloading."""
        n_train = int(len(self.data) * train_ratio)
        train_ds = _SlicedDataset(self.data[:n_train])
        val_ds = _SlicedDataset(self.data[n_train:])
        return train_ds, val_ds  # type: ignore[return-value]


class _SlicedDataset(GSM8KDataset):
    """Lightweight view into a pre-built list — skips __init__ download."""

    def __init__(self, data: list[dict]):  # type: ignore[override]
        self.data = data
