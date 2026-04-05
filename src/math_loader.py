"""
math_loader.py
--------------
Data loading, answer extraction, and verification for the MATH dataset
(Hendrycks et al. 2021 — hendrycks/competition_math on HuggingFace).

Public API
----------
MathDataset            – torch Dataset wrapping the MATH HuggingFace split
answer_extractor(text) -> str | None  – extract final answer from model output
verify(pred, ref)      -> bool        – correctness check
"""

from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _extract_boxed(text: str) -> Optional[str]:
    """
    Extract the content of the last \\boxed{...} in text.
    Handles nested braces (e.g. \\boxed{\\frac{1}{2}}).
    """
    results = []
    i = 0
    while i < len(text):
        pos = text.find(r"\boxed{", i)
        if pos == -1:
            break
        start = pos + 7  # len(r"\boxed{")
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            results.append(text[start : j - 1])
        i = pos + 1
    return results[-1].strip() if results else None


def answer_extractor(text: str) -> Optional[str]:
    """
    Extract the final answer from a model response.
    Priority: last \\boxed{...} → last bare number.
    """
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed
    nums = _NUMBER_RE.findall(text)
    if nums:
        return nums[-1].replace(",", "")
    return None


# ---------------------------------------------------------------------------
# Answer verification
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    """Normalise a math answer string for comparison."""
    s = s.strip().strip("$").replace(" ", "").replace(",", "")
    # \frac{a}{b} → a/b
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)
    # strip remaining LaTeX commands and braces
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("{", "").replace("}", "")
    return s.lower()


def verify(prediction: Optional[str], reference: str) -> bool:
    """
    Return True iff the extracted prediction matches the reference answer.
    Handles: integers, decimals, simple fractions, basic LaTeX expressions.
    """
    if prediction is None:
        return False

    pred = _normalize(prediction)
    ref = _normalize(reference)

    # Numeric (float) comparison
    try:
        return abs(float(pred) - float(ref)) < 1e-6
    except (ValueError, TypeError):
        pass

    # Fraction comparison (handles "3/4" == "3/4")
    try:
        return Fraction(pred) == Fraction(ref)
    except (ValueError, ZeroDivisionError):
        pass

    # Normalised string match
    return pred == ref


# ---------------------------------------------------------------------------
# Reference answer extraction from MATH gold solutions
# ---------------------------------------------------------------------------

def _extract_math_answer(solution: str) -> str:
    """Extract the answer from a MATH dataset gold solution (last \\boxed{})."""
    result = _extract_boxed(solution)
    if result:
        return result.strip()
    nums = _NUMBER_RE.findall(solution)
    return nums[-1].replace(",", "") if nums else ""


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Show your reasoning, then write your final answer inside \\boxed{{}}.\n\n"
    "Problem: {question}\n\nSolution:"
)


class MathDataset(Dataset):
    """
    Loads a subset of the MATH dataset (hendrycks/competition_math) and wraps it
    as a torch Dataset.

    Parameters
    ----------
    subset_size  : total problems to keep (randomly sampled after level filter)
    split        : "train" or "test"
    seed         : random seed for subsetting
    dataset_name : HuggingFace dataset name
    dataset_config: HuggingFace config name, or null for datasets with no named config
    level_filter : list of ints, e.g. [3,4,5] to keep only those difficulty
                   levels. None = keep all levels.
    cache_path   : optional JSON cache path to avoid re-downloading
    """

    def __init__(
        self,
        subset_size: int = 500,
        split: str = "train",
        seed: int = 42,
        dataset_name: str = "hendrycks/competition_math",
        dataset_config: str | None = None,
        level_filter: Optional[list[int]] = None,
        cache_path: str | Path | None = None,
    ):
        self.subset_size = subset_size
        self.split = split
        self.seed = seed
        self.level_filter = level_filter

        if cache_path and Path(cache_path).exists():
            self.data = self._load_cache(Path(cache_path))
        else:
            self.data = self._download_and_subset(dataset_name, dataset_config)
            if cache_path:
                self._save_cache(Path(cache_path))

    def _download_and_subset(self, name: str, config: str | None) -> list[dict]:
        hf_ds = (
            load_dataset(name, config, split=self.split, trust_remote_code=True)
            if config else
            load_dataset(name, split=self.split, trust_remote_code=True)
        )

        # Optional level filter — MATH levels are stored as "Level N" strings
        if self.level_filter:
            allowed = {f"Level {n}" for n in self.level_filter}
            hf_ds = hf_ds.filter(lambda x: x.get("level", "") in allowed)

        hf_ds = hf_ds.shuffle(seed=self.seed).select(
            range(min(self.subset_size, len(hf_ds)))
        )

        records = []
        for row in hf_ds:
            # Use pre-extracted 'answer' field; fall back to extracting from solution
            answer = (row.get("answer") or "").strip()
            if not answer:
                answer = _extract_math_answer(row["solution"])

            records.append(
                {
                    "question": row["problem"],
                    "solution": row["solution"],
                    "answer": answer,
                    "level": row.get("level", ""),
                    "subject": row.get("subject", row.get("type", "")),
                    "prompt": PROMPT_TEMPLATE.format(question=row["problem"]),
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

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]

    def train_val_split(
        self, train_ratio: float = 0.8
    ) -> tuple["MathDataset", "MathDataset"]:
        n_train = int(len(self.data) * train_ratio)
        train_ds = _SlicedDataset(self.data[:n_train])
        val_ds = _SlicedDataset(self.data[n_train:])
        return train_ds, val_ds  # type: ignore[return-value]


class _SlicedDataset(MathDataset):
    """Lightweight view into a pre-built list — skips __init__ download."""

    def __init__(self, data: list[dict]):  # type: ignore[override]
        self.data = data
