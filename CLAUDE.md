# CLAUDE.md — Verification Collapse Research Repo

## What This Project Is

This is the codebase for a NeurIPS 2026 submission.

**Core claim:** In iterative self-improving loops, the gap between a model's self-assigned verification score and external ground truth accuracy grows monotonically with training iterations — a failure mode we call _verification collapse_.

**Paper in two figures:**

1. Two curves (self-score vs. ground truth) diverging over iterations
2. Same setup with adversarial injection — gap stays bounded

**Deadline:** Abstract May 5, paper May 7, 2026 (Europe/Copenhagen).

---

## Repo Structure

```
verification-collapse/
├── CLAUDE.md                  # This file
├── README.md
├── notes/
│   ├── lab_notebook.md        # Experimental observations — update after every run
│   └── rsi-thesis-seed.md     # Living research direction document
├── src/
│   ├── loop.py                # Main self-improvement loop
│   ├── sampler.py             # Task sampling (GSM8K or code benchmark)
│   ├── generator.py           # Model generation
│   ├── verifier.py            # Dual scorer: self-score + ground truth score
│   └── inject.py              # Adversarial injection logic
├── experiments/
│   ├── run_baseline.py        # Baseline: loop with no injection
│   ├── run_injection.py       # Intervention: periodic adversarial injection
│   └── configs/               # YAML configs per experiment
├── results/
│   ├── figures/               # Generated plots (not committed — regenerate from data)
│   └── logs/                  # Raw CSV or W&B exports
└── paper/
    └── draft.tex              # NeurIPS submission
```

---

## The Experiment

**Domain:** GSM8K math subset (ground truth = correct final number, one-line check)
**Model:** Small open-weights model (Qwen3.5-9B or similar) via HuggingFace
**Loop:**

1. Sample N tasks
2. Model generates solutions
3. Dual score every solution: self-assigned score + external ground truth score
4. Fine-tune on problems the model "thinks" it solved correctly
5. Repeat for 10–20 iterations
6. Log the gap at every checkpoint

**Key measurements at every iteration:**

- `self_score`: fraction of problems the model judges itself correct on
- `gt_score`: fraction of problems actually correct against held-out ground truth
- `gap`: `self_score - gt_score`
- `iteration`: loop step

**The adversarial injection variant:**
Every N iterations, inject K problems the model demonstrably fails on (drawn from an external source). Measure whether the gap closes or stays bounded.

---

## Instrumentation Rules

- Every experiment run logs to W&B with a unique run name
- Every run also writes a CSV to `results/logs/` as a backup
- Config (model, n_tasks, n_iterations, injection_freq, injection_k) is always logged alongside results
- After any run, update `notes/lab_notebook.md` with: what you ran, what you saw, what it means

---

## Code Conventions

- Python 3.10+
- PyTorch for any training steps
- HuggingFace `transformers` and `datasets` for model and data loading
- `wandb` for experiment tracking
- Keep each component (sampler, generator, verifier, injector) in its own file — they need to be swappable
- No notebooks in the main codebase — prototype in notebooks, commit clean scripts
- Every script takes a config path as argument — no hardcoded hyperparameters
- `functional_call` pattern from MAML work applies here: if you need to run the model with modified parameters, use it

---

## What Not To Do

- Do not replicate AZR or SRLM in full — you only need the measurement infrastructure
- Do not use GPT-4 or closed APIs for the self-verifier — the model must judge its own outputs
- Do not commit W&B run artifacts or model checkpoints — log run IDs instead
- Do not start writing before the 3-iteration sanity check confirms degradation exists

---

## Key Concepts (for Claude Code context)

**Verification collapse:** The phenomenon where a self-improving model's internal verifier co-adapts to its own outputs, causing self-assigned scores to progressively overestimate actual performance relative to ground truth.

**Self-score:** What the model thinks it scored. Obtained by prompting the model to judge its own generated answer.

**Ground truth score:** What the model actually scored. Obtained by comparing the model's final answer to the held-out correct answer.

**Gap:** `self_score - gt_score`. The core measurement. The paper's claim is that this grows monotonically over iterations.

**Adversarial injection:** Periodic insertion of tasks drawn from an external source that the model demonstrably fails on. The intervention hypothesis: this forces the self-verifier to recalibrate against reality, slowing or resetting gap growth.

**Iteration:** One full cycle of: sample tasks → generate solutions → dual score → fine-tune on self-judged correct examples.

---

## Related Work (for context when editing paper/)

- **SRLM (Yuan et al. 2024):** Qualitative observation of judge-generator collapse. We quantify it.
- **STaR (Zelikman et al. 2022):** Ground-truth-anchored baseline that doesn't have the problem — useful contrast.
- **AZR (NeurIPS 2025 spotlight):** Direct foil. Assumes self-verification stays calibrated. We test that assumption.
- **Gao et al. 2023 (RM overoptimization):** Closest prior work — RM scores diverge from ground truth under optimization pressure. We show the same emerges endogenously in self-supervised loops.
- **Mind the Gap (ICLR 2025):** Measures generation-verification gap as function of pretraining compute (static). We measure dynamic drift during self-improvement.

---

## Phase Gates

| Gate                      | Deadline    | Criteria                                   |
| ------------------------- | ----------- | ------------------------------------------ |
| Research question locked  | ✅ March 27 | One falsifiable sentence — done            |
| Sanity check              | March 28    | 3-iteration run shows degradation signal   |
| Core experiments complete | April 25    | Degradation curve + injection curve logged |
| Full draft                | May 3       | All sections written                       |
| Abstract registered       | May 5       | Submitted to NeurIPS portal                |
| Paper submitted           | May 7       | No exceptions                              |
