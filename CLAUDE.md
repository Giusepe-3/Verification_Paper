# CLAUDE.md — Verification Collapse Research Repo

## What This Project Is

This is the codebase for a NeurIPS 2026 submission.

**Core claim:** In iterative self-improving loops, the gap between a model's self-assigned verification score and external ground truth accuracy grows monotonically with training iterations — a failure mode we call _verification collapse_.

**Paper in two figures:**

1. Two curves (self-score vs. ground truth) diverging over iterations — baseline run
2. Same setup with adversarial injection — gap stays bounded — injection run

**Deadline:** Abstract May 5, paper May 7, 2026 (Europe/Copenhagen).

---

## Current Status (as of April 2, 2026)

**Sanity check: PASSED.** 3-iteration run (40 train / 10 val, Qwen3.5-9B, 4-bit) confirmed:
- Gap is positive: self-score overestimates ground truth (as predicted)
- W&B + CSV logging confirmed working
- End-to-end loop runs: generate → score → filter → fine-tune → val eval → log

**Next step:** Run baseline and injection configs (each ~10h, back to back).

```bash
# Night 1 — baseline (no injection)
nohup python run_experiment.py --config experiments/configs/baseline.yaml > logs/baseline_run.log 2>&1 &

# Night 2 — injection
nohup python run_experiment.py --config experiments/configs/injection.yaml > logs/injection_run.log 2>&1 &
```

Monitor with: `tail -f logs/baseline_run.log`

---

## Repo Structure (actual files)

```
verification-collapse/
├── CLAUDE.md
├── README.md
├── config.yaml                          # Full 20-iter / 500-sample config (reference)
├── run_experiment.py                    # Entry point: --config and --iterations flags
├── src/
│   ├── experiment.py                    # Main loop (VerificationCollapseExperiment)
│   ├── verifier.py                      # ModelVerifier: generate, score, finetune
│   ├── gsm8k_loader.py                  # Dataset loading, answer_extractor, verify
│   └── utils.py                         # compute_gap, summarise_iteration, hard-neg mining
├── experiments/
│   └── configs/
│       ├── sanity_check.yaml            # 50 samples, 3 iters — quick smoke test
│       ├── baseline.yaml                # 250 samples, 10 iters, injection DISABLED
│       └── injection.yaml              # 250 samples, 10 iters, injection every 3 iters
├── data/
│   └── gsm8k_subset.json               # Cached after first download (gitignored)
├── logs/                                # CSV outputs per run (gitignored)
├── models/                              # Checkpoints (gitignored)
├── notebooks/
│   └── sanity_check.ipynb
├── notes/
│   └── lab_notebook.md                  # Update after every run
└── paper/
    └── draft.tex
```

---

## The Experiment

**Domain:** GSM8K math subset (ground truth = correct final number, one-line check)
**Model:** Qwen3.5-9B via HuggingFace, 4-bit NF4 quantization + QLoRA (r=16)
**Loop:**

1. Sample N tasks from GSM8K train split
2. Model generates solutions (chat template, thinking mode OFF)
3. Dual score: self-score (model judges its own answer) + GT score (extract number, compare)
4. Fine-tune on problems the model **thinks** it solved correctly (self_score > 0)
5. Evaluate GT score on held-out val set
6. Repeat for 10–20 iterations

**Key measurements per iteration:**

| Field | Description |
|---|---|
| `self_score_mean` | Fraction the model judges itself correct (train batch) |
| `gt_score_train` | Fraction actually correct on same train batch |
| `gt_score_val` | Fraction actually correct on held-out val set (post fine-tune) |
| `gap` | `self_score_mean - gt_score_train` — signed, should grow monotonically |
| `loss` | Fine-tune cross-entropy |
| `num_hard_negatives` | Hard negatives found this iteration |

**The adversarial injection variant:**
Every 3 iterations, inject 50% hard negatives (high-confidence-wrong examples from previous iterations) into the fine-tuning batch. Hypothesis: forces self-verifier to recalibrate against reality.

---

## Known Model Quirks

**Qwen3 thinking mode:** Qwen3 generates `<think>...</think>` blocks by default. With
`max_new_tokens=256`, the model exhausts its budget mid-thought and never outputs an answer,
causing gt_score ≈ 0 and self_score = 0. Fixed by passing `enable_thinking=False` to
`apply_chat_template` in `verifier.py`.

**DoRA with 4-bit NF4 is extremely slow:** DoRA decomposes weights at every forward pass.
With NF4 quantization, this caused generation to take 20+ minutes per sample. Fixed by
setting `use_dora: false` in all configs.

**`BitsAndBytesConfig` guard:** When `load_in_4bit: false` (e.g. for tiny models),
`verifier.py` now passes `quantization_config=None` instead of constructing a BnB config.

---

## Instrumentation

- Every run logs to W&B (`rsi-verification-collapse` project) and writes CSV to `logs/<run_name>/metrics.csv`
- Config is always logged alongside results
- After any run, update `notes/lab_notebook.md`

---

## Code Conventions

- Python 3.10+, PyTorch, HuggingFace `transformers` + `datasets`, `wandb`, `peft`
- Components are separated: loader / verifier / utils / experiment — keep them swappable
- No notebooks in main codebase — prototype in `notebooks/`, commit clean scripts
- Every script takes `--config` path — no hardcoded hyperparameters

---

## What Not To Do

- Do not use GPT-4 or closed APIs for the self-verifier — model must judge its own outputs
- Do not commit W&B artifacts or model checkpoints — log run IDs instead
- Do not enable DoRA for production runs — too slow with 4-bit
- Do not use `do_sample=True` — greedy decoding for deterministic scoring

---

## Related Work

- **SRLM (Yuan et al. 2024):** Qualitative observation of judge-generator collapse. We quantify it.
- **STaR (Zelikman et al. 2022):** Ground-truth-anchored baseline — no collapse by design. Useful contrast.
- **AZR (NeurIPS 2025 spotlight):** Direct foil. Assumes self-verification stays calibrated. We test that assumption.
- **Gao et al. 2023 (RM overoptimization):** RM scores diverge from GT under optimization pressure. We show same emerges endogenously in self-supervised loops.
- **Mind the Gap (ICLR 2025):** Measures generation-verification gap as function of pretraining compute (static). We measure dynamic drift during self-improvement.

---

## Phase Gates

| Gate                      | Deadline    | Status |
| ------------------------- | ----------- | ------ |
| Research question locked  | March 27    | ✅ Done |
| Sanity check              | March 28    | ✅ Done (April 2 — late but passed) |
| Core experiments complete | April 25    | ⏳ Baseline + injection runs needed |
| Full draft                | May 3       | ⏳ |
| Abstract registered       | May 5       | ⏳ |
| Paper submitted           | May 7       | ⏳ |
