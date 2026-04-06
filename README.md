# Verification Collapse in Iterative Self-Improving Language Models

> **NeurIPS 2026 Submission** · Abstract due May 5 · Paper due May 7, 2026

---

## Overview

Self-improving language models are trained to iteratively refine themselves using their own outputs as supervision. A tacit assumption underlying this paradigm is that the model's self-verification signal remains calibrated — that a model which judges itself correct is, in fact, more likely to be correct.

**This paper tests and breaks that assumption.**

We document a failure mode we call **verification collapse**: in iterative self-improvement loops, the gap between a model's self-assigned verification score and external ground-truth accuracy grows monotonically with training iterations. The model becomes an increasingly confident but increasingly unreliable judge of its own outputs. Crucially, this drift is not accompanied by genuine performance gains — ground-truth accuracy on held-out data remains flat while self-reported confidence climbs.

We further show that **adversarial hard-negative injection** — periodically reintroducing high-confidence-wrong examples into the fine-tuning batch — slows or bounds this collapse, suggesting a practical mitigation path.

---

## Core Result (Baseline)

A 20-iteration run on **Qwen2.5-7B-Instruct** (BF16, H100 SXM, MATH Level 3–5) confirms a clean, unambiguous verification collapse signal:

| Iteration | Self-Score | GT Val Accuracy | Gap |
|:---------:|:----------:|:---------------:|:---:|
| 1  | 0.465 | 0.400 | 0.105 |
| 5  | 0.620 | 0.420 | 0.220 |
| 10 | 0.675 | 0.440 | 0.250 |
| 15 | 0.750 | 0.440 | 0.295 |
| 20 | 0.770 | 0.460 | **0.320** |

- **Gap grew 3× over 20 iterations** (0.105 → 0.320)
- **GT validation accuracy: flat** (oscillates 0.36–0.46, no trend)
- **Self-score: monotonically increasing** (0.465 → 0.770)
- **Training loss near-zero by iteration 10** — model memorises self-judged-correct completions, not correct reasoning
- **Hard-negative count grew 25 → 67** — the model accumulates high-confidence errors as training progresses

---

## The Paper in Two Figures

1. **Diverging curves:** Self-score and GT accuracy plotted over 20 iterations. The gap opens monotonically — the signature of verification collapse.
2. **Bounded gap:** Same setup with adversarial injection every 3 iterations. The gap grows more slowly or plateaus — the injection forces the verifier to recalibrate.

---

## Method

**Domain:** MATH benchmark (Level 3–5), 250 problems, cached at `data/math_subset.json`

**Model:** Qwen2.5-7B-Instruct, BF16, Flash Attention 2, QLoRA (r=16), greedy decoding

**Iterative loop:**

```
for t = 1, 2, ..., T:
    1. Generate solutions for N training problems
    2. Dual-score: self-score (model judges own answer) + GT score (exact match)
    3. Fine-tune on problems the model judges correct (self_score > 0)
    4. Evaluate GT accuracy on held-out validation set
    5. [Injection variant] Every 3 iterations: inject 50% hard negatives
       (high-confidence-wrong examples from previous iterations)
```

**Key metrics per iteration:**

| Metric | Description |
|--------|-------------|
| `self_score_mean` | Fraction judged correct by model (train batch) |
| `gt_score_train` | Fraction actually correct (train batch, ground truth) |
| `gt_score_val` | Fraction actually correct (held-out val, ground truth) |
| `gap` | `self_score_mean − gt_score_train` — should grow monotonically |
| `loss` | Fine-tuning cross-entropy |
| `num_hard_negatives` | High-confidence-wrong examples accumulated |

---

## Relation to Prior Work

| Work | Relation |
|------|----------|
| **SRLM** (Yuan et al. 2024) | Qualitative observation of judge-generator collapse. We quantify and operationalise it. |
| **STaR** (Zelikman et al. 2022) | Ground-truth-anchored baseline — no collapse by design. A useful upper-bound contrast. |
| **AZR** (NeurIPS 2025) | Assumes self-verification remains calibrated. We empirically test and falsify that assumption. |
| **Gao et al. 2023** (RM overoptimization) | Reward model scores diverge from GT under optimisation pressure. We show the same emerges endogenously in self-supervised loops. |
| **Mind the Gap** (ICLR 2025) | Measures generation–verification gap as a function of pretraining compute (static). We measure *dynamic drift* during self-improvement. |

---

## Repository Structure

```
verification-collapse/
├── run_experiment.py                    # Entry point (--config, --iterations)
├── config.yaml                          # Reference config (full 20-iter / 500-sample)
├── src/
│   ├── experiment.py                    # Main loop: VerificationCollapseExperiment
│   ├── verifier.py                      # ModelVerifier: generate, score, finetune
│   ├── math_loader.py                   # MATH dataset loading + answer verification
│   └── utils.py                         # compute_gap, summarise_iteration, hard-neg mining
├── experiments/
│   └── configs/
│       ├── sanity_check.yaml            # 50 samples, 3 iters — quick smoke test
│       ├── baseline.yaml                # 250 samples, 20 iters, injection DISABLED
│       └── injection.yaml               # 250 samples, 20 iters, injection every 3 iters
├── data/
│   └── math_subset.json                 # 250 MATH L3–5 problems, committed to git
├── logs/                                # CSV outputs per run (gitignored)
├── models/                              # Checkpoints (gitignored)
├── notebooks/
│   └── sanity_check.ipynb
├── notes/
│   └── lab_notebook.md                  # Updated after every run
└── paper/
    └── draft.tex
```

---

## Quickstart

### Requirements

```bash
pip install -r requirements.txt
```

Flash Attention 2 is optional but recommended for H100/A100. The code falls back to `sdpa` if unavailable.

### Run sanity check (3 iterations, ~5 min on A100)

```bash
python run_experiment.py --config experiments/configs/sanity_check.yaml
```

### Run baseline (20 iterations — requires H100 SXM or equivalent)

```bash
mkdir -p logs/baseline models
nohup python run_experiment.py --config experiments/configs/baseline.yaml \
  > logs/baseline_run.log 2>&1 &
tail -f logs/baseline_run.log
```

### Run injection experiment

```bash
mkdir -p logs/injection models
nohup python run_experiment.py --config experiments/configs/injection.yaml \
  > logs/injection_run.log 2>&1 &
tail -f logs/injection_run.log
```

Results are written to `logs/<run_name>/metrics.csv`. W&B logging is optional — set `wandb.enabled: true` in the config if credentials are available.

---

## Experimental Status

| Experiment | Status | Key Result |
|------------|--------|------------|
| Sanity check (3 iter, 40 samples) | Done | Signal direction correct; too noisy to confirm monotonicity |
| **Baseline (20 iter, 200 samples)** | **Done** | **Gap 0.105 → 0.320 (3×) — clean collapse signal** |
| **Injection (20 iter, 200 samples, hard-neg every 3)** | **Done** | **Steady-state gap ~0.170 (−47% vs baseline at iter 20); gt_val trends to 0.50–0.52** |

---

## Hardware Notes

- **Recommended:** H100 SXM (80 GB) or equivalent
- Generation with `gen_batch_size=128` consumes ~77 GB. An explicit `torch.cuda.empty_cache()` between generation and fine-tuning is required to avoid OOM on the backward pass.
- BF16 (no quantization) is used for production runs. 4-bit NF4 + DoRA is disabled — it caused 20+ min/sample generation latency.
- Greedy decoding (`do_sample=False`) is enforced throughout for deterministic scoring.

---

## Citation

> Preprint forthcoming. NeurIPS 2026 submission.

---

## License

Research code. Not for production use.
