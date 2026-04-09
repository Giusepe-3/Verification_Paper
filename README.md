# Verification Collapse in Iterative Self-Improving Language Models

> **NeurIPS 2026 Submission** · Abstract due May 5 · Paper due May 7, 2026

---

## Overview

Self-improving language models are trained to iteratively refine themselves using their own outputs as supervision. A tacit assumption underlying this paradigm is that the model's self-verification signal remains calibrated — that a model which judges itself correct is, in fact, more likely to be correct.

**This paper tests and breaks that assumption.**

We document a failure mode we call **verification collapse**: in iterative self-improvement loops, the gap between a model's self-assigned verification score and external ground-truth accuracy grows monotonically with training iterations. The model becomes an increasingly confident but increasingly unreliable judge of its own outputs. Crucially, this drift is not accompanied by genuine performance gains — ground-truth accuracy on held-out data remains flat while self-reported confidence climbs.

We further show that **adversarial hard-negative injection** — periodically reintroducing high-confidence-wrong examples into the fine-tuning batch — slows or bounds this collapse, suggesting a practical mitigation path.

---

## Core Results

### Baseline — Qwen2.5-7B (gradual monotonic collapse)

20-iteration run on **Qwen2.5-7B-Instruct** (BF16, H100 SXM, MATH Level 3–5):

| Iteration | Self-Score | GT Val Accuracy | Gap |
|:---------:|:----------:|:---------------:|:---:|
| 1  | 0.465 | 0.400 | 0.105 |
| 5  | 0.620 | 0.420 | 0.220 |
| 10 | 0.675 | 0.440 | 0.250 |
| 15 | 0.750 | 0.440 | 0.295 |
| 20 | 0.770 | 0.460 | **0.320** |

Gap grew **3× over 20 iterations**. GT validation accuracy flat. Training loss near-zero by iter 10.

### Injection — Qwen2.5-7B (bounded gap)

Same setup with adversarial hard-negative injection every 3 iterations:

- Steady-state gap: **~0.170 at iter 20** (−47% vs baseline)
- GT val accuracy trends upward to 0.50–0.52 (vs flat 0.38–0.46 in baseline)

### GT-Anchored Control — Qwen2.5-7B (no collapse)

Same loop but fine-tuning filtered on ground-truth correctness instead of self-score:

- Gap **decreases** from 0.190 to 0.122 over 20 iterations
- GT train accuracy improves **+78%** (0.300 → 0.535)
- Proves self-scoring is the causal driver of collapse, not iterative fine-tuning per se

### Cross-Model Results

| Model | Regime | Gap at iter 1 | Gap at iter 20 |
|---|---|---|---|
| Qwen2.5-7B | Gradual monotonic | 0.105 | 0.320 |
| Llama-3-8B | Transient (peak iter 13, partial recovery) | 0.058 | 0.148 |
| Mistral-7B | Immediate lock-in | 0.580 | 0.850 |

Collapse is universal across model families. Severity scales inversely with baseline capability on the domain.

---

## Method

**Domain:** MATH benchmark (Level 3–5), `EleutherAI/hendrycks_math`

**Models:** Qwen2.5-7B-Instruct (primary), Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3

**Training:** BF16, Flash Attention 2, QLoRA (r=16, α=32, all-linear, no DoRA), greedy decoding

**Iterative loop:**

```
for t = 1, 2, ..., T:
    1. Generate solutions for N training problems
    2. Self-score: model judges own answer (yes/no) without reference
    3. GT-score: extract answer, compare to reference
    4. Fine-tune on problems the model judges correct (self_score > 0)
    5. Evaluate GT accuracy on held-out validation set
    6. [Injection variant] Every N iters: inject X% hard negatives
       (high-confidence-wrong examples trained toward gold solution)
```

**Key metrics per iteration:**

| Metric | Description |
|--------|-------------|
| `self_score_mean` | Fraction judged correct by model (train batch) |
| `gt_score_train` | Fraction actually correct (train batch, ground truth) |
| `gt_score_val` | Fraction actually correct (held-out val, ground truth) |
| `gap` | `self_score_mean − gt_score_train` — the verification gap Δ_t |
| `loss` | Fine-tuning cross-entropy |
| `num_hard_negatives` | High-confidence-wrong examples accumulated |

---

## Relation to Prior Work

| Work | Relation |
|------|----------|
| **SRLM** (Yuan et al. 2024) | Qualitative observation of judge-generator collapse. We quantify it over 20 iterations and propose a mitigation. |
| **STaR** (Zelikman et al. 2022) | Ground-truth-anchored — no collapse by design. We use this as our GT-anchored control condition. |
| **AZR** (Zhao et al., NeurIPS 2025) | Eliminates human labels but delegates verification to a code executor. Externally grounded. Our collapse findings apply to domains without such an anchor. |
| **Mind the Gap** (Song et al., ICLR 2025) | Measures filtering utility of self-verification (GV-Gap). We measure calibration drift (Δ_t). Orthogonal quantities — GV-Gap → 0 and Δ_t → large can coexist. |
| **Beyond Accuracy** (Li et al. 2025) | Measures ECE on MMLU over 5 rounds. We measure Δ_t on MATH over 20 rounds. Different metric, domain, duration, and fix. |
| **EpiCaR** (2026) | Calibration metrics on MATH over 3 iterations; fix requires objective modification. We extend to 20 iterations with a data-side-only fix. |
| **RLSR** (2025) | Self-reward divergence under RL/GRPO. We show the same collapse in pure SFT — no RL pressure required. |
| **Gao et al. 2023** | External RM scores diverge under optimisation pressure. We show the same divergence emerging endogenously with no external RM. |

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
│       ├── baseline.yaml                # 250 samples, 20 iters, injection DISABLED ✅
│       ├── injection.yaml               # 250 samples, 20 iters, injection every 3 ✅
│       ├── gt_anchored.yaml             # 500 samples, 20 iters, GT filter ✅
│       ├── llama3_baseline.yaml         # 500 samples, 20 iters, Llama-3-8B ✅
│       ├── mistral_baseline.yaml        # 500 samples, 20 iters, Mistral-7B ✅
│       ├── random_negatives.yaml        # 500 samples, 20 iters, threshold=0.0 ⏳
│       ├── injection_interval1.yaml     # 500 samples, 20 iters, inject every iter ⏳
│       ├── injection_interval6.yaml     # 500 samples, 20 iters, inject every 6 ⏳
│       ├── injection_ratio25.yaml       # 500 samples, 20 iters, 25% ratio ⏳
│       └── math_l5_baseline.yaml        # 500 samples, 20 iters, Level 5 only ⏳
├── data/
│   └── math_subset.json                 # 250 MATH L3–5 problems, committed to git
├── logs/                                # CSV outputs per run (gitignored)
├── models/                              # Checkpoints (gitignored)
├── notebooks/
│   └── sanity_check.ipynb
├── notes/
│   └── lab_notebook.md                  # Updated after every run
└── paper/
    ├── draft.tex                        # Main LaTeX — compiles on Overleaf
    ├── related_work.md                  # Positioning notes for §2
    └── problem_statement.md             # Core claim and Δ_t definition
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

### Run an experiment (production)

```bash
python run_experiment.py --config experiments/configs/baseline.yaml
```

Results are written to `logs/<run_name>/metrics.csv`. W&B logging is optional — set `wandb.enabled: true` in the config if credentials are available.

---

## Experimental Status

| Experiment | Status | Key Result |
|------------|--------|------------|
| Sanity check (3 iter, 40 samples) | ✅ | Signal direction correct |
| Baseline — Qwen (20 iter, 200 samples) | ✅ | Gap 0.105→0.320 (3×) |
| Injection — Qwen (20 iter, inject every 3) | ✅ | Steady-state gap ~0.170 (−47%) |
| GT-anchored control — Qwen | ✅ | Gap decreases; gt_train +78% |
| Llama-3-8B baseline | ✅ | Transient collapse (peak 0.368, iter 13) |
| Mistral-7B baseline | ✅ | Immediate collapse (0.580→0.850) |
| random_negatives | ⏳ | |
| math_l5_baseline | ⏳ | |
| injection_interval1/6, injection_ratio25 | ⏳ | |

---

## Hardware Notes

- **Minimum:** 80 GB VRAM — `gen_batch_size=128` fills ~77 GB. H100 PCIe or A100 80GB both work.
- An explicit `torch.cuda.empty_cache()` after generation and after fine-tuning is required to avoid OOM on the backward pass.
- BF16 (no quantization). DoRA disabled — caused 20+ min/sample latency with NF4.
- Greedy decoding (`do_sample=False`) enforced throughout for deterministic scoring.
- Llama-3 requires `export HF_TOKEN=hf_...` with accepted Meta Llama 3 license on HuggingFace.

---

## Citation

> Preprint forthcoming. NeurIPS 2026 submission.

---

## License

Research code. Not for production use.
