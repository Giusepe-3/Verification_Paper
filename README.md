# Verification Collapse in Iterative Self-Improving Language Models

Code and paper for an empirical study of how self-verification breaks down when language models train on their own judgments.

**Author:** Leonardo Gianola, University of Southern Denmark.
**Paper:** [`paper/draft.pdf`](paper/draft.pdf) (23 pages, preprint).
**DOI:** [10.5281/zenodo.19890194](https://doi.org/10.5281/zenodo.19890194)

---

## What this is

Self-improving language models (Self-Rewarding LMs, Meta-Rewarding, CREAM) iteratively refine themselves by selecting training examples through their own self-verification. The paradigm rests on one assumption: that self-evaluation stays calibrated against ground truth across training rounds. The same assumption underlies scalable-oversight proposals where a model judges its own outputs in place of human annotation.

This project tests that assumption and shows it fails.

We define the **verification gap** at iteration $t$:

$$\Delta_t \;=\; \bar{s}^{\text{self}}_t \;-\; \bar{s}^{\text{ext}}_t$$

the difference between the model's mean self-assigned correctness score and the mean ground-truth accuracy on the same problems. Running 20 iterations of self-improvement on the MATH benchmark (Level 3-5) across three model families, we find that $\Delta_t$ grows in every case while ground-truth accuracy stays flat or improves only slightly. We call this failure mode **verification collapse**.

We further isolate self-scoring as the causal driver (via a GT-anchored control), show the effect persists under full fine-tuning (not a LoRA artifact), and probe the mechanism with **adversarial hard-negative injection**, which cuts the steady-state gap by 47% on a capable model using 85% less ground-truth data than full GT supervision.

---

## Headline results

All runs: 20 iterations, MATH Level 3-5, BF16, greedy decoding.

| Condition | Model | Samples | $\Delta_1$ | Peak $\Delta$ | $\Delta_{20}$ |
|---|---|---:|---:|---:|---:|
| Baseline, LoRA | Qwen2.5-7B | 500 | 0.173 | 0.335 (i19) | **0.320** |
| Baseline, full FT (no LoRA) | Qwen2.5-7B | 500 | 0.183 | 0.360 | **0.338** |
| Injection ($N{=}3$, 50%) | Qwen2.5-7B | 250 | 0.105 | 0.475 | **0.170** |
| GT-anchored control | Qwen2.5-7B | 500 | 0.190 | 0.193 | **0.123** |
| Recovery (from collapsed ckpt) | Qwen2.5-7B | 500 | 0.320 | - | **0.203** |
| Baseline, self-score | Mistral-7B | 500 | 0.580 | 0.853 | **0.850** |
| Baseline, self-score | Llama-3-8B | 500 | 0.058 | 0.368 (i13) | **0.148** |

**Four concrete findings:**

1. **Three collapse regimes, ordered by base capability.** Capable models (Qwen) drift gradually; moderately capable models (Llama-3) peak and partially recover as self-judgment itself degrades; weak models (Mistral) lock in immediately.
2. **Self-scoring is the cause, not iterative SFT per se.** The GT-anchored control uses the same loop with ground-truth filtering: the gap *decreases* from 0.190 to 0.123 over 20 iterations.
3. **Not a LoRA artifact.** Full fine-tuning (all 7.6B params) yields $\Delta_{20} = 0.338$, slightly worse than LoRA's 0.320. Same three-regime shape.
4. **Injection is preventive and corrective, but capability-gated.** On Qwen it cuts the gap 47%; applied to a collapsed checkpoint it recovers 37% of the damage. On weaker Llama-3 it makes things worse by sustaining confident self-scoring the model would otherwise lose to parameter drift.

Two-seed replication (seeds 42, 43) confirms the direction and scale of both effects. Paper includes FPR analysis showing the self-score filter admits ~46% wrong examples by iteration 20 on the primary baseline.

---

## How to run

### Requirements

```bash
pip install -r requirements.txt
```

Flash Attention 2 is optional; the code falls back to PyTorch SDPA if it isn't installed.

### Smoke test (5 minutes on A100/H100)

```bash
python run_experiment.py --config experiments/configs/sanity_check.yaml
```

### Full experiment (2-3h on H100 SXM, MATH L3-5, Qwen2.5-7B)

```bash
python run_experiment.py --config experiments/configs/baseline_500.yaml
```

### Reproduce headline conditions

| Finding | Config |
|---|---|
| Primary baseline (collapse) | `experiments/configs/baseline_500.yaml` |
| Injection mitigation | `experiments/configs/injection.yaml` |
| GT-anchored control (causal isolation) | `experiments/configs/gt_anchored.yaml` |
| Full fine-tuning (LoRA confound test) | `experiments/configs/full_ft_baseline.yaml` |
| Recovery from collapsed checkpoint | `experiments/configs/recovery.yaml` |
| Cross-model: Mistral-7B | `experiments/configs/mistral_baseline.yaml` |
| Cross-model: Llama-3-8B | `experiments/configs/llama3_baseline.yaml` |
| Seed replication | `experiments/configs/baseline_seed43.yaml`, `injection_seed43.yaml` |

Llama-3 runs need `export HF_TOKEN=hf_...` with an accepted Meta Llama 3 license.

### RunPod (A100 PCIe or H100 SXM)

```bash
git clone https://github.com/Giusepe-3/Verification_Paper.git verification-collapse
cd verification-collapse
bash setup_runpod.sh <config_name>
```

`setup_runpod.sh` pins torch, skips flash-attn/bitsandbytes on pods where they segfault, and runs a pre-flight CUDA sanity check to detect broken pods before any training starts. See `CLAUDE.md` for known pod quirks (bitsandbytes segfaults, broken CUDA drivers, peft version pinning on H200).

### Output

Each run writes `logs/<run_name>/metrics.csv` with per-iteration rows:

| Column | Meaning |
|---|---|
| `self_score_mean` | Mean self-assigned correctness on train batch |
| `gt_score_train` | Mean ground-truth accuracy on train batch |
| `gt_score_val` | Mean ground-truth accuracy on held-out val |
| `gap` | $\Delta_t$ = `self_score_mean - gt_score_train` |
| `loss` | Fine-tuning cross-entropy |
| `num_hard_negatives` | High-confidence-wrong examples accumulated |

W&B is optional (`wandb.enabled: true` in the config).

---

## Repo structure

```
Verification_Paper/
├── run_experiment.py              # Entry point: --config path
├── config.yaml                    # Reference 20-iter / 500-sample config
├── requirements.txt
├── src/
│   ├── experiment.py              # Main loop; supports training_filter = self_score | gt_score
│   ├── verifier.py                # ModelVerifier: generate, score, finetune
│   ├── math_loader.py             # MATH dataset + answer extraction/verify
│   └── utils.py                   # gap computation, hard-negative mining
├── experiments/configs/           # One YAML per experimental condition
├── setup_runpod.sh                # Single-experiment pod bootstrap
├── setup_recovery_pod.sh          # baseline_500 -> recovery pipeline
├── scripts/                       # Figure generation, FPR analysis
├── data/                          # math_subset.json (250-problem subset, committed)
├── logs/                          # Per-run CSVs (gitignored)
├── models/                        # Checkpoints (gitignored)
├── notes/lab_notebook.md          # Updated after every run
├── paper/
│   ├── draft.tex                  # Main LaTeX source
│   ├── draft.pdf                  # Compiled paper
│   ├── refs.bib
│   ├── neurips_2026.sty
│   └── figures/
└── CLAUDE.md                      # Detailed project state + pod quirks
```

---

## Experimental status

All experiments in the paper are complete.

| Experiment | Status |
|---|---|
| Baseline, Qwen2.5-7B, 500 samples | Done |
| Baseline, Qwen2.5-7B, 250 samples | Done |
| Injection, Qwen2.5-7B | Done |
| GT-anchored control | Done |
| Llama-3-8B baseline | Done |
| Mistral-7B baseline | Done |
| Random negatives (any-wrong injection) | Done |
| Llama-3 injection (cross-model generalisation) | Done |
| Recovery (from collapsed checkpoint) | Done |
| Full fine-tuning baseline (LoRA confound) | Done |
| Seed 43 replication (baseline + injection) | Done |

---

## Hardware notes

- **VRAM floor:** 80 GB. `gen_batch_size=128` fills ~77 GB; anything smaller OOMs on the backward pass after generation.
- `torch.cuda.empty_cache()` is called after generation and after fine-tuning in each iteration; this is required, not a nice-to-have.
- BF16 throughout. 4-bit quantization off. DoRA off (20+ min/sample latency under NF4).
- Greedy decoding (`do_sample=False`) enforced for deterministic scoring.
- Full fine-tuning of Qwen2.5-7B needs 141 GB (H200 SXM).

---

## Related work (summary)

See `paper/related_work.md` for verified characterisations. One-line positioning:

- **SRLM** (Yuan et al., 2024): 3-iter qualitative observation of judge-generator drift. We quantify it over 20 iterations with a causal control.
- **STaR** (Zelikman et al., 2022): Ground-truth anchored by construction. We use this as our GT-anchored control condition.
- **AZR** (Zhao et al., 2025): Closed loop but verification delegated to a code executor. Externally grounded; our findings apply to domains without such an anchor.
- **Mind the Gap** (Song et al., ICLR 2025): Filtering utility (GV-Gap), not calibration drift. Orthogonal quantity; both can hold simultaneously.
- **Beyond Accuracy** (Huang et al., 2025): ECE on MMLU, 5 rounds. Different metric, domain, duration, fix.
- **EpiCaR** (2026): Calibration metrics on MATH, 3 iterations, objective-modification fix. We extend to 20 iterations with a data-side-only intervention.
- **RLSR** (2025): Self-reward divergence under GRPO. We show the same collapse without RL pressure; pure iterative SFT suffices.
- **Gao et al. (2023)**: External RM overoptimization. We show the same score-vs-reality divergence emerging endogenously with no external RM.

---

## Citation

```bibtex
@article{gianola2026verificationcollapse,
  title     = {Verification Collapse in Iterative Self-Improving Language Models},
  author    = {Gianola, Leonardo},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19890194},
  url       = {https://doi.org/10.5281/zenodo.19890194},
  note      = {Preprint}
}
```

---

## License

Research code. Not for production use.
