# Lab Notebook — Verification Collapse

---

## April 2, 2026 — Sanity Check (3 iterations, 40 train / 10 val)

**Config:** `experiments/configs/sanity_check.yaml`
**Model:** Qwen3.5-9B, 4-bit NF4, QLoRA r=16, DoRA disabled
**W&B run:** `cosmic-sound-2` (bfiqb7es)

### Results

| Iter | self | gt_train | gt_val | gap   | loss   | hard_negs |
|------|------|----------|--------|-------|--------|-----------|
| 1    | 0.20 | 0.15     | 0.30   | +0.05 | 1.2860 | 3         |
| 2    | 0.175| 0.10     | 0.30   | +0.075| 0.9255 | 3         |
| 3    | 0.225| 0.225    | 0.30   | 0.00  | 0.7052 | 1         |

### Observations

- Gap is **positive** in iters 1–2 (model overestimates itself) — direction matches paper's claim
- Gap collapses to 0 at iter 3: likely noise artifact of tiny dataset (40 samples; 1 problem = 2.5% swing)
- `gt_val` is flat at 0.30 — self-training on self-judged examples does not improve generalisation
- Hard negatives collected and injection triggered at iter 2 (interval=2 in sanity config)
- Loss decreasing normally (1.29 → 0.93 → 0.71)

### What it means

The plumbing works. The signal direction is correct. 3 iterations × 40 samples is too noisy
to confirm monotonic growth — need 10 iterations × 200+ samples.

### Bugs found and fixed during this session

1. **DoRA with 4-bit NF4** caused 20+ min/sample generation. Fixed: `use_dora: false`.
2. **Qwen3 thinking mode** caused gt_score ≈ 0 (model exhausted token budget in `<think>` blocks).
   Fixed: `enable_thinking=False` in `apply_chat_template`.
3. **Fine-tune was training on full batch**, not self-judged-correct examples only.
   Fixed: filter `batch` to `self_scores > 0` before calling `finetune()`.
4. **`dtype` vs `torch_dtype`**: transformers version uses `dtype`, not `torch_dtype`.

---

## Next Runs

Run baseline overnight, then injection the following night.

```bash
# Baseline (no injection)
nohup python run_experiment.py --config experiments/configs/baseline.yaml > logs/baseline_run.log 2>&1 &

# Injection (after baseline completes)
nohup python run_experiment.py --config experiments/configs/injection.yaml > logs/injection_run.log 2>&1 &
```

**What to look for in baseline:**
- `gap` trending upward over 10 iterations
- `self_score_mean` rising while `gt_score_val` stays flat or drops
- `gt_score_train` uncorrelated with `self_score_mean`

**What to look for in injection:**
- Gap growth slowed or reversed after injection iterations (3, 6, 9)
- `gt_score_val` staying more stable than in baseline
