# Related Work Notes

Working document for `draft.tex` §2 (Background and Related Work).
Verified against papers: April 2026.

---

## Cluster 1 — Self-improving loops

**STaR** (Zelikman et al., 2022): Uses ground truth as an anchor at every iteration (rationalization step requires correct answer). No collapse by design — the external signal never leaves the loop. Useful contrast: shows that grounding prevents divergence.

**AZR / DeepSeek** (NeurIPS 2025 spotlight): Assumes self-verification stays calibrated across training. Does not test that assumption empirically. We provide the test — and it fails.

**ReST / RLHF variants**: Use an external reward model. Related but distinct failure mode: RM overoptimization (see Cluster 2). Not endogenous.

---

## Cluster 2 — Reward model overoptimization

**Gao et al. (2023)**: Shows that RM scores diverge from ground truth under KL-unconstrained optimization pressure (explicit RL). The RM is external to the generator.

Our distinction: we show the same score-vs-reality divergence emerging **endogenously** in closed self-supervised loops — no external RM, no explicit RL. The model is simultaneously the generator and the reward model, which is strictly more dangerous.

---

## Cluster 3 — Generation-verification gap

**Song et al. (2025) = Mind the Gap** (Yuda Song et al., ICLR 2025, arXiv:2412.02674)

> ⚠️ "Song et al. (2025)" appearing across the literature is the same paper as "Mind the Gap." One paper. Cite everywhere as Song et al. (2025) / arXiv 2412.02674.

### What they measure

`gap(f, g) = J(f[w(û_g)]) − J(f)`

The accuracy improvement obtainable by filtering a batch of generations using the self-verifier `û_g`. A positive gap means the verifier ranks good outputs above bad ones. Their key finding: this gap saturates to **zero within 2–3 rounds** of iterative self-improvement, due to diversity collapse (the model stops generating diverse candidates).

### What they do NOT measure (verified against Appendix D.3)

Algorithm 1 uses the self-verification score `s(x,y) = û_{f_{t-1}}(x,y)` purely as a **filter** — examples above threshold τ go into the training batch. The score value is never recorded. They never plot `s_t^self` against `s_t^ext`. They observe only downstream benchmark accuracy at each round, not verifier calibration.

A model could be increasingly overconfident across all 20 iterations and their setup would not detect it.

### What we measure

`Δ_t = s_t^self − s_t^ext`

Whether the model's self-assigned score **drifts above external ground truth accuracy** as training progresses. This is overconfidence drift — not whether the verifier can rank generations, but whether it stays calibrated to reality.

### Precise two-sentence positioning (use verbatim in §2)

> Song et al. (2025) define the generation-verification gap as the accuracy improvement obtainable by filtering generations via self-verification, and show this gap saturates to zero within a few rounds of iterative self-improvement. We study a complementary but distinct failure mode: not whether the verifier can rank generations correctly, but whether the model's self-assigned scores remain calibrated to ground truth as training progresses — a divergence that can grow even as Song et al.'s gap collapses.

### These are orthogonal quantities

A model can simultaneously have:
- `gap(f) → 0` (verifier stops ranking usefully — Song et al.'s finding), **AND**
- `Δ_t → large` (model becomes increasingly overconfident — our finding)

They measure different failure modes of the same mechanism.

---

## Cluster 4 — Judge-generator collapse

**SRLM** (Yuan et al., 2024): Qualitatively observes co-degradation of judge and generator in iterative loops. Flags reward hacking as an open question but does not measure it. Runs only 3 iterations. Never records `Δ_t`.

Our contribution: we quantify the phenomenon with a controlled 20-iteration experiment, define `Δ_t` as the tracking metric, and show the gap grows monotonically (baseline) and can be bounded with adversarial injection.

---

## Cluster 5 — Concurrent calibration papers (cite and distinguish)

### Li et al. (2025) — "Beyond Accuracy: The Role of Calibration in Self-Improving LLMs" (arXiv:2504.02902)

**Setup:** Llama-2-7B-chat-hf and DeepSeek-R1-Distill-Llama-8B. Dataset: MMLU (57 sub-datasets, general QA). Up to 5 rounds of iterative self-improvement. Fix: iterative temperature scaling (post-hoc recalibration applied at each round).

**What they measure:** Expected Calibration Error (ECE) — `ECE = Σ(|B_k|/N)|acc(B_k) - conf(B_k)|` over 10 bins. This is an aggregate population statistic: it bins predictions by confidence level and measures mean accuracy per bin. They plot calibration diagrams (confidence vs. accuracy distribution) but do **not** track `Δ_t = self_score_mean - gt_score_mean` as a per-iteration scalar.

**What they do NOT measure:** The raw drift of the model's mean self-judgment above mean ground-truth accuracy across iterations. ECE requires post-hoc binning; `Δ_t` is a live signal computable per iteration without calibration infrastructure. They also do not test the math reasoning domain or propose an in-loop data-side fix.

**Two-sentence positioning (use verbatim in §2):**
> Li et al. (2025) document growing overconfidence in iterative self-improvement using Expected Calibration Error — an aggregate binned statistic — on general QA (MMLU) over 5 rounds, and propose iterative temperature scaling as a post-hoc recalibration fix. We measure a complementary quantity in the math reasoning domain: the raw per-iteration drift `Δ_t = s_t^self − s_t^ext`, tracking it over 20 rounds and showing it grows 3× without any external calibration infrastructure — and propose adversarial hard-negative injection as an in-loop, oracle-free mitigation.

---

### EpiCaR (2026) — arXiv:2601.06786

**Setup:** Llama-3 (1B, 3B, 8B) and Qwen-3 (1.7B, 4B, 8B). Datasets: MATH (train), GSM8K (OOD zero-shot), MBPP (code). T=3 iterations. Metrics: ECE, AUROC, Brier Score — all aggregate/post-hoc.

**What they measure:** Calibration cost of iterative STaR-style SFT across model families and sizes, using multiple aggregate calibration metrics. They find "standard iterative SFT consistently incurs a calibration cost." Focus is on the fix: a dual-task training objective that reinforces correct reasoning while simultaneously training explicit self-evaluation on incorrect outputs.

**What they do NOT measure:** `Δ_t` as a per-iteration scalar. T=3 is too short to observe the monotonic growth trajectory. They do not run beyond 3 iterations. Their fix requires changes to the training objective architecture (dual-task loss); ours is data-side only and requires no objective modification.

**One-sentence positioning:**
> EpiCaR (2026) document calibration degradation in iterative SFT across 3 iterations using aggregate metrics (ECE, AUROC, Brier Score) and propose a dual-task training fix; we extend the diagnostic to 20 iterations, track the raw drift trajectory `Δ_t` that aggregate metrics obscure, and show an in-loop data-side mitigation requiring no objective modification.

---

### RLSR (2025) — arXiv:2505.08827

**Setup:** Qwen-2.5-7B-Instruct, Qwen-2.5-7B-DeepSeek-Distilled, Llama-3.2-3B as generators; same models plus DeepSeek-R1 as judges. Tasks: synthetic countdown puzzles and integration problems. Regime: **RL (GRPO)**, not SFT. Continuous joint updates to generator and judge.

**What they observe:** When the judge is updated jointly with the generator every training step, "the model's performance degraded as it learned to exploit its own evaluation biases" (Figure 2 shows self-reward diverging from formal reward — structurally similar to our `Δ_t`). Fix: freeze the judge (use a fixed external model like DeepSeek-R1 to evaluate).

**Critical distinction from our work:**
1. **RL vs SFT**: Their collapse happens under GRPO with *continuous* parameter updates per step. Our collapse happens in *discrete* iterative SFT rounds — a more common and practically important training regime.
2. **Continuous vs discrete**: Their judge updates every gradient step; ours updates once per iteration (20 discrete checkpoints). This means their collapse is faster but less representative of how practitioners run self-improvement loops.
3. **Fix philosophy**: They fix collapse by removing the self-judge (replacing it with an external model). We fix it while *keeping* the single-model setup via adversarial injection — no separate judge model required.
4. **Domain**: Synthetic countdown/integration puzzles vs. standardized MATH benchmark.

**Two-sentence positioning:**
> RLSR (2025) observe that continuous joint RL (GRPO) updates to generator and judge cause self-reward to diverge from formal reward, and fix this by replacing the self-judge with a frozen external model. We study the complementary SFT regime — discrete iterative fine-tuning rounds with a single self-judging model — and show the same divergence emerges without RL pressure, growing monotonically over 20 iterations, with an in-loop mitigation that requires no external judge.

---

## Citation keys (for refs.bib)

```
song2025mindthegap     — arXiv:2412.02674, ICLR 2025
zelikman2022star       — STaR
yuan2024self           — SRLM
deepseek2025azr        — AZR, NeurIPS 2025
gao2023scaling         — RM overoptimization
li2025beyondaccuracy   — arXiv:2504.02902 (Beyond Accuracy / Calibration)
epicar2026             — arXiv:2601.06786 (EpiCaR)
rlsr2025               — arXiv:2505.08827 (RLSR)
```
