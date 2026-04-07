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

## Citation keys (for refs.bib)

```
song2025mindthegap     — arXiv:2412.02674, ICLR 2025
zelikman2022star       — STaR
yuan2024self           — SRLM
deepseek2025azr        — AZR, NeurIPS 2025
gao2023scaling         — RM overoptimization
```
