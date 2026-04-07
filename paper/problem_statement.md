# Problem Statement

Working document for `draft.tex` §1 (Introduction) and §3 (Setup → Measurements).
Verified against papers: April 2026.

---

## Core claim

In iterative self-improving loops, the gap between a model's self-assigned verification score and external ground truth accuracy grows monotonically with training iterations — a failure mode we call **verification collapse**.

Formally: let `s_t^self` be the mean self-score on the training batch at iteration `t`, and `s_t^ext` be the mean ground-truth accuracy on the same batch. Define:

```
Δ_t = s_t^self − s_t^ext
```

**Claim:** `Δ_t` grows monotonically with `t` in the absence of external grounding signal.

**Empirical result:** `Δ_1 = 0.105`, `Δ_{20} = 0.320` (3× growth). `s_t^ext` stays flat throughout. The model is not learning to solve problems — it is learning to ratify its own wrong answers.

---

## Distinction from prior work

Song et al. (2025) employ self-verification scores as a filtering mechanism in iterative self-improvement (Algorithm 1, Appendix D), measuring only downstream benchmark accuracy at each round. They do not instrument the verification scores themselves. Our paper measures what their setup leaves unobserved: whether `s_t^self` drifts above `s_t^ext` across iterations — the calibration of the verifier producing those scores, not just the outcomes it selects.

### Longer technical distinction (for reviewers / Appendix)

Song et al. (2025) use the self-verification score `s(x,y) = û_{f_{t-1}}(x,y)` purely as a filter in Algorithm 1: examples with score above threshold τ enter the fine-tuning batch; the score value itself is never recorded. Their instrumentation observes only the downstream benchmark accuracy at each round. Accordingly, their key finding — that the generation-verification gap collapses to zero within 2–3 rounds due to diversity collapse — concerns the verifier's ability to *rank* candidates, not its *calibration* to reality. A model could be monotonically overconfident across all 20 of their iterative rounds (i.e., `Δ_t → 1` in our notation) and their setup would not detect it, because `Δ_t` is not in their measurement set. We close this gap by tracking `Δ_t` explicitly at every iteration and showing it grows 3× over 20 rounds in a controlled setting.

---

## Why it matters

Any self-supervised loop that uses the model's own verification signal as a training filter is vulnerable to verification collapse:

- **STaR-like methods**: If the rationalization step is dropped or approximated, the external anchor is lost.
- **AZR**: Explicitly assumes self-verification stays calibrated. We show it does not.
- **RLHF with learned reward models**: External RM provides a partial anchor, but the RM can itself overfit (Gao et al., 2023). In our setting, there is no external RM — collapse is endogenous.

---

## The mitigation

Adversarial hard-negative injection: every 3 iterations, replace 50% of the fine-tuning batch with examples where `s_t^self ≥ 0.7` but the extracted answer is wrong, trained toward the gold solution.

Result at iteration 20: `Δ_{20} = 0.170` vs baseline `Δ_{20} = 0.320` — **47% reduction** with no external oracle.
