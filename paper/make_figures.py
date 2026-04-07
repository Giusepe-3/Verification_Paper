"""
make_figures.py — generate all paper figures from experiment CSVs.

Run from the paper/ directory:
    python make_figures.py

Outputs (paper/figures/):
    fig1_baseline_injection.pdf/.png  — Fig 1: baseline dynamics + injection comparison
    fig2_cross_model.pdf/.png         — Fig 2: three collapse regimes + GT-anchored control
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
PAPER = Path(__file__).parent          # paper/
ROOT  = PAPER.parent                   # repo root
LOGS  = ROOT / "logs"
OUT   = PAPER / "figures"
OUT.mkdir(exist_ok=True)

# ── style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "legend.fontsize":    8,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "lines.linewidth":    1.6,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
})

BLUE   = "#2166ac"
ORANGE = "#d6604d"
GREEN  = "#4dac26"
PURPLE = "#7b3294"
GRAY   = "#888888"

# ── data ───────────────────────────────────────────────────────────────────
base  = pd.read_csv(LOGS / "baseline"         / "metrics.csv")
inj   = pd.read_csv(LOGS / "injection"        / "metrics.csv")
gt    = pd.read_csv(LOGS / "gt_anchored"      / "metrics.csv")
mist  = pd.read_csv(LOGS / "mistral_baseline" / "metrics.csv")
llama = pd.read_csv(LOGS / "llama3_baseline"  / "metrics.csv")

iters = base["iteration"]


# ══════════════════════════════════════════════════════════════════════════
# Figure 1 — Baseline dynamics (left) + Injection vs Baseline gap (right)
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7))
fig.subplots_adjust(wspace=0.40)

# ── (a) left: baseline score trajectories ──
ax = axes[0]
ax.plot(iters, base["self_score_mean"], color=BLUE,
        label=r"Self-score $\bar{s}^{\,\mathrm{self}}$")
ax.plot(iters, base["gt_score_train"],  color=ORANGE,
        label=r"GT-train $\bar{s}^{\,\mathrm{gt}}_{\mathrm{train}}$")
ax.plot(iters, base["gt_score_val"],    color=GREEN,  linestyle="--",
        label=r"GT-val $\bar{s}^{\,\mathrm{gt}}_{\mathrm{val}}$")
ax.fill_between(iters, base["gt_score_train"], base["self_score_mean"],
                alpha=0.12, color=BLUE)

# annotate the final gap
ax.annotate(
    r"$\Delta_{20}=0.32$",
    xy=(20, (base["self_score_mean"].iloc[-1] + base["gt_score_train"].iloc[-1]) / 2),
    xytext=(15, 0.72),
    arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8),
    fontsize=7.5, color=GRAY,
)

ax.set_xlabel("Iteration $t$")
ax.set_ylabel("Score")
ax.set_xlim(1, 20)
ax.set_ylim(0, 1.0)
ax.set_title("(a) Baseline — Qwen2.5-7B")
ax.legend(loc="upper left", frameon=False)

# ── (b) right: gap comparison ──
ax = axes[1]

# shade injection steps (every 3rd iteration starting at 3)
for xi in range(3, 21, 3):
    ax.axvline(xi, color=GRAY, linewidth=0.6, linestyle=":", alpha=0.6)

ax.plot(iters, base["gap"], color=BLUE,
        label="Baseline", zorder=3)
ax.plot(iters, inj["gap"],  color=ORANGE, linestyle="--",
        label="Injection (every 3 iters)", zorder=3)

# annotate 47% reduction
ax.annotate("", xy=(20, inj["gap"].iloc[-1]), xytext=(20, base["gap"].iloc[-1]),
            arrowprops=dict(arrowstyle="<->", color=GRAY, lw=0.9))
ax.text(19.4, (inj["gap"].iloc[-1] + base["gap"].iloc[-1]) / 2,
        "−47%", ha="right", va="center", fontsize=7.5, color=GRAY)

ax.set_xlabel("Iteration $t$")
ax.set_ylabel(r"Verification gap $\Delta_t$")
ax.set_xlim(1, 20)
ax.set_ylim(0, 0.55)
ax.set_title("(b) Hard-negative injection reduces gap by 47\\%")
ax.legend(loc="upper left", frameon=False)
ax.text(0.97, 0.04, "dotted lines = injection steps",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=GRAY)

fig.savefig(OUT / "fig1_baseline_injection.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig1_baseline_injection.png", bbox_inches="tight", dpi=200)
plt.close(fig)
print("Saved fig1_baseline_injection")


# ══════════════════════════════════════════════════════════════════════════
# Figure 2 — Cross-model collapse regimes (+ GT-anchored control)
# ══════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(4.8, 2.7))

ax2.plot(iters, base["gap"],  color=BLUE,   zorder=4,
         label="Qwen2.5-7B — gradual")
ax2.plot(iters, mist["gap"],  color=PURPLE, zorder=3,
         label="Mistral-7B — immediate lock-in")
ax2.plot(iters, llama["gap"], color=ORANGE, linestyle="--", zorder=3,
         label="Llama-3-8B — transient peak")
ax2.plot(iters, gt["gap"],    color=GREEN,  linestyle=(0, (4, 1, 1, 1)), zorder=3,
         label="Qwen GT-anchored — control")

# annotate Mistral final gap
ax2.text(20.2, mist["gap"].iloc[-1], "0.85", va="center", fontsize=7.5, color=PURPLE)
# annotate Llama-3 peak
peak_iter = llama["gap"].idxmax() + 1
ax2.annotate(
    f"peak $\\Delta_{{13}}=0.37$",
    xy=(peak_iter, llama["gap"].max()),
    xytext=(peak_iter - 5, llama["gap"].max() + 0.06),
    arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8),
    fontsize=7.5, color=ORANGE,
)

ax2.set_xlabel("Iteration $t$")
ax2.set_ylabel(r"Verification gap $\Delta_t$")
ax2.set_xlim(1, 20)
ax2.set_ylim(0, 1.0)
ax2.set_title("Three collapse regimes across model families")
ax2.legend(loc="upper left", frameon=False, fontsize=7.5)
ax2.axhline(0, color="black", linewidth=0.5, linestyle="-")

fig2.savefig(OUT / "fig2_cross_model.pdf", bbox_inches="tight")
fig2.savefig(OUT / "fig2_cross_model.png", bbox_inches="tight", dpi=200)
plt.close(fig2)
print("Saved fig2_cross_model")

print(f"\nAll figures written to {OUT}")
