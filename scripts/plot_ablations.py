"""Generate fig3_ablations.pdf — 4-panel ablation figure for §5."""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
OUT  = ROOT / "paper" / "figures" / "fig3_ablations.pdf"

def load_csv(run_name):
    path = LOGS / run_name / "metrics.csv"
    iters, gaps, gt_vals = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            iters.append(int(row["iteration"]))
            gaps.append(float(row["gap"]))
            gt_vals.append(float(row["gt_score_val"]))
    return iters, gaps, gt_vals

PANELS = [
    ("random_negatives",  "(a) Random negatives",      "Qwen2.5-7B, random neg."),
    ("llama3_injection",  "(b) Llama-3 injection",     "Llama-3-8B, injection"),
    ("baseline_500",      "(c) Baseline 500",          "Qwen2.5-7B, 500 samples"),
    ("recovery",          "(d) Recovery",              "Qwen2.5-7B, post-collapse"),
]

fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=False)

for ax, (run, title, subtitle) in zip(axes, PANELS):
    iters, gaps, gt_vals = load_csv(run)

    ax.plot(iters, gaps,    "o-", color="#d62728", markersize=3, linewidth=1.4, label=r"$\Delta_t$ (gap)")
    ax.plot(iters, gt_vals, "s--", color="#1f77b4", markersize=3, linewidth=1.2, label=r"$\bar{a}^{\mathrm{val}}_t$")

    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel("Iteration", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0.5, len(iters) + 0.5)
    ax.set_ylim(-0.02, max(max(gaps), max(gt_vals)) * 1.15)
    ax.text(0.5, -0.28, subtitle, transform=ax.transAxes,
            ha="center", fontsize=7, fontstyle="italic")

axes[0].set_ylabel("Score", fontsize=8)
axes[0].legend(fontsize=7, loc="upper left", framealpha=0.8)

fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(OUT, bbox_inches="tight", dpi=300)
print(f"Saved: {OUT}")
