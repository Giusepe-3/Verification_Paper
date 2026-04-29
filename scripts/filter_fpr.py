"""Compute and plot self-score filter false positive rate per iteration.

FPR_t = gap_t / self_score_mean_t  (lower bound: assumes all correct examples pass filter)

Outputs:
  - paper/fig_filter_fpr.pdf  (3-panel figure: baseline vs injection vs gt_anchored)
  - stdout table with key numbers for the paper
"""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
OUT  = ROOT / "paper" / "figures" / "fig_filter_fpr.pdf"


def load_csv(run_name):
    path = LOGS / run_name / "metrics.csv"
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            it = int(row["iteration"])
            ss = float(row["self_score_mean"])
            gt = float(row["gt_score_train"])
            gap = float(row["gap"])
            fpr = gap / ss if ss > 0 else 0.0
            rows.append({"iter": it, "self": ss, "gt": gt, "gap": gap, "fpr": fpr})
    return rows


RUNS = [
    ("baseline",     "Baseline (Qwen, 250)",    "#d62728"),
    ("baseline_500", "Baseline (Qwen, 500)",    "#ff7f0e"),
    ("injection",    "Injection (Qwen, 250)",   "#2ca02c"),
    ("gt_anchored",  "GT-anchored (Qwen, 500)", "#1f77b4"),
    ("mistral_baseline", "Mistral-7B",          "#9467bd"),
    ("llama3_baseline",  "Llama-3-8B",          "#8c564b"),
    ("full_ft_baseline", "Full FT (Qwen, 500)", "#e377c2"),
]

# Print table
print(f"{'Run':<28} {'FPR_1':>6} {'FPR_10':>6} {'FPR_20':>6}  {'gap_20/self_20'}")
print("-" * 72)
all_data = {}
for run_name, label, _ in RUNS:
    rows = load_csv(run_name)
    all_data[run_name] = rows
    fpr1 = rows[0]["fpr"]
    fpr10 = rows[9]["fpr"]
    fpr20 = rows[-1]["fpr"]
    print(f"{label:<28} {fpr1:>5.1%} {fpr10:>5.1%} {fpr20:>5.1%}  "
          f"{rows[-1]['gap']:.3f}/{rows[-1]['self']:.3f}")

# Plot: 2-row figure
fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))

# Panel (a): Qwen baselines + full FT
ax = axes[0]
for run_name in ["baseline", "baseline_500", "full_ft_baseline"]:
    rows = all_data[run_name]
    label = [l for r, l, _ in RUNS if r == run_name][0]
    color = [c for r, _, c in RUNS if r == run_name][0]
    ax.plot([r["iter"] for r in rows], [r["fpr"] for r in rows],
            "o-", markersize=3, linewidth=1.3, label=label, color=color)
ax.set_title("(a) Qwen baselines", fontsize=9, fontweight="bold")
ax.set_xlabel("Iteration", fontsize=8)
ax.set_ylabel("Filter FPR", fontsize=8)
ax.set_ylim(-0.02, 0.65)
ax.legend(fontsize=6.5, loc="upper left")
ax.tick_params(labelsize=7)

# Panel (b): Injection + GT-anchored
ax = axes[1]
for run_name in ["baseline", "injection", "gt_anchored"]:
    rows = all_data[run_name]
    label = [l for r, l, _ in RUNS if r == run_name][0]
    color = [c for r, _, c in RUNS if r == run_name][0]
    ax.plot([r["iter"] for r in rows], [r["fpr"] for r in rows],
            "o-", markersize=3, linewidth=1.3, label=label, color=color)
ax.set_title("(b) Baseline vs. mitigations", fontsize=9, fontweight="bold")
ax.set_xlabel("Iteration", fontsize=8)
ax.set_ylim(-0.02, 0.65)
ax.legend(fontsize=6.5, loc="upper left")
ax.tick_params(labelsize=7)

# Panel (c): Cross-model
ax = axes[2]
for run_name in ["baseline_500", "mistral_baseline", "llama3_baseline"]:
    rows = all_data[run_name]
    label = [l for r, l, _ in RUNS if r == run_name][0]
    color = [c for r, _, c in RUNS if r == run_name][0]
    ax.plot([r["iter"] for r in rows], [r["fpr"] for r in rows],
            "o-", markersize=3, linewidth=1.3, label=label, color=color)
ax.set_title("(c) Cross-model", fontsize=9, fontweight="bold")
ax.set_xlabel("Iteration", fontsize=8)
ax.set_ylim(-0.02, 1.0)
ax.legend(fontsize=6.5, loc="lower right")
ax.tick_params(labelsize=7)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", dpi=300)
print(f"\nSaved: {OUT}")
