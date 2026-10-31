"""Generate publication-quality comparison figures for the paper.

Produces three figures from the ablation benchmark results:
  paper/figures/fig_all_schedulers_avgwait.png  — avg wait, all schedulers, both workloads
  paper/figures/fig_all_schedulers_starvation.png — starvation rate, all schedulers
  paper/figures/fig_contribution_decomp.png — step-by-step contribution breakdown
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT   = os.path.dirname(__file__)
FIGDIR = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIGDIR, exist_ok=True)

# ── Load ablation summary ────────────────────────────────────────────────────
rows = []
with open(os.path.join(ROOT, "results", "ablation_summary.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

def get(model, workload, field):
    for r in rows:
        if r["model"] == model and r["workload"] == workload:
            return float(r[field])
    return None

# ── Colour + order definitions ───────────────────────────────────────────────
CLASSICAL = ["FCFS", "Round Robin (q=2)", "Priority + Aging", "MLFQ", "CFS", "Stride"]
RL        = ["Plain MaskablePPO", "GraphSAGE-RL (uniform)"]
ORACLE    = ["SRTF (oracle)"]
ORDER     = CLASSICAL + RL + ORACLE

LABELS = {
    "FCFS":                     "FCFS",
    "Round Robin (q=2)":        "RR (q=2)",
    "Priority + Aging":         "Pri+Aging",
    "MLFQ":                     "MLFQ",
    "CFS":                      "CFS",
    "Stride":                   "Stride",
    "Plain MaskablePPO":        "PPO (flat)",
    "GraphSAGE-RL (uniform)":   "GS+PPO\n(ours)",
    "SRTF (oracle)":            "SRTF†",
}

C_CLASSICAL = "#6c757d"
C_CFS       = "#2c6fad"   # highlight CFS as the strong baseline
C_RL_FLAT   = "#f4a261"
C_RL_GRAPH  = "#2a9d8f"   # our model
C_ORACLE    = "#e76f51"

def bar_color(name):
    if name == "CFS":           return C_CFS
    if name in CLASSICAL:       return C_CLASSICAL
    if name == "GraphSAGE-RL (uniform)": return C_RL_GRAPH
    if name == "Plain MaskablePPO":      return C_RL_FLAT
    return C_ORACLE

# ── IEEE-style rcparams ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          9,
    "axes.titlesize":     9,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "figure.dpi":         200,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "axes.grid.axis":     "y",
    "grid.alpha":         0.35,
    "grid.linestyle":     "--",
})

x   = np.arange(len(ORDER))
w   = 0.35
lbl = [LABELS[m] for m in ORDER]
colors = [bar_color(m) for m in ORDER]

# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — Average Waiting Time
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0), sharey=False)

for ax, wl, title in zip(axes, ["uniform", "realistic"],
                          ["(a) Uniform workloads", "(b) Realistic workloads"]):
    vals = [get(m, wl, "average_waiting") for m in ORDER]
    ci_lo = [get(m, wl, "aw_ci_low")  for m in ORDER]
    ci_hi = [get(m, wl, "aw_ci_high") for m in ORDER]
    yerr_lo = [v - lo for v, lo in zip(vals, ci_lo)]
    yerr_hi = [hi - v  for v, hi in zip(vals, ci_hi)]

    bars = ax.bar(x, vals, color=colors, width=0.62,
                  edgecolor="white", linewidth=0.5, zorder=3)
    ax.errorbar(x, vals, yerr=[yerr_lo, yerr_hi],
                fmt="none", color="#333", capsize=3, linewidth=0.8, zorder=4)

    # Annotate our model
    idx = ORDER.index("GraphSAGE-RL (uniform)")
    ax.annotate("* Ours", xy=(x[idx], vals[idx]),
                xytext=(x[idx], vals[idx] + 1.5),
                ha="center", fontsize=7, color=C_RL_GRAPH, fontweight="bold")

    # Dashed line at CFS level
    cfs_val = get("CFS", wl, "average_waiting")
    ax.axhline(cfs_val, color=C_CFS, linewidth=1.0, linestyle=":", alpha=0.8,
               label=f"CFS baseline ({cfs_val:.1f})")

    ax.set_title(title, pad=4)
    ax.set_ylabel("Average Waiting Time (ticks)")
    ax.set_xticks(x); ax.set_xticklabels(lbl, rotation=35, ha="right")
    ax.set_ylim(0, max(vals) * 1.20)
    ax.legend(fontsize=7, framealpha=0.7)

# Legend patches
patches = [
    mpatches.Patch(color=C_CLASSICAL, label="Classical"),
    mpatches.Patch(color=C_CFS,       label="CFS (strong baseline)"),
    mpatches.Patch(color=C_RL_FLAT,   label="RL flat MLP"),
    mpatches.Patch(color=C_RL_GRAPH,  label="GraphSAGE+PPO (ours)"),
    mpatches.Patch(color=C_ORACLE,    label="SRTF oracle †"),
]
fig.legend(handles=patches, loc="lower center", ncol=5,
           fontsize=7, framealpha=0.8, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Average Waiting Time — All Schedulers vs. GraphSAGE+MaskablePPO",
             fontsize=9, y=1.01)
plt.tight_layout(rect=[0, 0.06, 1, 1])
out = os.path.join(FIGDIR, "fig_all_schedulers_avgwait.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — Starvation Rate
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0), sharey=False)

for ax, wl, title in zip(axes, ["uniform", "realistic"],
                          ["(a) Uniform workloads", "(b) Realistic workloads"]):
    vals = [get(m, wl, "starvation_rate") * 100 for m in ORDER]

    bars = ax.bar(x, vals, color=colors, width=0.62,
                  edgecolor="white", linewidth=0.5, zorder=3)

    idx = ORDER.index("GraphSAGE-RL (uniform)")
    ax.annotate("* Ours", xy=(x[idx], vals[idx]),
                xytext=(x[idx], vals[idx] + 1.5),
                ha="center", fontsize=7, color=C_RL_GRAPH, fontweight="bold")

    cfs_val = get("CFS", wl, "starvation_rate") * 100
    ax.axhline(cfs_val, color=C_CFS, linewidth=1.0, linestyle=":",
               alpha=0.8, label=f"CFS baseline ({cfs_val:.1f}%)")

    ax.set_title(title, pad=4)
    ax.set_ylabel("Starvation Rate (% processes waiting >10 ticks)")
    ax.set_xticks(x); ax.set_xticklabels(lbl, rotation=35, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, framealpha=0.7)

fig.legend(handles=patches, loc="lower center", ncol=5,
           fontsize=7, framealpha=0.8, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Starvation Rate — All Schedulers vs. GraphSAGE+MaskablePPO",
             fontsize=9, y=1.01)
plt.tight_layout(rect=[0, 0.06, 1, 1])
out = os.path.join(FIGDIR, "fig_all_schedulers_starvation.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ════════════════════════════════════════════════════════════════════════════
# Figure 3 — Contribution Decomposition (step-wise, both workloads)
# ════════════════════════════════════════════════════════════════════════════

# Data from contribution_analysis.py results
contrib = {
    "uniform": {
        "Classical (FCFS)":        (23.91, 81.6),
        "RL only (flat MLP)":      (19.30, 60.9),
        "RL + GraphSAGE\n(ours)":  (18.54, 58.7),
        "SRTF oracle †":           (14.49, 49.0),
    },
    "realistic": {
        "Classical (CFS)":         (28.57, 69.6),
        "RL only (flat MLP)":      (29.16, 60.1),
        "RL + GraphSAGE\n(ours)":  (28.42, 60.5),
        "SRTF oracle †":           (17.51, 45.2),
    },
}

fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2))

tier_colors = [C_CLASSICAL, C_RL_FLAT, C_RL_GRAPH, C_ORACLE]

for ax, (wl, data), title in zip(
        axes,
        contrib.items(),
        ["(a) Uniform workloads", "(b) Realistic workloads"]):

    tiers = list(data.keys())
    aw    = [data[t][0] for t in tiers]
    st    = [data[t][1] for t in tiers]
    xi    = np.arange(len(tiers))

    ax2 = ax.twinx()
    bars1 = ax.bar(xi - 0.18, aw, 0.33, color=tier_colors, alpha=0.85,
                   edgecolor="white", label="Avg Wait")
    bars2 = ax2.bar(xi + 0.18, st, 0.33, color=tier_colors, alpha=0.45,
                    edgecolor="white", hatch="//", label="Starvation %")

    ax.set_ylabel("Avg Waiting Time (ticks)", color="#333")
    ax2.set_ylabel("Starvation Rate (%)", color="#666")
    ax2.set_ylim(0, 100)
    ax.set_ylim(0, max(aw) * 1.25)
    ax.set_xticks(xi)
    ax.set_xticklabels(tiers, fontsize=7.5, rotation=10, ha="right")
    ax.set_title(title, pad=4)

    # Annotate deltas on avg wait
    for i in range(1, len(aw)):
        delta = aw[i] - aw[i-1]
        col   = C_RL_GRAPH if delta < 0 else "#c0392b"
        ax.annotate(f"{delta:+.2f}", xy=(xi[i] - 0.18, aw[i] + 0.3),
                    ha="center", fontsize=7, color=col, fontweight="bold")

solid  = mpatches.Patch(color="#555", label="Avg Wait (solid bar)")
hatch  = mpatches.Patch(color="#aaa", hatch="//", label="Starvation % (hatched)")
fig.legend(handles=[solid, hatch], loc="lower center", ncol=2,
           fontsize=7.5, framealpha=0.8, bbox_to_anchor=(0.5, -0.03))
fig.suptitle("Contribution Decomposition: Classical → RL → RL+Graph → Oracle",
             fontsize=9, y=1.01)
plt.tight_layout(rect=[0, 0.06, 1, 1])
out = os.path.join(FIGDIR, "fig_contribution_decomp.png")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

print("\nAll figures saved to paper/figures/")
