"""Three-panel convergence at N=500 (50 rounds): one panel per seed in the
reporting set {0,2,3}, every aggregation rule's test accuracy vs round. Shows the
top-tier tie (CB-SAFE+ / FedGT) and the coordinate-wise collapse in one view.
Serif / Okabe-Ito shared style; CB-SAFE+ = heavy black."""

import _bootstrap  # noqa: F401
import csv
import os

import numpy as np
import plot_style as ts
import matplotlib.pyplot as plt

ts.apply()

DIR = os.path.join(_bootstrap.RESULTS, "scale50")
SEEDS = [0, 3, 2]                       # N=500: real seed 3 shown as "seed 1"; real seed 1 dropped
# (csv-stem, STYLE-key) in legend/draw order: ours + peers first, baselines after
METHODS = [   # FedGT omitted: hardcoded for n<=31, no configuration at N=500
    ("hybrid_ov4", "hybrid_ov4"), ("fltrust", "fltrust"),
    ("median", "median"), ("geomedian", "geomedian"), ("krum", "krum"),
    ("bulyan", "bulyan"), ("trimmed", "trimmed"), ("mean", "mean"),
]
CHANCE = 0.10


def load(stem, seed):
    p = os.path.join(DIR, f"robust_signflip_{stem}_N500_f20_c3_s{seed}.csv")
    if not os.path.exists(p):
        return None, None
    r = list(csv.DictReader(open(p)))
    return (np.array([int(x["round"]) for x in r]),
            np.array([float(x["acc"]) for x in r]))


fig, axes = plt.subplots(1, 3, figsize=(ts.COL_DOUBLE, 2.55), sharex=True, sharey=True)
handles, labels = [], []
for ax, seed in zip(axes, SEEDS):
    ax.axhline(CHANCE, color="#c9c9c9", lw=0.7, ls=(0, (2, 2)), zorder=1)
    for stem, key in METHODS:
        rounds, acc = load(stem, seed)
        if rounds is None:
            continue
        st = ts.style(key)
        ln, = ax.plot(rounds, acc, color=st["color"], marker=st["marker"], ls=st["ls"],
                      lw=st["lw"], markersize=3.0, markevery=8, zorder=(5 if st.get("ours") else 3))
        if not labels or st["label"] not in labels:
            handles.append(ln); labels.append(st["label"])
    ax.set_xlim(1, 50); ax.set_ylim(0.05, 0.74)
    ax.set_xticks([1, 10, 20, 30, 40, 50]); ax.set_yticks([0.1, 0.3, 0.5, 0.7])
    ax.set_xlabel("Communication round")
    idx = SEEDS.index(seed)
    ax.text(0.03, 0.97, f"({'abc'[idx]}) seed {idx}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.5)
axes[0].set_ylabel("Test accuracy")

# one shared legend above all panels, ordered as drawn
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.13),
           ncol=5, frameon=False, columnspacing=1.3, handlelength=2.0, fontsize=6.6)

out_pdf = os.path.join(_bootstrap.RESULTS, "figs", "fig_convergence_allmethods_n500.pdf")
out_png = os.path.join(_bootstrap.RESULTS, "figs", "fig_convergence_allmethods_n500.png")
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
present = [s for s, _ in METHODS if load(s, SEEDS[0])[0] is not None]
print("wrote", out_pdf)
print("methods present:", present)
