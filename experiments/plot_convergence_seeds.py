"""Per-seed convergence of CB-SAFE+ at N=500 over 50 rounds, showing that the
30-round budget truncated every seed mid-recovery. Reads results/scale50/.
Serif/Okabe-Ito shared style; mean is the heavy black 'ours' line."""

import _bootstrap  # noqa: F401
import csv
import os

import numpy as np
import plot_style as ts
import matplotlib.pyplot as plt

ts.apply()

DIR = os.path.join(_bootstrap.RESULTS, "scale50")
SEEDS = [0, 1, 2, 3]
CUT = 30           # old round budget
CHANCE = 0.10      # 10-class FashionMNIST


def load(seed):
    p = os.path.join(DIR, f"robust_signflip_hybrid_ov4_N500_f20_c3_s{seed}.csv")
    r = list(csv.DictReader(open(p)))
    rounds = np.array([int(row["round"]) for row in r])
    acc = np.array([float(row["acc"]) for row in r])
    return rounds, acc


# Okabe-Ito, distinct colour+marker per seed (colour-blind & greyscale safe)
SEED_STY = [
    dict(color="#56B4E9", marker="o", label="seed 0"),
    dict(color="#E69F00", marker="s", label="seed 1"),
    dict(color="#009E73", marker="^", label="seed 2"),
    dict(color="#CC79A7", marker="D", label="seed 3"),
]

fig, ax = plt.subplots(figsize=(ts.COL_SINGLE, 2.7))

# chance reference
ax.axhline(CHANCE, color="#bbbbbb", lw=0.7, ls=(0, (2, 2)), zorder=1)
ax.text(1.5, CHANCE + 0.012, "chance (10 classes)", fontsize=6, color="#8a8a8a", va="bottom")

# individual seeds (thin)
all_acc = []
for seed, sty in zip(SEEDS, SEED_STY):
    rounds, acc = load(seed)
    all_acc.append(acc)
    ax.plot(rounds, acc, color=sty["color"], marker=sty["marker"], ls="-",
            lw=1.0, markersize=3.0, markevery=6, alpha=0.85, label=sty["label"], zorder=3)

# mean +/- 1 SD (the privileged 'ours' style: heavy black)
A = np.vstack(all_acc)
m, sd = A.mean(0), A.std(0, ddof=1)
rr = np.arange(1, A.shape[1] + 1)
ax.fill_between(rr, m - sd, m + sd, color="#000000", alpha=0.10, lw=0, zorder=2)
ax.plot(rr, m, color="#000000", marker="o", ls="-", lw=2.2, markersize=3.2,
        markevery=6, label="mean $\\pm$ 1 SD", zorder=5)

# old-budget cutoff
ax.axvline(CUT, color="#D55E00", lw=1.0, ls=(0, (4, 2)), zorder=2)
m30, m50 = m[CUT - 1], m[-1]
ax.text(31.2, 0.155, f"round 30 cutoff:\nmean {m30:.2f} (mid-recovery)",
        fontsize=6.2, color="#7a3410", va="center", ha="left")
ax.text(49.3, m50 + 0.035, f"converged\nmean {m50:.2f}", fontsize=6.2,
        color="#333333", va="center", ha="right")

ax.set_xlabel("Communication round")
ax.set_ylabel("Test accuracy")
ax.set_xlim(1, 50)
ax.set_ylim(0.05, 0.74)
ax.set_xticks([1, 10, 20, 30, 40, 50])
ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
ax.legend(loc="upper left", frameon=False, ncol=1, handlelength=1.8,
          borderaxespad=0.2, labelspacing=0.25)

out_pdf = os.path.join(_bootstrap.RESULTS, "figs", "fig_convergence_n500.pdf")
out_png = os.path.join(_bootstrap.RESULTS, "figs", "fig_convergence_n500.png")
os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
fig.savefig(out_pdf)
fig.savefig(out_png, dpi=300)
print("wrote", out_pdf)
print(f"mean r30={m30:.4f}  r50={m50:.4f}  final SD={sd[-1]:.4f}")
print("per-seed final:", {s: round(float(a[-1]), 4) for s, a in zip(SEEDS, all_acc)})
