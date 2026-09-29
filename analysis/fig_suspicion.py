"""Figure B: temporal suspicion separation (CB-SAFE+ reputation, ov1, CIFAR-10).
Population-mean flagged fraction of malicious vs honest clients over rounds, per
malicious fraction f, with the predicted honest rate p_h = 1-(1-f)^(c-1) and the
warmup shaded. Faithful to the implementation: exclusion is by an adaptive gap,
not a fixed tau; per-client lines are not logged (means only). Imports plot_style."""

import _bootstrap  # noqa: F401

import glob
import os

import numpy as np
import pandas as pd

import plot_style as st

st.apply()
import matplotlib.pyplot as plt  # noqa: E402

R = _bootstrap.RESULTS
FIGS = os.path.join(R, "figs")
FS = [10, 20, 30]
C = 3           # cluster size
W = 5           # warmup
MAL = "#D55E00"     # vermillion (malicious)
HON = "#0072B2"     # blue (honest)


def mean_series(f, col):
    curves = []
    for p in glob.glob(os.path.join(R, f"robust_signflip_reputation_f{f:02d}_c3_s*.csv")):
        d = pd.read_csv(p)
        if col in d.columns:
            curves.append(d[col].to_numpy())
    if not curves:
        return None, None
    n = min(len(c) for c in curves)
    M = np.vstack([c[:n] for c in curves])
    return np.arange(1, n + 1), M.mean(0)


fig, axes = plt.subplots(1, 3, figsize=(st.COL_DOUBLE, 2.2), sharey=True)
for j, f in enumerate(FS):
    ax = axes[j]
    ph = 1 - (1 - f / 100) ** (C - 1)
    ax.axvspan(0.5, W + 0.5, color="#f0f0f0", zorder=0)
    x, m = mean_series(f, "susp_mal")
    _, h = mean_series(f, "susp_hon")
    if x is not None:
        ax.plot(x, m, color=MAL, lw=1.8, marker="o", markevery=4, label="malicious (mean)")
        ax.plot(x, h, color=HON, lw=1.8, marker="s", markevery=4, label="honest (mean)")
    ax.axhline(ph, color=HON, ls=(0, (4, 2)), lw=1.0, zorder=2)
    ax.annotate(f"predicted honest $p_h={ph:.2f}$", (x[-1] if x is not None else 30, ph),
                textcoords="offset points", xytext=(-2, 4), ha="right", fontsize=6, color="#4a6b8a")
    ax.set_ylim(0, 1)
    ax.set_xlim(1, None)
    ax.set_title(None)
    ax.set_xlabel("communication round")
    if j == 0:
        ax.set_ylabel("flagged fraction $s_i^{\\,r}/r$")
    ax.text(0.02, 0.94, f"$f{{=}}{f}\\%$", transform=ax.transAxes, fontsize=8, va="top")
axes[0].annotate("warmup", (W / 2, 0.04), ha="center", fontsize=6, color="#8a8a8a")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.10), fontsize=7)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.subplots_adjust(wspace=0.08)
out = os.path.join(FIGS, "fig_suspicion")
fig.savefig(out + ".pdf")
fig.savefig(out + ".png", dpi=200)
print("wrote", out + ".pdf/.png")
