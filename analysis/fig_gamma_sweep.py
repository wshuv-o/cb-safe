"""Gamma-basin figure: test accuracy vs sign-flip amplification gamma (CIFAR-10,
f=0.2, N=30, 3 seeds). Shows the coordinate-wise collapse holds across the whole
gamma>=gamma* basin (not a knife-edge at gamma*=5), while CB-SAFE+ stays top-tier.
gamma=5 is read from the main results; other gamma from results/gamma_sweep."""

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
GAMMAS = [2, 3, 5, 8, 15]
METHODS = ["median", "trimmed", "geomedian", "krum", "bulyan", "hybrid_ov4"]


def acc(agg, g):
    if g == 5:
        paths = glob.glob(os.path.join(R, f"robust_signflip_{agg}_f20_c3_s*.csv"))
    else:
        paths = glob.glob(os.path.join(R, "gamma_sweep",
                                       f"robust_signflip_{agg}_f20_c3_g{int(g * 10):03d}_s*.csv"))
    vals = [pd.read_csv(p)["acc"].tail(5).mean() * 100 for p in paths]
    return (np.mean(vals), np.std(vals)) if vals else (None, None)


fig, ax = plt.subplots(figsize=(st.COL_SINGLE, 2.5))
ax.axhline(10, color="#bdbdbd", ls=(0, (1, 2)), lw=0.7, zorder=1)
ax.axhline(59, color="#bdbdbd", ls=(0, (1, 2)), lw=0.7, zorder=1)
ax.axvline(5, color="#9e9e9e", ls=(0, (4, 3)), lw=0.7, zorder=1)
ax.annotate("$\\gamma^{*}$", (5, 12), fontsize=7, color="#6b6b6b", ha="center")
ax.annotate("clean", (14.5, 59), fontsize=6, color="#8a8a8a", va="bottom", ha="right")

for agg in METHODS:
    xs, ys = [], []
    for g in GAMMAS:
        m, _ = acc(agg, g)
        if m is not None:
            xs.append(g); ys.append(m)
    sty = st.style(agg)
    ax.plot(xs, ys, color=sty["color"], ls=sty["ls"], lw=sty["lw"], marker=sty["marker"],
            ms=4, label=sty["label"], zorder=6 if sty.get("ours") else 3)

ax.set_xscale("log")
from matplotlib.ticker import NullLocator  # noqa: E402
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticks(GAMMAS); ax.set_xticklabels([str(g) for g in GAMMAS])
ax.set_xlim(1.8, 16.5)
ax.set_ylim(5, 62)
ax.set_xlabel("sign-flip amplification $\\gamma$")
ax.set_ylabel("test accuracy (\\%)")
ax.legend(fontsize=6, frameon=False, loc="upper center",
          bbox_to_anchor=(0.5, -0.24), ncol=3, handlelength=2.0)
out = os.path.join(FIGS, "fig_gamma_basin")
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
print("wrote", out + ".pdf/.png")
