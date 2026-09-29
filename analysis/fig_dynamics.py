"""Figure A: training dynamics under sign-flip. 4 datasets (cols) x 3 malicious
fractions (rows), accuracy vs communication round, 8 aggregation rules. Mean over
seeds; +-1 SD band for CB-SAFE+ and FedGT only. Shared y-range within a column
(dataset). Imports plot_style (no local rcParams)."""

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

# Datasets where CB-SAFE+ wins with a clear margin (CIFAR-10, a tie with FedGT,
# stays in the tables but is omitted from this dynamics figure).
ROWS = [("CIFAR-10", "", 10.0, 59.0),             # (name, subdir, chance%, clean baseline%)
        ("FashionMNIST", "fmnist", 10.0, 88.5),
        ("EMNIST", os.path.join("kaggle", "emnist"), 2.1, 86.0),
        ("Edge-IIoTset", os.path.join("kaggle", "edgeiiot"), 6.7, 63.7)]
FS = [10, 20, 30]
METHODS = ["mean", "trimmed", "median", "krum", "bulyan", "geomedian", "fedgt", "hybrid_ov4"]
BAND = {"hybrid_ov4", "fedgt"}     # show +-1 SD band only for these
W = 5                              # warmup / exclusions-begin round


# FedGT curve uses its ACTUAL BCJR decoder (faithful reimplementation) at N=30;
# it has no configuration for Edge-IIoTset (tabular loader) so it drops from that row.
_FAITHFUL = os.path.join(R, "fedgt_faithful")
_DS = {"": "cifar10", "fmnist": "fmnist", os.path.join("kaggle", "emnist"): "emnist"}


def series(subdir, agg, f):
    """Return (rounds, mean_acc%, std_acc%) averaged over seeds, or None."""
    import re
    if agg == "fedgt":
        ds = _DS.get(subdir)
        paths = sorted(glob.glob(os.path.join(
            _FAITHFUL, f"faithful_fedgt_{ds}_signflip_f{f:02d}_N30_s*.csv"))) if ds else []
        curves = []
        for p in paths:
            if "_oneshot" in p:
                continue
            d = pd.read_csv(p)
            curves.append(d["acc"].to_numpy() * 100)
        if not curves:
            return None
        n = min(len(c) for c in curves)
        M = np.vstack([c[:n] for c in curves])
        return np.arange(1, n + 1), M.mean(0), M.std(0)
    base = os.path.join(R, subdir) if subdir else R
    paths = sorted(glob.glob(os.path.join(base, f"robust_signflip_{agg}_f{f:02d}_c3_s*.csv")))
    curves = []
    for p in paths:
        core = re.sub(r"_f\d+_c3_s\d+\.csv$", "", os.path.basename(p)).replace("robust_signflip_", "")
        if core != agg:
            continue
        d = pd.read_csv(p)
        curves.append(d["acc"].to_numpy() * 100)
    if not curves:
        return None
    n = min(len(c) for c in curves)
    M = np.vstack([c[:n] for c in curves])
    rounds = np.arange(1, n + 1)
    return rounds, M.mean(0), M.std(0)


fig, axes = plt.subplots(3, 4, figsize=(st.COL_DOUBLE, 5.0), sharex=True)
handles_labels = {}
for c, (dsname, subdir, chance, base) in enumerate(ROWS):   # datasets -> columns
    # shared y-range within the column (per dataset)
    ymin = min(chance - 3, 5)
    ymax = base + 5
    for r, f in enumerate(FS):                              # fractions -> rows
        ax = axes[r][c]
        ax.axhline(chance, color="#bdbdbd", ls=(0, (1, 2)), lw=0.7, zorder=1)
        ax.axhline(base, color="#bdbdbd", ls=(0, (1, 2)), lw=0.7, zorder=1)
        ax.axvline(W, color="#9e9e9e", ls=(0, (4, 3)), lw=0.7, zorder=1)
        for agg in METHODS:
            s = series(subdir, agg, f)
            if s is None:
                continue
            x, m, sd = s
            sty = st.style(agg)
            line, = ax.plot(x, m, color=sty["color"], ls=sty["ls"], lw=sty["lw"],
                            marker=sty["marker"], markevery=4, zorder=5 if sty.get("ours") else 3)
            if agg in BAND and len(x) > 1:
                ax.fill_between(x, m - sd, m + sd, color=sty["color"], alpha=0.15, lw=0, zorder=2)
            handles_labels[sty["label"]] = line
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(1, None)
        if r == 0:
            ax.set_title(dsname)
        if r == len(FS) - 1:
            ax.set_xlabel("communication round")
        if c == 0:
            ax.set_ylabel(f"$f{{=}}{f}\\%$\ntest accuracy (\\%)")
# annotate "exclusions begin" once, top-left panel
axes[0][0].annotate("exclusions begin", xy=(W, axes[0][0].get_ylim()[0] + 6),
                    xytext=(W + 1.5, axes[0][0].get_ylim()[0] + 3),
                    fontsize=6, color="#6b6b6b")

# one shared legend above the grid, 4 columns
labels = ["Mean (no robustness)", "Trimmed mean", "Median", "Multi-Krum",
          "Bulyan", "Geo-median/RFA", "FedGT [18]", "CB-SAFE+ (ours)"]
hs = [handles_labels[l] for l in labels if l in handles_labels]
ls = [l for l in labels if l in handles_labels]
fig.legend(hs, ls, ncol=4, frameon=False, loc="upper center",
           bbox_to_anchor=(0.5, 1.045), fontsize=7, handlelength=2.4)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.subplots_adjust(hspace=0.18, wspace=0.14)
out = os.path.join(FIGS, "fig_dynamics_signflip")
fig.savefig(out + ".pdf")
fig.savefig(out + ".png", dpi=200)
print("wrote", out + ".pdf/.png")
