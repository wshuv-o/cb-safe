import _bootstrap  # noqa: F401

import glob
import os
import re

import numpy as np
import pandas as pd

import plot_style as st  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

st.apply()
R = _bootstrap.RESULTS
FIGS = os.path.join(R, "figs")
os.makedirs(FIGS, exist_ok=True)

ROWS = [("CIFAR-10", "", 59.0),
        ("FashionMNIST", "fmnist", 88.5),
        ("EMNIST", os.path.join("kaggle", "emnist"), 86.0),
        ("Edge-IIoTset", os.path.join("kaggle", "edgeiiot"), 63.7)]
FS = [0.1, 0.2, 0.3]
METHODS = ["mean", "trimmed", "median", "krum", "bulyan", "geomedian", "fltrust", "hybrid_ov4"]
_FAITHFUL = os.path.join(R, "fedgt_faithful")
_DS = {"": "cifar10", "fmnist": "fmnist", os.path.join("kaggle", "emnist"): "emnist",
       os.path.join("kaggle", "edgeiiot"): "edgeiiot"}


def acc_at(subdir, agg, f):
    """(mean, std) last-5-of-first-50-round acc% over seeds at fraction f, or (None, None)."""
    if agg == "fedgt":
        ds = _DS.get(subdir)
        paths = [p for p in glob.glob(os.path.join(
            _FAITHFUL, f"faithful_fedgt_{ds}_signflip_f{int(f*100):02d}_N30_s*.csv"))
            if "_oneshot" not in p] if ds else []
    else:
        base = os.path.join(R, subdir) if subdir else R
        paths = [p for p in glob.glob(os.path.join(base, f"robust_signflip_{agg}_f{int(f*100):02d}_c3_s*.csv"))
                 if re.sub(r"_f\d+_c3_s\d+\.csv$", "", os.path.basename(p)).replace("robust_signflip_", "") == agg]
    vs = []
    for p in paths:
        d = pd.read_csv(p)
        if len(d) >= 50:
            vs.append(d.iloc[:50].tail(5)["acc"].mean() * 100)
    return (float(np.mean(vs)), float(np.std(vs))) if vs else (None, None)


ORACLE_STYLE = dict(color="#000000", marker=None, ls=(0, (1, 1)), lw=1.2, label="Oracle (perfect detection)")

fig, axes = plt.subplots(1, 4, figsize=(st.COL_DOUBLE, 2.3))
for ax, (name, sub, clean) in zip(axes, ROWS):
    ax.axhline(clean, ls=":", color="#bbbbbb", lw=0.8, zorder=0)
    order = list(METHODS)
    if _DS.get(sub):
        order.insert(0, "fedgt")
    if acc_at(sub, "oracle", 0.1)[0] is not None:
        order.append("oracle")
    for agg in order:
        pts = [acc_at(sub, agg, f) for f in FS]
        xs = [f for f, (m, _) in zip(FS, pts) if m is not None]
        ys = [m for (m, _) in pts if m is not None]
        es = [s for (m, s) in pts if m is not None]
        if not ys:
            continue
        sty = ORACLE_STYLE if agg == "oracle" else st.style(agg)
        ax.errorbar(xs, ys, yerr=es, color=sty["color"], marker=sty.get("marker"),
                    ls=sty["ls"], lw=sty["lw"], ms=3, capsize=1.5, label=sty["label"])
    ax.set_xlabel(r"malicious fraction $f$")
    ax.set_xticks(FS)
    ax.set_ylim(0, 100)
    ax.set_title(name, fontsize=8)
axes[0].set_ylabel(r"test accuracy (\%)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=6, frameon=False,
           bbox_to_anchor=(0.5, -0.10))
fig.tight_layout()
out = os.path.join(FIGS, "fig_acc_vs_f")
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
print("wrote", out + ".pdf/.png")
