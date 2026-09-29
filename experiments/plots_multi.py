"""Multi-seed, multi-dataset figures: per-dataset sign-flip curves with error bars,
and the cross-dataset generalization bar chart (the 'it holds on every modality'
figure). Reads every results root like stats.py."""

import _bootstrap  # noqa: F401

import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = _bootstrap.RESULTS
FIGS = os.path.join(R, "figs")
os.makedirs(FIGS, exist_ok=True)
ROOTS = {"cifar10": R, "fmnist": os.path.join(R, "fmnist"),
         "emnist": os.path.join(R, "kaggle", "emnist"),
         "edgeiiot": os.path.join(R, "kaggle", "edgeiiot")}
PAT = re.compile(r"robust_(\w+?)_(\w+?)_f(\d+)_c(\d+)_s(\d+)\.csv$")

C = {"blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100", "green": "#008300",
     "violet": "#4a3aa7", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
     "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}
AGG = {"mean": (C["blue"], "Mean"), "trimmed": (C["aqua"], "Trimmed mean"),
       "median": (C["yellow"], "Median"), "krum": (C["green"], "Multi-Krum"),
       "reputation": (C["violet"], "CB-SAFE+ (ours)")}
DS_TITLE = {"cifar10": "CIFAR-10", "fmnist": "FashionMNIST", "emnist": "EMNIST",
            "edgeiiot": "Edge-IIoTset"}

plt.rcParams.update({
    "font.size": 8, "font.family": "sans-serif",
    "axes.edgecolor": C["axis"], "axes.labelcolor": C["ink2"],
    "xtick.color": C["muted"], "ytick.color": C["muted"],
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": C["surface"],
    "savefig.bbox": "tight", "savefig.dpi": 300})
FIG_W = 3.45


def long() -> pd.DataFrame:
    rows = []
    for ds, root in ROOTS.items():
        if not os.path.isdir(root):
            continue
        for p in glob.glob(os.path.join(root, "robust_*.csv")):
            m = PAT.search(os.path.basename(p))
            if not m:
                continue
            df = pd.read_csv(p).tail(5)
            rows.append({"dataset": ds, "attack": m[1], "agg": m[2],
                         "f": int(m[3]) / 100, "c": int(m[4]), "seed": int(m[5]),
                         "acc": df["acc"].mean() * 100})
    return pd.DataFrame(rows)


def fig_signflip(data: pd.DataFrame, ds: str) -> None:
    d = data[(data.dataset == ds) & (data.attack == "signflip") & (data.c == 3)]
    if d.empty:
        return
    base = data[(data.dataset == ds) & (data.attack == "none")]["acc"].mean()
    fig, ax = plt.subplots(figsize=(FIG_W, 2.4))
    for agg, (color, label) in AGG.items():
        s = d[d["agg"] == agg].groupby("f")["acc"].agg(["mean", "std"]).reset_index()
        if s.empty:
            continue
        ax.errorbar(s["f"] * 100, s["mean"], yerr=s["std"].fillna(0), color=color,
                    lw=2, marker="o", ms=4, capsize=2, label=label)
    if not np.isnan(base):
        ax.axhline(base, color=C["muted"], lw=1, ls=(0, (4, 3)))
        ax.annotate("no-attack baseline", (d["f"].min() * 100, base),
                    textcoords="offset points", xytext=(0, 4), fontsize=6.5, color=C["muted"])
    ax.legend(fontsize=6, frameon=False, loc="best")
    n_seed = d["seed"].nunique()
    ax.set_title(f"{DS_TITLE[ds]}: sign-flip robustness (c=3, {n_seed} seed"
                 f"{'s' if n_seed > 1 else ''})", fontsize=8, color=C["ink"], loc="left")
    ax.set_xlabel("Malicious fraction f (%)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlim(left=0)
    fig.savefig(os.path.join(FIGS, f"fig_signflip_{ds}.pdf"))
    fig.savefig(os.path.join(FIGS, f"fig_signflip_{ds}.png"))
    plt.close(fig)


def fig_generalization(data: pd.DataFrame) -> None:
    """Grouped bars at f=0.3 sign-flip: CB-SAFE+ vs best-static vs worst-static,
    normalized to each dataset's clean baseline (so scales are comparable)."""
    order = [ds for ds in ["cifar10", "fmnist", "edgeiiot", "emnist"]
             if ds in data["dataset"].unique()]
    fig, ax = plt.subplots(figsize=(FIG_W, 2.5))
    x = np.arange(len(order))
    w = 0.26
    rep_v, best_v, worst_v = [], [], []
    for ds in order:
        d = data[(data.dataset == ds) & (data.attack == "signflip") & (data.c == 3)
                 & (data.f == 0.3)]
        base = data[(data.dataset == ds) & (data.attack == "none")]["acc"].mean()
        rep = d[d["agg"] == "reputation"]["acc"].mean()
        stat = d[d["agg"] != "reputation"].groupby("agg")["acc"].mean()
        rep_v.append(rep / base * 100)
        best_v.append(stat.max() / base * 100)
        worst_v.append(stat.min() / base * 100)
    ax.bar(x - w, worst_v, w, color=C["blue"], label="Worst static rule")
    ax.bar(x, best_v, w, color=C["green"], label="Best static rule")
    ax.bar(x + w, rep_v, w, color=C["violet"], label="CB-SAFE+ (ours)")
    ax.set_xticks(x, [DS_TITLE[d] for d in order], fontsize=7.5)
    ax.set_ylabel("Accuracy retained vs\nclean baseline (%)")
    ax.axhline(100, color=C["muted"], lw=0.8, ls=(0, (4, 3)))
    ax.legend(fontsize=6.5, frameon=False, loc="upper center", ncol=1)
    ax.set_title("Sign-flip at f=30%: CB-SAFE+ preserves most of the\n"
                 "clean accuracy where static rules collapse (all datasets)",
                 fontsize=7.5, color=C["ink"], loc="left")
    fig.savefig(os.path.join(FIGS, "fig_generalization.pdf"))
    fig.savefig(os.path.join(FIGS, "fig_generalization.png"))
    plt.close(fig)


if __name__ == "__main__":
    data = long()
    for ds in ROOTS:
        fig_signflip(data, ds)
    fig_generalization(data)
    print("multi-dataset figures written to", FIGS)
