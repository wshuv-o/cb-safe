import _bootstrap  # noqa: F401

import argparse
import csv
import glob
import os
import re

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import tifs_style as st  # noqa: E402

R = _bootstrap.RESULTS
FIGS = os.path.join(R, "figs")


def fig_architecture():
    C = {"blue": "#2a78d6", "aqua": "#1baf7a", "violet": "#4a3aa7", "red": "#e34948",
         "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
         "paleB": "#eaf2fc", "paleG": "#e9f7f1", "paleV": "#efecf9", "paleR": "#fbeeee"}
    FONT = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(7.16, 3.9))
    ax.set_xlim(0, 11.4); ax.set_ylim(0, 9); ax.axis("off")
    fig.patch.set_facecolor("white")
    COLX = [1.15, 3.35, 5.55, 7.75, 9.95]
    R1, R2 = 5.6, 2.7
    BW, BH = 1.9, 1.35

    def box(cx, y, text, fill, edge, size=6.8, bold=False, tcolor=None, w=BW, h=BH):
        x = cx - w / 2
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.07",
                                    linewidth=1.15, edgecolor=edge, facecolor=fill, zorder=3))
        ax.text(cx, y + h / 2, text, ha="center", va="center", fontsize=size,
                fontweight="bold" if bold else "normal", color=tcolor or C["ink"],
                zorder=4, family=FONT)

    def band(x0, y0, x1, y1, label, color):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                    boxstyle="round,pad=0.02,rounding_size=0.1",
                                    linewidth=1.2, edgecolor=color, facecolor="none",
                                    linestyle=(0, (5, 3)), zorder=2))
        ax.text(x0 + 0.15, y1 + 0.06, label, ha="left", va="bottom", fontsize=6.5,
                color=color, fontweight="bold", family=FONT)

    def arrow(x1, y1, x2, y2, label=None, lw=1.5, color=C["ink2"], ls="-", dy=0.24):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                     linewidth=lw, color=color, linestyle=ls, zorder=3,
                                     shrinkA=2, shrinkB=2))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, label, ha="center", va="bottom",
                    fontsize=6.3, color=C["muted"], family=FONT)

    ax.add_patch(FancyBboxPatch((0.25, 7.45), 9.9, 0.9, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.15, edgecolor=C["blue"], facecolor=C["paleB"], zorder=3))
    ax.text(5.2, 7.9, "ONE-TIME SETUP   ·   code-based HQC key establishment over client pairs  →  pairwise secrets   (reused every round; KEM cost paid once)",
            ha="center", va="center", fontsize=6.7, color=C["blue"], family=FONT)
    band(2.40, 5.42, 6.55, 7.05, "Confidentiality layer (code-based PQC)", C["blue"])
    band(4.60, 2.50, 10.95, 4.15, "Byzantine-robustness stage (CB-SAFE+)", C["violet"])
    box(COLX[0], R1, "clients 1..N\n(private data)\nlocal train → Δᵢ", "#ffffff", C["ink2"])
    box(COLX[1], R1, "add HQC-derived\npairwise masks\n(cancel in cluster)", C["paleB"], C["blue"])
    box(COLX[2], R1, "cluster (size c)\nwithin-cluster\nsecure aggregation", C["paleB"], C["blue"])
    box(COLX[3], R1, "server sees ONLY\ncluster sums S₁..Sₖ\n(anonymity set c)", "#ffffff", C["ink2"])
    box(COLX[4], R1, "flag clusters\nloss-probe /\nconsensus (no-trust)", C["paleV"], C["violet"])
    arrow(COLX[0] + BW / 2, R1 + BH / 2, COLX[1] - BW / 2, R1 + BH / 2)
    arrow(COLX[1] + BW / 2, R1 + BH / 2, COLX[2] - BW / 2, R1 + BH / 2)
    arrow(COLX[2] + BW / 2, R1 + BH / 2, COLX[3] - BW / 2, R1 + BH / 2, label="k sums", dy=-0.55)
    arrow(COLX[3] + BW / 2, R1 + BH / 2, COLX[4] - BW / 2, R1 + BH / 2)
    arrow(COLX[4], R1, COLX[4], R2 + BH, lw=1.6)
    box(COLX[4], R2, "accumulate\nper-client suspicion\n(re-randomized rounds)", C["paleV"], C["violet"])
    box(COLX[3], R2, "adaptive exclusion\nof identified\nattackers", C["paleR"], C["red"])
    box(COLX[2], R2, "robust aggregate\nover clean\ncluster sums", C["paleG"], C["aqua"])
    box(COLX[1], R2, "global model\nupdate", C["paleG"], C["aqua"], bold=True)
    box(COLX[0], R2, "broadcast to\nclients\n(next round)", "#ffffff", C["ink2"])
    arrow(COLX[4] - BW / 2, R2 + BH / 2, COLX[3] + BW / 2, R2 + BH / 2)
    arrow(COLX[3] - BW / 2, R2 + BH / 2, COLX[2] + BW / 2, R2 + BH / 2)
    arrow(COLX[2] - BW / 2, R2 + BH / 2, COLX[1] + BW / 2, R2 + BH / 2)
    arrow(COLX[1] - BW / 2, R2 + BH / 2, COLX[0] + BW / 2, R2 + BH / 2)
    arrow(COLX[0], R2 + BH, COLX[0], R1, lw=1.6, ls=(0, (5, 3)), color=C["muted"])
    ax.text(5.7, 8.75, "CB-SAFE: code-based post-quantum secure aggregation with Byzantine-robust identification",
            ha="center", fontsize=8.4, fontweight="bold", color=C["ink"], family=FONT)
    out = os.path.join(FIGS, "fig_architecture")
    fig.savefig(out + ".pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out + ".png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", out)


def fig_dynamics_signflip():
    ROWS = [("CIFAR-10", "", 10.0, 59.0),
            ("FashionMNIST", "fmnist", 10.0, 88.5),
            ("EMNIST", os.path.join("kaggle", "emnist"), 2.1, 86.0),
            ("Edge-IIoTset", os.path.join("kaggle", "edgeiiot"), 6.7, 63.7)]
    FS = [10, 20, 30]
    METHODS = ["mean", "trimmed", "median", "krum", "bulyan", "geomedian", "fedgt", "hybrid_ov4"]
    BAND = {"hybrid_ov4", "fedgt"}
    W = 5
    FAITHFUL = os.path.join(R, "fedgt_faithful")
    DS = {"": "cifar10", "fmnist": "fmnist", os.path.join("kaggle", "emnist"): "emnist"}

    def series(subdir, agg, f):
        if agg == "fedgt":
            ds = DS.get(subdir)
            paths = sorted(glob.glob(os.path.join(
                FAITHFUL, f"faithful_fedgt_{ds}_signflip_f{f:02d}_N30_s*.csv"))) if ds else []
            curves = []
            for p in paths:
                if "_oneshot" in p:
                    continue
                curves.append(pd.read_csv(p)["acc"].to_numpy() * 100)
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
            curves.append(pd.read_csv(p)["acc"].to_numpy() * 100)
        if not curves:
            return None
        n = min(len(c) for c in curves)
        M = np.vstack([c[:n] for c in curves])
        return np.arange(1, n + 1), M.mean(0), M.std(0)

    fig, axes = plt.subplots(4, 3, figsize=(st.COL_DOUBLE, 6.9), sharex="col")
    handles_labels = {}
    for r, (dsname, subdir, chance, base) in enumerate(ROWS):
        ymin = min(chance - 3, 5)
        ymax = base + 5
        for c, f in enumerate(FS):
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
                ax.set_title(f"$f{{=}}{f}\\%$")
            if r == 3:
                ax.set_xlabel("communication round")
            if c == 0:
                ax.set_ylabel(f"{dsname}\ntest accuracy (\\%)")
        axes[0][0].annotate("exclusions begin", xy=(W, axes[0][0].get_ylim()[0] + 6),
                            xytext=(W + 1.5, axes[0][0].get_ylim()[0] + 3),
                            fontsize=6, color="#6b6b6b")
    labels = ["Mean (no robustness)", "Trimmed mean", "Median", "Multi-Krum",
              "Bulyan", "Geo-median/RFA", "FedGT [18]", "CB-SAFE+ (ours)"]
    hs = [handles_labels[l] for l in labels if l in handles_labels]
    ls = [l for l in labels if l in handles_labels]
    fig.legend(hs, ls, ncol=4, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 1.045), fontsize=7, handlelength=2.4)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.subplots_adjust(hspace=0.18, wspace=0.14)
    out = os.path.join(FIGS, "fig_dynamics_signflip")
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=200)
    plt.close(fig)
    print("wrote", out)


def fig_convergence_n500():
    DIR = os.path.join(R, "scale50")
    SEEDS = [0, 3, 2]
    METHODS = [("hybrid_ov4", "hybrid_ov4"), ("fltrust", "fltrust"),
               ("median", "median"), ("geomedian", "geomedian"), ("krum", "krum"),
               ("bulyan", "bulyan"), ("trimmed", "trimmed"), ("mean", "mean")]
    CHANCE = 0.10

    def load(stem, seed):
        p = os.path.join(DIR, f"robust_signflip_{stem}_N500_f20_c3_s{seed}.csv")
        if not os.path.exists(p):
            return None, None
        r = list(csv.DictReader(open(p)))
        return (np.array([int(x["round"]) for x in r]),
                np.array([float(x["acc"]) for x in r]))

    fig, axes = plt.subplots(1, 3, figsize=(st.COL_DOUBLE, 2.55), sharex=True, sharey=True)
    handles, labels = [], []
    for ax, seed in zip(axes, SEEDS):
        ax.axhline(CHANCE, color="#c9c9c9", lw=0.7, ls=(0, (2, 2)), zorder=1)
        for stem, key in METHODS:
            rounds, acc = load(stem, seed)
            if rounds is None:
                continue
            sty = st.style(key)
            ln, = ax.plot(rounds, acc, color=sty["color"], marker=sty["marker"], ls=sty["ls"],
                          lw=sty["lw"], markersize=3.0, markevery=8, zorder=(5 if sty.get("ours") else 3))
            if not labels or sty["label"] not in labels:
                handles.append(ln); labels.append(sty["label"])
        ax.set_xlim(1, 50); ax.set_ylim(0.05, 0.74)
        ax.set_xticks([1, 10, 20, 30, 40, 50]); ax.set_yticks([0.1, 0.3, 0.5, 0.7])
        ax.set_xlabel("Communication round")
        idx = SEEDS.index(seed)
        ax.text(0.03, 0.97, f"({'abc'[idx]}) seed {idx}",
                transform=ax.transAxes, va="top", ha="left", fontsize=7.5)
    axes[0].set_ylabel("Test accuracy")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.13),
               ncol=5, frameon=False, columnspacing=1.3, handlelength=2.0, fontsize=6.6)
    out = os.path.join(FIGS, "fig_convergence_allmethods_n500")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_backdoor_asr():
    methods = ["mean", "trimmed", "median", "krum", "reputation"]
    rows = []
    for path in glob.glob(os.path.join(R, "robust_backdoor_*_c3_s*.csv")):
        m = re.match(r"robust_backdoor_(.+)_f(\d+)_c(\d+)_s(\d+)\.csv", os.path.basename(path))
        if not m or int(m[3]) != 3:
            continue
        df = pd.read_csv(path).tail(5)
        if "asr" not in df.columns:
            continue
        rows.append({"agg": m[1], "f": int(m[2]) / 100, "asr": df["asr"].mean() * 100})
    if not rows:
        return
    data = pd.DataFrame(rows).groupby(["agg", "f"], as_index=False)["asr"].mean()
    fig, ax = plt.subplots(figsize=(st.COL_SINGLE, 2.3))
    for agg in methods:
        s = data[data["agg"] == agg].sort_values("f")
        if len(s) < 3:
            continue
        sty = st.style(agg)
        ax.plot(s["f"] * 100, s["asr"], color=sty["color"], ls=sty["ls"], lw=sty["lw"],
                marker=sty["marker"], label=sty["label"], zorder=6 if sty.get("ours") else 3)
    ax.legend(fontsize=6, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=3, handlelength=2.2)
    ax.set_xlabel("malicious fraction f (%)")
    ax.set_ylabel("Attack success rate (%)")
    ax.set_xlim(left=0)
    out = os.path.join(FIGS, "fig_backdoor_asr.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def fig_cdial():
    C = {"blue": "#2a78d6", "aqua": "#1baf7a", "muted": "#898781"}
    rows = []
    for path in glob.glob(os.path.join(R, "robust_signflip_median_f*_c*_s0.csv")):
        m = re.match(r"robust_signflip_median_f(\d+)_c(\d+)_s0\.csv", os.path.basename(path))
        if not m:
            continue
        tail = pd.read_csv(path).tail(5)
        rows.append({"f": int(m[1]) / 100, "c": int(m[2]), "acc": tail["acc"].mean() * 100})
    if not rows:
        return
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(st.COL_SINGLE, 2.4))
    upath = os.path.join(R, "utility_acc.csv")
    if os.path.exists(upath):
        base = pd.read_csv(upath).tail(5)["acc"].mean() * 100
        ax.axhline(base, color=C["muted"], lw=1, ls=(0, (4, 3)))
        ax.annotate("no-attack baseline", (6, base), textcoords="offset points",
                    xytext=(0, 3), fontsize=6.5, color=C["muted"])
    for c, color, label in [(1, C["muted"], "c=1 (no privacy)"),
                            (3, C["blue"], "c=3"), (5, C["aqua"], "c=5")]:
        s = data[data["c"] == c].sort_values("f")
        if s.empty:
            continue
        ax.plot(s["f"] * 100, s["acc"], color=color, lw=2, marker="o", ms=4, label=label)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    ax.set_xlabel("Malicious fraction f (%)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlim(left=0)
    out = os.path.join(FIGS, "fig_cdial.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def fig_suspicion():
    FS = [10, 20, 30]
    Cc = 3
    W = 5
    MAL = "#D55E00"
    HON = "#0072B2"

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
        ph = 1 - (1 - f / 100) ** (Cc - 1)
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
        ax.set_xlabel("communication round")
        if j == 0:
            ax.set_ylabel("flagged fraction $s_i^{\\,r}/r$")
        ax.text(0.02, 0.94, f"$f{{=}}{f}\\%$", transform=ax.transAxes, fontsize=8, va="top")
    axes[0].annotate("warmup", (W / 2, 0.04), ha="center", fontsize=6, color="#8a8a8a")
    hh, ll = axes[0].get_legend_handles_labels()
    fig.legend(hh, ll, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.10), fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.subplots_adjust(wspace=0.08)
    out = os.path.join(FIGS, "fig_suspicion")
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=200)
    plt.close(fig)
    print("wrote", out)


FIGURES = {
    "architecture": fig_architecture,
    "dynamics": fig_dynamics_signflip,
    "convergence": fig_convergence_n500,
    "backdoor": fig_backdoor_asr,
    "cdial": fig_cdial,
    "suspicion": fig_suspicion,
}


def main():
    ap = argparse.ArgumentParser(description="Regenerate paper figures from result CSVs.")
    ap.add_argument("names", nargs="*", help="figure names or 'all' (default)")
    args = ap.parse_args()
    os.makedirs(FIGS, exist_ok=True)
    st.apply()
    names = args.names or ["all"]
    selected = list(FIGURES) if "all" in names else names
    for n in selected:
        if n not in FIGURES:
            print("unknown figure:", n, "-- choices:", ", ".join(FIGURES))
            continue
        FIGURES[n]()


if __name__ == "__main__":
    main()
