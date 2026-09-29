"""Paper figures from results CSVs (IEEE single-column PDFs).

Follows the dataviz method: one axis per chart, categorical hues in fixed slot
order (never cycled), direct labels where series <= 4, thin marks, recessive
grid/axis chrome, values labeled where contrast is low.
"""

import _bootstrap  # noqa: F401

import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_style as st  # shared serif style
st.apply()

R = _bootstrap.RESULTS
FIGS = os.path.join(R, "figs")
os.makedirs(FIGS, exist_ok=True)

# Reference categorical palette (validated, fixed slot order)
C = {
    "blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100", "green": "#008300",
    "violet": "#4a3aa7", "red": "#e34948",
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb",
}
C["black"] = "#0b0b0b"
C["magenta"] = "#b0228c"
AGG_COLOR = {"mean": C["blue"], "trimmed": C["aqua"], "median": C["yellow"],
             "krum": C["green"], "bulyan": C["red"], "geomedian": C["violet"],
             "fedgt": C["black"], "reputation": C["violet"], "hybrid_ov4": C["magenta"]}
AGG_LABEL = {"mean": "Mean (no robustness)", "trimmed": "Trimmed mean",
             "median": "Median", "krum": "Multi-Krum", "bulyan": "Bulyan",
             "geomedian": "Geo-median/RFA", "fedgt": "FedGT [TIFS'25]",
             "reputation": "CB-SAFE+ (ours)", "hybrid_ov4": "CB-SAFE+ hybrid (ours)"}
# per-(attack,metric) series order; only methods with complete data are drawn
ACC_METHODS = ["mean", "trimmed", "median", "krum", "bulyan", "geomedian",
               "fedgt", "hybrid_ov4"]
ASR_METHODS = ["mean", "trimmed", "median", "krum", "reputation"]

FIG_W = st.COL_SINGLE  # IEEE single-column width (in), from the shared style


def fig_utility() -> None:
    upath = os.path.join(R, "utility_acc.csv")
    if not os.path.exists(upath):
        return
    acc = pd.read_csv(upath)
    fig, ax = plt.subplots(figsize=(FIG_W, 2.2))
    ax.plot(acc["round"] + 1, acc["acc"] * 100, color=C["blue"], lw=2)
    eq = os.path.join(R, "secure_equivalence.csv")
    if os.path.exists(eq):
        pass  # no in-axes title; details belong in the caption
    ax.set_xlabel("Round")
    ax.set_ylabel("Test accuracy (%)")
    fig.savefig(os.path.join(FIGS, "fig_utility.pdf"))
    plt.close(fig)


def fig_amortization() -> None:
    """Dot plot, log x: one-time setup traffic per KEM vs ONE round of masked
    updates. The point is that even HQC-256's setup is a fraction of a single
    round, so the code-based premium amortizes to noise."""
    opath = os.path.join(R, "overhead.csv")
    if not os.path.exists(opath):
        return
    df = pd.read_csv(opath)
    df = df[df.cluster_size == 3].set_index("kem")
    order = ["mlkem-512", "hqc-128", "mlkem-768", "hqc-192", "mlkem-1024", "hqc-256"]
    names = {"mlkem-512": "ML-KEM-512", "hqc-128": "HQC-128", "mlkem-768": "ML-KEM-768",
             "hqc-192": "HQC-192", "mlkem-1024": "ML-KEM-1024", "hqc-256": "HQC-256"}
    fig, ax = plt.subplots(figsize=(FIG_W, 2.4))
    one_round = (df["round_up_B"].iloc[0] + df["round_down_B"].iloc[0]) / 1024
    for yi, kem in enumerate(order):
        row = df.loc[kem]
        kib = (row.setup_up_B + row.setup_down_B) / 1024
        color = C["blue"] if kem.startswith("hqc") else C["aqua"]
        ax.plot([kib], [yi], "o", ms=6, color=color)
        ax.hlines(yi, 0.9, kib, color=color, lw=1, alpha=0.35)
        ax.annotate(f"{kib:,.1f} KiB", (kib, yi), textcoords="offset points",
                    xytext=(6, -2.5), fontsize=6.5, color=C["ink2"])
    ax.axvline(one_round, color=C["muted"], lw=1, ls=(0, (4, 3)))
    ax.annotate(f"one round of masked updates\n({one_round:,.0f} KiB, identical for all KEMs)",
                (one_round, 0.35), fontsize=6.5, color=C["muted"], ha="right",
                textcoords="offset points", xytext=(-6, 0))
    ax.set_yticks(range(len(order)), [names[k] for k in order], fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(0.9, one_round * 4)
    ax.set_xlabel("Traffic per client (KiB, log scale)")
    ax.grid(axis="y", visible=False)
    fig.savefig(os.path.join(FIGS, "fig_amortization.pdf"))
    plt.close(fig)


def _robust_runs(subdir: str = "") -> pd.DataFrame:
    base = os.path.join(R, subdir) if subdir else R
    rows = []
    for path in glob.glob(os.path.join(base, "robust_*.csv")):
        m = re.match(r"robust_(signflip|labelflip|backdoor|none)_(.+)_f(\d+)_c(\d+)_s(\d+)\.csv",
                     os.path.basename(path))
        if not m:
            continue
        if int(m[4]) != 3:  # c != 3 belongs to the c-dial figure, not these
            continue
        if m[2].startswith("fedgt"):  # skip COMP stand-in; use the real BCJR below
            continue
        df = pd.read_csv(path).tail(5)
        rows.append({"attack": m[1], "agg": m[2], "f": int(m[3]) / 100,
                     "acc": df["acc"].mean() * 100,
                     "asr": df["asr"].mean() * 100 if "asr" in df.columns else np.nan})
    # FedGT = its actual BCJR decoder (faithful reimplementation) at N=30
    ds = {"": "cifar10", "fmnist": "fmnist", os.path.join("kaggle", "emnist"): "emnist"}.get(subdir)
    if ds:
        for path in glob.glob(os.path.join(R, "fedgt_faithful",
                                           f"faithful_fedgt_{ds}_signflip_f*_N30_s*.csv")):
            if "_oneshot" in path:
                continue
            fm = re.search(r"_signflip_f(\d+)_N30", path)
            df = pd.read_csv(path).tail(5)
            rows.append({"attack": "signflip", "agg": "fedgt", "f": int(fm[1]) / 100,
                         "acc": df["acc"].mean() * 100, "asr": np.nan})
    return pd.DataFrame(rows)


def _acc_vs_f(subdir, attack, metric, methods, ylabel, outname, baseline=None):
    data = _robust_runs(subdir)
    sub = data[(data.attack == attack) & data[metric].notna()] if not data.empty else data
    if sub is None or sub.empty:
        return
    sub = sub.groupby(["agg", "f"], as_index=False)[metric].mean()
    fig, ax = plt.subplots(figsize=(FIG_W, 2.3))
    for agg in methods:
        s = sub[sub["agg"] == agg].sort_values("f")
        if len(s) < 3:
            continue
        sty = st.style(agg)
        ax.plot(s["f"] * 100, s[metric], color=sty["color"], ls=sty["ls"], lw=sty["lw"],
                marker=sty["marker"], label=sty["label"], zorder=6 if sty.get("ours") else 3)
    if baseline is not None:
        ax.axhline(baseline, color="#9e9e9e", lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.annotate("no-attack baseline", (2, baseline), textcoords="offset points",
                    xytext=(0, 3), fontsize=6, color="#6b6b6b")
    # legend below the axes so it never covers data (§5); no in-axes title (§2)
    ax.legend(fontsize=6, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=3, handlelength=2.2)
    ax.set_xlabel("malicious fraction f (%)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)
    fig.savefig(os.path.join(FIGS, outname))
    plt.close(fig)


def fig_robustness() -> None:
    # Fig. 6: sign-flip accuracy on FashionMNIST (a clear-margin win, not the CIFAR tie).
    _acc_vs_f("fmnist", "signflip", "acc", ACC_METHODS, "Test accuracy (%)",
              "fig_signflip_acc.pdf", baseline=88.5)
    # Fig. 7: backdoor ASR on CIFAR-10 (where the backdoor sweep was run).
    _acc_vs_f("", "backdoor", "asr", ASR_METHODS, "Attack success rate (%)",
              "fig_backdoor_asr.pdf")


def fig_tradeoff() -> None:
    """Clean-cluster probability (1-f)^c: the privacy-robustness tension curve."""
    f = np.linspace(0, 0.4, 200)
    fig, ax = plt.subplots(figsize=(FIG_W, 2.3))
    for c, color in [(1, C["muted"]), (3, C["blue"]), (5, C["aqua"]), (10, C["yellow"])]:
        ax.plot(f * 100, (1 - f) ** c * 100, lw=2, color=color)
        ax.annotate(f"c={c}" + (" (no privacy)" if c == 1 else ""),
                    (f[-1] * 100, ((1 - f[-1]) ** c) * 100),
                    textcoords="offset points", xytext=(4, 0), fontsize=6.5,
                    color=color, va="center")
    ax.set_xlabel("Malicious fraction f (%)")
    ax.set_ylabel("P(cluster sum clean) (%)")
    ax.set_xlim(0, 46)
    fig.savefig(os.path.join(FIGS, "fig_tradeoff.pdf"))
    plt.close(fig)


def fig_cdial() -> None:
    """Empirical privacy-robustness dial: median-rule accuracy vs f for c in
    {1,3,5} under sign-flip. The breakdown frontier shifts left as c grows,
    tracking (1-f)^c."""
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
    fig, ax = plt.subplots(figsize=(FIG_W, 2.4))
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
    fig.savefig(os.path.join(FIGS, "fig_cdial.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_utility()
    fig_amortization()
    fig_robustness()
    fig_tradeoff()
    print("figures written to", FIGS)
