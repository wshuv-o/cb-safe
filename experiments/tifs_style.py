import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL_SINGLE = 3.5
COL_DOUBLE = 7.16

_OK = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
       "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
       "purple": "#CC79A7", "grey": "#7f7f7f"}

STYLE = {
    "mean":       dict(color=_OK["grey"],       marker="x", ls=(0, (1, 1)),  lw=1.1, label="Mean (no robustness)"),
    "trimmed":    dict(color=_OK["orange"],     marker="s", ls=(0, (4, 2)),  lw=1.1, label="Trimmed mean"),
    "median":     dict(color=_OK["sky"],        marker="^", ls=(0, (4, 2)),  lw=1.1, label="Median"),
    "krum":       dict(color=_OK["green"],      marker="D", ls=(0, (3, 1, 1, 1)), lw=1.1, label="Multi-Krum"),
    "bulyan":     dict(color=_OK["blue"],       marker="v", ls=(0, (3, 1, 1, 1)), lw=1.1, label="Bulyan"),
    "geomedian":  dict(color=_OK["purple"],     marker="P", ls=(0, (3, 1, 1, 1)), lw=1.1, label="Geo-median/RFA"),
    "fltrust":    dict(color=_OK["yellow"],     marker="*", ls=(0, (4, 2)),  lw=1.1, label="FLTrust"),
    "fedgt":      dict(color=_OK["vermillion"], marker="o", ls="-",          lw=1.6, label="FedGT [18]"),
    "hybrid_ov4": dict(color=_OK["black"],      marker="o", ls="-",          lw=2.0, label="CB-SAFE+ (ours)", ours=True),
    "reputation":    dict(color=_OK["black"],   marker="o", ls=(0, (4, 2)),  lw=1.3, label="CB-SAFE+ (ov1)"),
    "reputation_ov4":dict(color=_OK["blue"],    marker="s", ls=(0, (4, 2)),  lw=1.3, label="CB-SAFE+ (ov4)"),
}


def apply():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Latin Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": "#333333", "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5,
        "xtick.color": "#333333", "ytick.color": "#333333",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "lines.markersize": 3.5,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def style(agg):
    return STYLE.get(agg, dict(color=_OK["grey"], marker=".", ls="-", lw=1.0, label=agg))
