"""CB-SAFE full architecture figure (one training round, end to end), drawn as a
clean rectangular loop. Renders results/figs/fig_architecture.{pdf,png}."""

import _bootstrap  # noqa: F401

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

C = {"blue": "#2a78d6", "aqua": "#1baf7a", "violet": "#4a3aa7", "red": "#e34948",
     "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
     "paleB": "#eaf2fc", "paleG": "#e9f7f1", "paleV": "#efecf9", "paleR": "#fbeeee"}
FONT = "DejaVu Sans"

fig, ax = plt.subplots(figsize=(7.16, 3.9))
ax.set_xlim(0, 11.4); ax.set_ylim(0, 9); ax.axis("off")
fig.patch.set_facecolor("white")

COLX = [1.15, 3.35, 5.55, 7.75, 9.95]       # 5 column centers
R1, R2 = 5.6, 2.7                             # row baselines (box bottoms)
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


# --- setup ribbon ---
ax.add_patch(FancyBboxPatch((0.25, 7.45), 9.9, 0.9, boxstyle="round,pad=0.02,rounding_size=0.08",
                            linewidth=1.15, edgecolor=C["blue"], facecolor=C["paleB"], zorder=3))
ax.text(5.2, 7.9, "ONE-TIME SETUP   ·   code-based HQC key establishment over client pairs  →  pairwise secrets   (reused every round; KEM cost paid once)",
        ha="center", va="center", fontsize=6.7, color=C["blue"], family=FONT)

# --- bands (drawn first, behind boxes) ---
band(2.40, 5.42, 6.55, 7.05, "Confidentiality layer (code-based PQC)", C["blue"])
band(4.60, 2.50, 10.95, 4.15, "Byzantine-robustness stage (CB-SAFE+)", C["violet"])

# --- row 1: forward pipeline (left -> right) ---
box(COLX[0], R1, "clients 1..N\n(private data)\nlocal train → Δᵢ", "#ffffff", C["ink2"])
box(COLX[1], R1, "add HQC-derived\npairwise masks\n(cancel in cluster)", C["paleB"], C["blue"])
box(COLX[2], R1, "cluster (size c)\nwithin-cluster\nsecure aggregation", C["paleB"], C["blue"])
box(COLX[3], R1, "server sees ONLY\ncluster sums S₁..Sₖ\n(anonymity set c)", "#ffffff", C["ink2"])
box(COLX[4], R1, "flag clusters\nloss-probe /\nconsensus (no-trust)", C["paleV"], C["violet"])

arrow(COLX[0] + BW / 2, R1 + BH / 2, COLX[1] - BW / 2, R1 + BH / 2)
arrow(COLX[1] + BW / 2, R1 + BH / 2, COLX[2] - BW / 2, R1 + BH / 2)
arrow(COLX[2] + BW / 2, R1 + BH / 2, COLX[3] - BW / 2, R1 + BH / 2, label="k sums", dy=-0.55)
arrow(COLX[3] + BW / 2, R1 + BH / 2, COLX[4] - BW / 2, R1 + BH / 2)

# --- turn down at right ---
arrow(COLX[4], R1, COLX[4], R2 + BH, lw=1.6)

# --- row 2: robustness stage flowing back (right -> left) ---
box(COLX[4], R2, "accumulate\nper-client suspicion\n(re-randomized rounds)", C["paleV"], C["violet"])
box(COLX[3], R2, "adaptive exclusion\nof identified\nattackers", C["paleR"], C["red"])
box(COLX[2], R2, "robust aggregate\nover clean\ncluster sums", C["paleG"], C["aqua"])
box(COLX[1], R2, "global model\nupdate", C["paleG"], C["aqua"], bold=True)
box(COLX[0], R2, "broadcast to\nclients\n(next round)", "#ffffff", C["ink2"])

arrow(COLX[4] - BW / 2, R2 + BH / 2, COLX[3] + BW / 2, R2 + BH / 2)
arrow(COLX[3] - BW / 2, R2 + BH / 2, COLX[2] + BW / 2, R2 + BH / 2)
arrow(COLX[2] - BW / 2, R2 + BH / 2, COLX[1] + BW / 2, R2 + BH / 2)
arrow(COLX[1] - BW / 2, R2 + BH / 2, COLX[0] + BW / 2, R2 + BH / 2)

# --- close the loop: broadcast up to clients ---
arrow(COLX[0], R2 + BH, COLX[0], R1, lw=1.6, ls=(0, (5, 3)), color=C["muted"])

ax.text(5.7, 8.75, "CB-SAFE: code-based post-quantum secure aggregation with Byzantine-robust identification",
        ha="center", fontsize=8.4, fontweight="bold", color=C["ink"], family=FONT)

out = os.path.join(_bootstrap.RESULTS, "figs", "fig_architecture")
fig.savefig(out + ".pdf", bbox_inches="tight", dpi=300)
fig.savefig(out + ".png", bbox_inches="tight", dpi=200)
print("wrote", out)
