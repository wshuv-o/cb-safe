"""Non-IID partition figure: per-client class-sample counts under Dirichlet(alpha).
Grayscale (black-first) heatmap, serif/Computer-Modern font to match the paper.
Writes results/figs/fig_dirichlet.pdf."""

import _bootstrap  # noqa: F401

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.federated.data import dirichlet_partition, load_dataset  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.dpi": 200,
})


def main():
    train, _ = load_dataset("cifar10")
    labels = np.array(train.targets)
    parts = dirichlet_partition(labels, 30, alpha=0.5, seed=0)
    n_classes = 10
    M = np.zeros((30, n_classes), dtype=int)
    for ci, idx in enumerate(parts):
        for c in range(n_classes):
            M[ci, c] = int(np.sum(labels[idx] == c))

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    im = ax.imshow(M, aspect="auto", cmap="Greys", vmin=0)
    ax.set_xlabel("class label")
    ax.set_ylabel("client index")
    ax.set_xticks(range(0, n_classes, 2))
    ax.set_yticks(range(0, 30, 5))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("samples", fontsize=7)
    cb.ax.tick_params(labelsize=6, width=0.6)
    cb.outline.set_linewidth(0.6)
    fig.tight_layout(pad=0.3)
    out = os.path.join(_bootstrap.RESULTS, "figs", "fig_dirichlet.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out, "| client totals min/max:", M.sum(1).min(), M.sum(1).max())


if __name__ == "__main__":
    main()
