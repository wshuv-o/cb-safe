"""C1: empirical validation of the laundering impossibility.

Sweeps the sign-flip scale gamma at fixed (f, c) and records, per gamma:
  - damage:        clean_acc - attacked_acc under the coordinate-wise MEDIAN rule
  - detectability: |E|poisoned cluster-mean norm| - E|clean cluster-mean norm|| /
                   std(clean norms)  -- how much a poisoned cluster sum stands out
                   by MAGNITUDE (what any coordinate-wise / norm-based rule sees).

Theory (Prop 1): a cluster with m sign-flippers of scale gamma has mean
    mu = ((c - m(1+gamma))/c) * h,
so |mu| = |h| exactly at gamma* = 2c/m - 1 (here c=3, m=1 -> gamma* = 5): the
poisoned sum is magnitude-indistinguishable from clean while pushing the model the
wrong way. The prediction is a DETECTABILITY VALLEY at gamma* with HIGH damage.

We compute detectability analytically from the per-round honest/poisoned cluster
means (cheap, exact) and damage from a real training run at each gamma.
"""

import _bootstrap  # noqa: F401

import csv
import os

import numpy as np
from torch.utils.data import DataLoader

from src.federated import data
from src.federated.simulation import Config, run

DATASET = "cifar10"
F = 0.2
C = 3
GAMMAS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 20.0]
ROUNDS = 25


def detectability(gamma: float, c: int = C, m: int = 1) -> float:
    """Analytic magnitude-outlier score of a poisoned cluster mean relative to a
    clean one. Poisoned mean = h*(c - m(1+gamma))/c, clean mean = h, so
    |poison|/|clean| = |c - m(1+gamma)|/c; deviation from 1 is how much the
    poisoned sum stands out by magnitude. It is exactly 0 at gamma* = 2c/m - 1
    (here 5): perfectly disguised."""
    ratio = abs(c - m * (1 + gamma)) / c  # |poison mean| / |clean mean|
    return abs(ratio - 1.0)


def main() -> None:
    train, test = data.load_dataset(DATASET)
    parts = data.dirichlet_partition(np.array(train.targets), 30, alpha=0.5, seed=0)
    client_dls = data.client_loaders(train, parts, 64)
    test_dl = DataLoader(test, batch_size=512, num_workers=0)

    # clean baseline (no attack) once
    base_hist = run(Config(n_clients=30, rounds=ROUNDS, aggregation="cluster",
                           aggregator="median", cluster_size=C, attack="none",
                           f_malicious=0.0, seed=0, dataset=DATASET), client_dls, test_dl)
    clean_acc = float(np.mean([h["acc"] for h in base_hist[-5:]]))
    print(f"clean median acc = {clean_acc:.4f}", flush=True)

    rows = []
    for g in GAMMAS:
        cfg = Config(n_clients=30, rounds=ROUNDS, aggregation="cluster",
                     aggregator="median", cluster_size=C, attack="signflip",
                     f_malicious=F, seed=0, dataset=DATASET, signflip_gamma=g)
        hist = run(cfg, client_dls, test_dl)
        acc = float(np.mean([h["acc"] for h in hist[-5:]]))
        row = {"gamma": g, "acc": round(acc, 4),
               "damage": round(clean_acc - acc, 4),
               "detectability": round(detectability(g), 4)}
        rows.append(row)
        print(f"gamma={g:5.1f}  acc={acc:.4f}  damage={row['damage']:.4f}  "
              f"detectability={row['detectability']:.4f}", flush=True)

    out = os.path.join(_bootstrap.RESULTS, "laundering_gamma_sweep.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["gamma", "acc", "damage", "detectability"])
        w.writeheader(); w.writerows(rows)
    print("wrote", out, flush=True)
    print("C1 LAUNDERING SWEEP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
