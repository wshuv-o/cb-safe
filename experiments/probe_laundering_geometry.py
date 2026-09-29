"""Measure the laundering geometry on real FashionMNIST cluster means.
For c=3, m=1 and each amplification gamma, form the poisoned cluster mean
mu = (h_a + h_b - gamma*h_k)/c from real client updates and record
|mu|/|mu_clean| and cos(mu, mu_clean). Validates Proposition 1 (gamma*=2c/m-1=5).
Writes results/laundering_gamma_geom.csv."""

import _bootstrap  # noqa: F401

import csv
import os

import numpy as np
import torch

from src.federated.simulation import Config, local_train
from src.federated import data, models
from src.federated.models import flat_params, make_model

GAMMAS = [1, 2, 3, 5, 8, 12, 20]
ROUNDS = 3
C, M = 3, 1

cfg = Config(n_clients=30, rounds=1, aggregation="cluster", aggregator="median",
             cluster_size=C, attack="signflip", f_malicious=0.0, seed=0,
             dataset="fmnist", n_classes=models.n_classes_of("fmnist"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train, _ = data.load_dataset("fmnist")
parts = data.dirichlet_partition(np.array(train.targets), 30, cfg.alpha, cfg.seed)
loaders = data.client_loaders(train, parts, cfg.batch_size)
global_flat = flat_params(make_model("fmnist"))

rec = {g: {"ratio": [], "cos": []} for g in GAMMAS}
for r in range(ROUNDS):
    D = np.stack([local_train(global_flat, loaders[i], cfg, device, malicious=False)
                  for i in range(30)])
    for k in range(10):                      # 10 clusters of 3
        idx = [3 * k, 3 * k + 1, 3 * k + 2]
        clean = D[idx].mean(0)
        cn = np.linalg.norm(clean)
        for mal in idx:                      # rotate which member is the attacker
            for g in GAMMAS:
                pois = np.stack([(-g * D[i] if i == mal else D[i]) for i in idx]).mean(0)
                pn = np.linalg.norm(pois)
                if cn > 0 and pn > 0:
                    rec[g]["ratio"].append(pn / cn)
                    rec[g]["cos"].append(float(np.dot(pois, clean) / (pn * cn)))
    global_flat = global_flat + D.mean(0)     # FedAvg to progress training realistically

out = os.path.join(_bootstrap.RESULTS, "laundering_gamma_geom.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["gamma", "ratio_mean", "ratio_std", "cos_mean", "cos_std", "ratio_pred", "cos_pred"])
    for g in GAMMAS:
        rr, cc = np.array(rec[g]["ratio"]), np.array(rec[g]["cos"])
        s = C - M * (1 + g)
        w.writerow([g, rr.mean(), rr.std(), cc.mean(), cc.std(), abs(s) / C, 1.0 if s > 0 else -1.0])
        print(f"gamma={g:2d}  ratio={rr.mean():.2f}+-{rr.std():.2f} (pred {abs(s)/C:.2f})  "
              f"cos={cc.mean():+.2f} (pred {1.0 if s>0 else -1.0:+.0f})")
print("wrote", out)
