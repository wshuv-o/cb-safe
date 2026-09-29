"""Oracle (perfect-detection) upper bound under sign-flip: the true malicious clients
are excluded from round 1 and the server mean-aggregates the honest clients only.
This is the same idealized baseline FedGT plots in every figure. Output matches the
robustness naming so the plotters pick it up automatically:
results/.../robust_signflip_oracle_f{XX}_c3_s{seed}.csv. Resumable; sharded."""

import _bootstrap  # noqa: F401

import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.federated import data, models
from src.federated.simulation import Config, local_train, evaluate, set_seeds, pick_malicious
from src.federated.models import make_model, flat_params

ROUNDS = 50
F = [0.1, 0.2, 0.3]
SEEDS = [0, 1, 2]
DATASETS = ["cifar10", "fmnist", "emnist", "edgeiiot"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def out_path(ds, f, s):
    sub = {"fmnist": "fmnist", "emnist": os.path.join("kaggle", "emnist"),
           "edgeiiot": os.path.join("kaggle", "edgeiiot")}.get(ds, "")
    outdir = os.path.join(_bootstrap.RESULTS, sub)
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, f"robust_signflip_oracle_f{int(f*100):02d}_c3_s{s}.csv")


def load(ds, seed):
    if ds == "edgeiiot":
        from src.federated import kaggle_datasets as kd
        csvp = kd.find_edgeiiot_csv(os.environ.get("EDGEIIOT_ROOT", "/kaggle/input"))
        tr, te, lab = kd.load_edgeiiot(csvp, seed=seed)
        return tr, te, lab, 128
    tr, te = data.load_dataset(ds)
    return tr, te, np.array(tr.targets), 64


def done(p):
    return os.path.exists(p) and sum(1 for _ in open(p)) >= ROUNDS + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    args = ap.parse_args()
    dsets = [d for d in args.datasets.split(",") if d]
    jobs = [(ds, f, s) for ds in dsets for f in F for s in SEEDS]
    mine = [j for i, j in enumerate(jobs) if i % args.workers == args.shard]
    todo = [j for j in mine if not done(out_path(*j))]
    print(f"oracle shard {args.shard}/{args.workers}: {len(todo)}/{len(mine)} pending", flush=True)

    cache = {}
    for ds, f, s in todo:
        if (ds, s) not in cache:
            tr, te, lab, batch = load(ds, s)
            parts = data.dirichlet_partition(lab, 30, 0.5, s)
            cache[(ds, s)] = (tr, DataLoader(te, batch_size=512), parts, batch)
        tr, tdl, parts, batch = cache[(ds, s)]
        cfg = Config(n_clients=30, rounds=ROUNDS, aggregator="mean", cluster_size=3,
                     attack="signflip", f_malicious=f, seed=s, dataset=ds,
                     n_classes=models.n_classes_of(ds))
        set_seeds(s)
        malicious = pick_malicious(cfg)
        honest = [i for i in range(30) if i not in malicious]
        cdls = [DataLoader(Subset(tr, parts[i].tolist()), batch_size=batch, shuffle=True)
                for i in range(30)]
        gflat = flat_params(make_model(ds).to(DEV))
        hist = []
        for r in range(ROUNDS):
            deltas = [local_train(gflat, cdls[i], cfg, DEV, malicious=False) for i in honest]
            gflat = gflat + np.mean(np.stack(deltas), axis=0).astype(np.float32)
            acc = evaluate(gflat, tdl, DEV, ds)
            hist.append(dict(round=r, acc=round(acc, 4), n_malicious=len(malicious),
                             excluded_malicious=len(malicious), excluded_honest=0))
        p = out_path(ds, f, s)
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(hist[0])); w.writeheader(); w.writerows(hist)
        print(f"  wrote {os.path.basename(p)} ({ds}) acc={hist[-1]['acc']:.4f}", flush=True)
    print(f"oracle shard {args.shard} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
