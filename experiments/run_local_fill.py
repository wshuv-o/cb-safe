"""Resumable local grid filler for the gaps not yet covered by earlier runs.

Fills, on the local GPU, the method/dataset cells missing from results/:
  - CIFAR-10 + FashionMNIST: the three modern baselines added later
    (bulyan, geomedian/RFA, fltrust), which no earlier local run produced.
  - EMNIST: everything (its earlier grid died on a full C: drive).

Resumable: one CSV per (dataset, attack, method, f, seed); existing CSVs are
skipped. Two workers can share the load without colliding by giving one
`--reverse` (it walks the job list from the other end).

Examples
  # high-value, low-cost: new baselines on the two image sets, sign-flip 3 seeds
  python run_local_fill.py --datasets cifar10,fmnist \
      --methods bulyan,geomedian,fltrust --attacks signflip
  # EMNIST, all methods, sign-flip (run two workers: add --reverse to the 2nd)
  python run_local_fill.py --datasets emnist --attacks signflip
"""

import _bootstrap  # noqa: F401

import argparse
import csv
import os

import numpy as np
from torch.utils.data import DataLoader, Subset

from src.federated import data, models
from src.federated.simulation import Config, run

# (aggregator, overlap, needs_root) -- same roster as the Kaggle runner.
METHODS = [
    ("mean", 1, False), ("trimmed", 1, False), ("median", 1, False), ("krum", 1, False),
    ("bulyan", 1, False), ("geomedian", 1, False), ("fltrust", 1, True), ("fedgt", 1, True),
    ("reputation", 1, True), ("reputation", 4, True), ("reputation_tf", 1, False),
    ("hybrid", 4, True),
]
FS = [0.1, 0.2, 0.3]
SF_SEEDS = [0, 1, 2]     # sign-flip: 3 seeds (headline)
OTHER_SEEDS = [0]        # label-flip / backdoor: 1 seed


def outdir_for(ds):
    root = _bootstrap.RESULTS
    sub = {"fmnist": "fmnist", "emnist": os.path.join("kaggle", "emnist")}.get(ds)
    return os.path.join(root, sub) if sub else root


def prep(ds, seed, root_size):
    train, test = data.load_dataset(ds)
    parts = data.dirichlet_partition(np.array(train.targets), 30, 0.5, seed)
    server_dl = None
    rng = np.random.default_rng(seed + 99)
    root_idx = set(rng.choice(len(train), size=root_size, replace=False).tolist())
    parts = [np.array([i for i in p if i not in root_idx]) for p in parts]
    client_dls = data.client_loaders(train, parts, 64)
    server_dl = DataLoader(Subset(train, sorted(root_idx)), batch_size=64, shuffle=True)
    test_dl = DataLoader(test, batch_size=512)
    return client_dls, test_dl, server_dl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="cifar10,fmnist,emnist")
    p.add_argument("--methods", default="")   # empty = all in METHODS
    p.add_argument("--attacks", default="signflip,labelflip,backdoor")
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--reverse", action="store_true")
    p.add_argument("--fs", default="")      # e.g. "0.3,0.2,0.1"; empty = default grid, in given order
    p.add_argument("--seeds", default="")   # e.g. "0"; empty = default per-attack seeds
    args = p.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    want_methods = {m.strip() for m in args.methods.split(",") if m.strip()}
    attacks_req = [a.strip() for a in args.attacks.split(",") if a.strip()]
    fs_override = [float(x) for x in args.fs.split(",") if x.strip()] or None
    seeds_override = [int(x) for x in args.seeds.split(",") if x.strip()] or None

    jobs = []
    for ds in datasets:
        ds_attacks = [a for a in attacks_req if not (a == "backdoor" and ds == "edgeiiot")]
        for attack in ds_attacks:
            seeds = seeds_override or (SF_SEEDS if attack == "signflip" else OTHER_SEEDS)
            for agg, ov, needs_root in METHODS:
                if want_methods and agg not in want_methods:
                    continue
                for f in (fs_override or FS):
                    for s in seeds:
                        jobs.append((ds, attack, agg, ov, needs_root, f, s))
    if args.reverse:
        jobs = list(reversed(jobs))

    prepared = {}
    total = len(jobs)
    ran = skipped = 0
    for i, (ds, attack, agg, ov, needs_root, f, s) in enumerate(jobs):
        suf = f"_ov{ov}" if ov != 1 else ""
        name = f"robust_{attack}_{agg}{suf}_f{int(f * 100):02d}_c3_s{s}.csv"
        odir = outdir_for(ds)
        os.makedirs(odir, exist_ok=True)
        path = os.path.join(odir, name)
        if os.path.exists(path):
            skipped += 1
            continue
        if (ds, s) not in prepared:
            prepared[(ds, s)] = prep(ds, s, root_size=200)
        client_dls, test_dl, server_dl = prepared[(ds, s)]
        cfg = Config(n_clients=30, rounds=args.rounds, aggregation="cluster",
                     aggregator=agg, cluster_size=3, attack=attack, f_malicious=f,
                     seed=s, dataset=ds, overlap=ov,
                     n_classes=models.n_classes_of(ds))
        print(f"[{i + 1}/{total}] run {ds}/{name}", flush=True)
        hist = run(cfg, client_dls, test_dl, server_dl=(server_dl if needs_root else None))
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(hist[0]))
            w.writeheader(); w.writerows(hist)
        ran += 1
    print(f"LOCAL FILL COMPLETE (ran {ran}, skipped {skipped}, of {total})", flush=True)


if __name__ == "__main__":
    main()
