"""Full N=30 re-run at 50 rounds with the privacy-safe (temporal) CB-SAFE+.
Priority order: CB-SAFE+ (hybrid, temporal) first, then the accuracy-competitive
baselines (Bulyan, Multi-Krum), then the rest, then the ablation variants. Writes to
the standard result paths so the table builders pick up 50-round numbers; skips a
cell only if a >=50-round CSV already exists (resumable). Edge-IIoTset is excluded
(Kaggle-only loader). N=500 excluded by design (run later).
Sharded: --workers W --shard K; each loads its datasets once and runs its slice."""

import _bootstrap  # noqa: F401

import argparse
import csv
import os

import numpy as np
from torch.utils.data import DataLoader, Subset

from src.federated import data, models
from src.federated.simulation import Config, run

ROUNDS = 50
F = [0.1, 0.2, 0.3]
SEEDS = [0, 1, 2]
DATASETS = ["cifar10", "fmnist", "emnist"]

# (name, aggregator, overlap, temporal, need_root) in PRIORITY order
METHODS = [
    ("hybrid_ov4",     "hybrid",        4, True,  True),   # CB-SAFE+ (ours) -- FIRST
    ("bulyan",         "bulyan",        1, False, False),  # strongest baselines next
    ("krum",           "krum",          1, False, False),
    ("geomedian",      "geomedian",     1, False, False),
    ("median",         "median",        1, False, False),
    ("trimmed",        "trimmed",       1, False, False),
    ("mean",           "mean",          1, False, False),
    ("fltrust",        "fltrust",       1, False, True),
    ("reputation",     "reputation",    1, False, True),   # ablation variants
    ("reputation_ov4", "reputation",    4, True,  True),
    ("reputation_tf",  "reputation_tf", 1, False, False),
    ("hybrid_tf_ov4",  "hybrid_tf",     4, True,  False),
]
# backdoor only for the roster that appears in the ASR table
BACKDOOR_METHODS = {"mean", "trimmed", "median", "krum", "reputation", "hybrid_ov4"}
ATTACKS = ["signflip", "labelflip", "backdoor"]


def job_list():
    jobs = []
    for name, agg, ov, tmp, root in METHODS:
        for attack in ATTACKS:
            if attack == "backdoor" and name not in BACKDOOR_METHODS:
                continue
            for ds in DATASETS:
                for f in F:
                    for s in SEEDS:
                        jobs.append(dict(name=name, agg=agg, ov=ov, tmp=tmp, root=root,
                                         attack=attack, ds=ds, f=f, seed=s))
    return jobs


def out_path(j):
    ov = f"_ov{j['ov']}" if j["ov"] != 1 else ""
    name = f"robust_{j['attack']}_{j['name'].replace('_ov4','')}{ov}_f{int(j['f']*100):02d}_c3_s{j['seed']}.csv"
    outdir = _bootstrap.RESULTS
    sub = {"fmnist": "fmnist", "emnist": os.path.join("kaggle", "emnist")}.get(j["ds"])
    if sub:
        outdir = os.path.join(outdir, sub)
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, name)


def done(path):
    if not os.path.exists(path):
        return False
    with open(path) as fh:
        return sum(1 for _ in fh) >= ROUNDS + 1


def loaders_for(train, test, seed, ds, need_root):
    parts = data.dirichlet_partition(np.array(train.targets), 30, 0.5, seed)
    server_dl = None
    if need_root:
        rng = np.random.default_rng(seed + 99)
        root_idx = set(rng.choice(len(train), size=200, replace=False).tolist())
        parts = [np.array([i for i in p if i not in root_idx]) for p in parts]
        server_dl = DataLoader(Subset(train, sorted(root_idx)), batch_size=64, shuffle=True)
    return data.client_loaders(train, parts, 64), DataLoader(test, batch_size=512), server_dl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)          # job-index slice (for splitting across machines)
    ap.add_argument("--end", type=int, default=10 ** 9)
    ap.add_argument("--datasets", type=str, default="")      # comma-sep filter, e.g. "fmnist,emnist" -> run those first
    args = ap.parse_args()

    jobs = job_list()[args.start:args.end]
    if args.datasets:
        keep = set(args.datasets.split(","))
        jobs = [j for j in jobs if j["ds"] in keep]
    mine = [j for i, j in enumerate(jobs) if i % args.workers == args.shard]
    todo = [j for j in mine if not done(out_path(j))]
    print(f"shard {args.shard}/{args.workers}: {len(todo)} of {len(mine)} pending", flush=True)
    ds_cache, loader_cache = {}, {}
    for k, j in enumerate(todo):
        if j["ds"] not in ds_cache:
            ds_cache[j["ds"]] = data.load_dataset(j["ds"])
        train, test = ds_cache[j["ds"]]
        key = (j["ds"], j["seed"], j["root"])
        if key not in loader_cache:
            loader_cache[key] = loaders_for(train, test, j["seed"], j["ds"], j["root"])
        cdls, tdl, sdl = loader_cache[key]
        cfg = Config(n_clients=30, rounds=ROUNDS, aggregation="cluster", aggregator=j["agg"],
                     cluster_size=3, trim=2, attack=j["attack"], f_malicious=j["f"], seed=j["seed"],
                     dataset=j["ds"], overlap=j["ov"], temporal_overlap=j["tmp"],
                     n_classes=models.n_classes_of(j["ds"]))
        p = out_path(j)
        hb = os.environ.get("CBSAFE_HEARTBEAT")
        if hb:
            open(hb, "w").write(f"shard {args.shard}: [{k+1}/{len(todo)}] {os.path.basename(p)}\n")
        hist = run(cfg, cdls, tdl, server_dl=sdl)
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(hist[0])); w.writeheader(); w.writerows(hist)
        print(f"  wrote {os.path.basename(p)} ({j['ds']}) acc={hist[-1]['acc']:.4f}", flush=True)
    print(f"shard {args.shard} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
