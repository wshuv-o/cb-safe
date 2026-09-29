"""Two robustness sweeps on CIFAR-10 (N=30, c=3, f=0.2, sign-flip):
  gamma sweep  -- vary sign-flip amplification gamma in {2,3,8,15} (gamma=5 is the
                  main result) to show the coordinate-wise collapse is a wide basin,
                  not a knife-edge at gamma*=2c/m-1.
  duty sweep   -- CB-SAFE+ hybrid vs a duty-cycled attacker that poisons only a
                  fraction of rounds to duck the tau=0.85 exclusion threshold.
Sharded: launch several workers with --workers W --shard K; each loads CIFAR once and
runs its slice, skipping existing outputs. Outputs mirror run_robustness naming under
results/gamma_sweep and results/duty_sweep."""

import _bootstrap  # noqa: F401

import argparse
import csv
import os

import numpy as np
from torch.utils.data import DataLoader, Subset

from src.federated import data, models
from src.federated.simulation import Config, run

F = 0.2
DS = "cifar10"
ROUNDS = 50
SEEDS = [0, 1, 2]


def job_list():
    jobs = []
    # CB-SAFE+ (hybrid) first: duty-cycle sweep, then hybrid across gamma
    for d in [0.5, 0.7, 0.8, 0.84, 0.9]:
        for s in SEEDS:
            jobs.append(dict(agg="hybrid", ov=4, gamma=5.0, duty=d, seed=s))
    for g in [2.0, 3.0, 8.0, 15.0]:
        for s in SEEDS:
            jobs.append(dict(agg="hybrid", ov=4, gamma=g, duty=1.0, seed=s))
    # then the coordinate-wise baselines across gamma
    for agg, ov in [("median", 1), ("trimmed", 1), ("krum", 1), ("bulyan", 1), ("geomedian", 1)]:
        for g in [2.0, 3.0, 8.0, 15.0]:
            for s in SEEDS:
                jobs.append(dict(agg=agg, ov=ov, gamma=g, duty=1.0, seed=s))
    return jobs


def out_path(j):
    ov = f"_ov{j['ov']}" if j["ov"] != 1 else ""
    gtag = f"_g{int(round(j['gamma'] * 10)):03d}" if j["gamma"] != 5.0 else ""
    dtag = f"_d{int(round(j['duty'] * 100)):03d}" if j["duty"] != 1.0 else ""
    name = f"robust_signflip_{j['agg']}{ov}_f{int(F * 100):02d}_c3{gtag}{dtag}_s{j['seed']}.csv"
    sub = "gamma_sweep" if gtag else "duty_sweep"
    outdir = os.path.join(_bootstrap.RESULTS, sub)
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, name)


def loaders_for(train, test, seed, need_root):
    parts = data.dirichlet_partition(np.array(train.targets), 30, 0.5, seed)
    server_dl = None
    if need_root:
        rng = np.random.default_rng(seed + 99)
        root_idx = set(rng.choice(len(train), size=200, replace=False).tolist())
        parts = [np.array([i for i in p if i not in root_idx]) for p in parts]
        server_dl = DataLoader(Subset(train, sorted(root_idx)), batch_size=64, shuffle=True)
    client_dls = data.client_loaders(train, parts, 64)
    test_dl = DataLoader(test, batch_size=512)
    return client_dls, test_dl, server_dl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    args = ap.parse_args()

    mine = [j for i, j in enumerate(job_list()) if i % args.workers == args.shard]
    todo = [j for j in mine if not os.path.exists(out_path(j))]
    print(f"shard {args.shard}/{args.workers}: {len(todo)} of {len(mine)} jobs pending", flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return

    train, test = data.load_dataset(DS)
    cache = {}
    for k, j in enumerate(todo):
        need_root = j["agg"] in ("hybrid", "reputation", "fltrust", "fedgt")
        key = (j["seed"], need_root)
        if key not in cache:
            cache[key] = loaders_for(train, test, j["seed"], need_root)
        client_dls, test_dl, server_dl = cache[key]
        cfg = Config(n_clients=30, rounds=ROUNDS, aggregation="cluster", aggregator=j["agg"],
                     cluster_size=3, trim=2, attack="signflip", f_malicious=F, seed=j["seed"],
                     dataset=DS, overlap=j["ov"], temporal_overlap=(j["ov"] == 4),
                     signflip_gamma=j["gamma"], attack_duty=j["duty"],
                     n_classes=models.n_classes_of(DS))
        p = out_path(j)
        hb = os.environ.get("CBSAFE_HEARTBEAT")
        if hb:
            open(hb, "w").write(f"shard {args.shard}: [{k + 1}/{len(todo)}] {os.path.basename(p)}\n")
        hist = run(cfg, client_dls, test_dl, server_dl=server_dl)
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(hist[0]))
            w.writeheader()
            w.writerows(hist)
        print(f"  wrote {os.path.basename(p)} acc={hist[-1]['acc']:.4f}", flush=True)
    print(f"shard {args.shard} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
