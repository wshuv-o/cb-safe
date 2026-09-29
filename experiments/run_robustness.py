"""One robustness run: (attack, f, aggregator, cluster_size) -> per-round acc/ASR CSV."""

import _bootstrap  # noqa: F401

import argparse
import csv
import os

import numpy as np
from torch.utils.data import DataLoader

from src.federated import data
from src.federated.simulation import Config, run


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--attack", required=True, choices=["none", "labelflip", "signflip", "backdoor"])
    p.add_argument("--f", type=float, required=True)
    p.add_argument("--aggregator", required=True,
                   choices=["mean", "median", "trimmed", "krum", "bulyan", "geomedian",
                            "fltrust", "reputation", "reputation_tf", "hybrid",
                            "hybrid_tf", "fedgt"])
    p.add_argument("--cluster-size", type=int, default=3)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--trim", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "fmnist", "emnist"])
    p.add_argument("--n-clients", type=int, default=30)
    p.add_argument("--overlap", type=int, default=1)
    p.add_argument("--signflip-gamma", type=float, default=5.0)
    p.add_argument("--attack-duty", type=float, default=1.0)
    p.add_argument("--temporal", action="store_true")   # privacy-safe temporal overlap
    args = p.parse_args()

    from src.federated import models as _models
    cfg = Config(
        n_clients=args.n_clients, rounds=args.rounds, aggregation="cluster",
        aggregator=args.aggregator, cluster_size=args.cluster_size, trim=args.trim,
        attack=args.attack, f_malicious=args.f, seed=args.seed, dataset=args.dataset,
        overlap=args.overlap, signflip_gamma=args.signflip_gamma,
        attack_duty=args.attack_duty, temporal_overlap=args.temporal,
        n_classes=_models.n_classes_of(args.dataset),
    )
    train, test = data.load_dataset(args.dataset)
    parts = data.dirichlet_partition(np.array(train.targets), cfg.n_clients, cfg.alpha, cfg.seed)
    server_dl = None
    if args.aggregator in ("reputation", "fedgt", "hybrid", "fltrust"):
        # reserve a small server root set (trust anchor), excluded from all clients
        rng = np.random.default_rng(cfg.seed + 99)
        root_idx = set(rng.choice(len(train), size=cfg.root_size, replace=False).tolist())
        parts = [np.array([i for i in p if i not in root_idx]) for p in parts]
        from torch.utils.data import Subset
        server_dl = DataLoader(Subset(train, sorted(root_idx)), batch_size=64, shuffle=True)
    client_dls = data.client_loaders(train, parts, cfg.batch_size)
    test_dl = DataLoader(test, batch_size=512, num_workers=0)

    history = run(cfg, client_dls, test_dl, server_dl=server_dl)

    ov = f"_ov{args.overlap}" if args.overlap != 1 else ""
    tmp = "_tmp" if args.temporal else ""
    gtag = f"_g{int(round(args.signflip_gamma * 10)):03d}" if args.signflip_gamma != 5.0 else ""
    dtag = f"_d{int(round(args.attack_duty * 100)):03d}" if args.attack_duty != 1.0 else ""
    name = f"robust_{args.attack}_{args.aggregator}{ov}{tmp}_f{int(args.f * 100):02d}_c{args.cluster_size}{gtag}{dtag}_s{args.seed}.csv"
    outdir = _bootstrap.RESULTS
    if tmp:                                      # temporal-overlap validation runs
        outdir = os.path.join(outdir, "temporal_check")
    elif gtag:                                   # gamma-sweep runs stay isolated from main results
        outdir = os.path.join(outdir, "gamma_sweep")
    elif dtag:                                   # duty-cycle-sweep runs likewise
        outdir = os.path.join(outdir, "duty_sweep")
    else:
        # keep EMNIST results where the analysis/table scripts already look (kaggle/emnist),
        # whether computed locally or on Kaggle; fmnist stays in results/fmnist
        subdir = {"fmnist": "fmnist", "emnist": os.path.join("kaggle", "emnist")}.get(args.dataset)
        if subdir:
            outdir = os.path.join(outdir, subdir)
        if args.n_clients != 30:
            outdir = os.path.join(outdir, f"n{args.n_clients}")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, name)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
