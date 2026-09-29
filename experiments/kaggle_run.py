"""Self-contained Kaggle runner: extends the CB-SAFE robustness evaluation to
EMNIST (handwritten, 47 classes) and Edge-IIoTset (tabular IoT intrusion
detection), writing per-run CSVs in the same format as the local experiments.

Kaggle usage (GPU + Internet on):
    !python kaggle_run.py --out /kaggle/working/results

Resumable: any run whose CSV already exists is skipped, so a 12h Kaggle session
that times out can be continued by re-running the same cell.

Notes
- Crypto (liboqs) is NOT needed here: masking correctness is exact arithmetic,
  already proven dataset-independently on CIFAR-10, and the robustness sweep uses
  the plain cluster-mean path. This runner is pure PyTorch.
- Edge-IIoTset has no natural image trigger, so the backdoor attack is run only on
  EMNIST; sign-flip and label-flip run on both.
"""

import argparse
import csv
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np
from torch.utils.data import DataLoader, Subset

from src.federated import kaggle_datasets as kd
from src.federated import models
from src.federated.data import dirichlet_partition
from src.federated.simulation import Config, run

ATTACKS = {
    "emnist": ["signflip", "backdoor", "labelflip"],
    "edgeiiot": ["signflip", "labelflip"],
}
FS = [0.3, 0.2, 0.1]
RULES = ["mean", "median", "krum", "reputation", "trimmed"]


def client_loaders(train_ds, partitions, batch_size):
    return [DataLoader(Subset(train_ds, ix.tolist()), batch_size=batch_size,
                       shuffle=True, num_workers=0) for ix in partitions]


def prepare(dataset, out_root, seed, n_clients, root_size):
    """Load a dataset, build client loaders + server root loader for one seed."""
    if dataset == "emnist":
        train, test, labels = kd.load_emnist(os.path.join(out_root, "_data"))
        batch = 64
    elif dataset == "edgeiiot":
        csv_path = kd.find_edgeiiot_csv()
        if not csv_path:
            print(f"[skip] Edge-IIoTset CSV not found under /kaggle/input", flush=True)
            return None
        train, test, labels = kd.load_edgeiiot(csv_path, seed=seed)
        batch = 128
    else:
        raise ValueError(dataset)

    parts = dirichlet_partition(labels, n_clients, alpha=0.5, seed=seed)
    # reserve a server root set (trust anchor) from the union, excluded from clients
    rng = np.random.default_rng(seed + 99)
    all_idx = np.concatenate(parts)
    root_idx = set(rng.choice(all_idx, size=min(root_size, len(all_idx) // 4),
                              replace=False).tolist())
    parts = [np.array([i for i in p if i not in root_idx]) for p in parts]
    server_dl = DataLoader(Subset(train, sorted(root_idx)), batch_size=64, shuffle=True)
    client_dls = client_loaders(train, parts, batch)
    test_dl = DataLoader(test, batch_size=512, num_workers=0)
    return client_dls, test_dl, server_dl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(REPO, "results", "kaggle"))
    p.add_argument("--datasets", nargs="+", default=["edgeiiot", "emnist"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--rounds", type=int, default=25)
    p.add_argument("--n-clients", type=int, default=30)
    p.add_argument("--cluster-size", type=int, default=3)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for dataset in args.datasets:
        dsdir = os.path.join(args.out, dataset)
        os.makedirs(dsdir, exist_ok=True)
        prepared = {}  # seed -> (client_dls, test_dl, server_dl)
        for seed in args.seeds:
            jobs = [(a, f, g) for a in ATTACKS[dataset] for f in FS for g in RULES]
            # one no-attack baseline per seed
            jobs = [("none", 0.0, "mean")] + jobs
            for attack, f, agg in jobs:
                name = (f"robust_{attack}_{agg}_f{int(f * 100):02d}"
                        f"_c{args.cluster_size}_s{seed}.csv")
                path = os.path.join(dsdir, name)
                if os.path.exists(path):
                    print(f"[skip exists] {dataset}/{name}", flush=True)
                    continue
                if seed not in prepared:
                    got = prepare(dataset, args.out, seed, args.n_clients, 200)
                    if got is None:
                        break
                    prepared[seed] = got
                client_dls, test_dl, server_dl = prepared[seed]
                cfg = Config(
                    n_clients=args.n_clients, rounds=args.rounds,
                    aggregation="fedavg" if attack == "none" else "cluster",
                    aggregator=agg, cluster_size=args.cluster_size,
                    attack=attack, f_malicious=f, seed=seed, dataset=dataset,
                    n_classes=models.n_classes_of(dataset),
                )
                print(f"[run] {dataset} {name}", flush=True)
                hist = run(cfg, client_dls, test_dl,
                           server_dl=server_dl if agg == "reputation" else None)
                with open(path, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(hist[0]))
                    w.writeheader(); w.writerows(hist)
    print("KAGGLE RUN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
