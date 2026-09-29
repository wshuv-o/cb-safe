"""One-shot Kaggle runner: every method x every dataset x every attack, resumable.
Upload the bundle as a Kaggle dataset, add the Edge-IIoTset dataset, enable a T4
GPU + Internet, and run this. It writes one CSV per (dataset, attack, method, f,
seed) under /kaggle/working/results/. Re-run to continue where a 12h session left
off (finished CSVs are skipped).

Methods compared (all under the SAME harness = controlled comparison):
  baselines:  mean, trimmed mean, median, Multi-Krum, Bulyan (ICML'18),
              geometric-median/RFA, FLTrust (NDSS'21), FedGT (TIFS'25)
  ours:       CB-SAFE+ (reputation), CB-SAFE+ ov4, trust-free, hybrid
Each is our own faithful implementation of the published algorithm (cited).
"""

import _bootstrap  # noqa: F401

import csv
import os

import numpy as np
from torch.utils.data import DataLoader, Subset

from src.federated import data, models
from src.federated import kaggle_datasets as kd
from src.federated.data import dirichlet_partition
from src.federated.simulation import Config, run

OUT = os.environ.get("CBSAFE_OUT", "/kaggle/working/results")
# Scope the grid to a subset of datasets via CBSAFE_DATASETS (comma-separated),
# e.g. CBSAFE_DATASETS=edgeiiot for the Edge-only Kaggle bundle. Default: all four.
_ALL_DATASETS = ["cifar10", "fmnist", "emnist", "edgeiiot"]
DATASETS = [d.strip() for d in os.environ.get("CBSAFE_DATASETS", ",".join(_ALL_DATASETS)).split(",") if d.strip()]
# (aggregator, overlap, needs_root)
METHODS = [
    ("mean", 1, False), ("trimmed", 1, False), ("median", 1, False), ("krum", 1, False),
    ("bulyan", 1, False), ("geomedian", 1, False), ("fltrust", 1, True), ("fedgt", 1, True),
    ("reputation", 1, True), ("reputation", 4, True), ("reputation_tf", 1, False),
    ("hybrid", 4, True),
]
_M = [m.strip() for m in os.environ.get("CBSAFE_METHODS", "").split(",") if m.strip()]
if _M:                     # e.g. CBSAFE_METHODS=fedgt to top up one aggregator
    METHODS = [m for m in METHODS if m[0] in _M]
FS = [0.1, 0.2, 0.3]
# Scope the malicious fractions, e.g. CBSAFE_FS=0.1 to run a single control config
# instead of the full 3x3 grid.
_FS = [float(x) for x in os.environ.get("CBSAFE_FS", "").split(",") if x.strip()]
if _FS:
    FS = _FS
SF_SEEDS = [0, 1, 2]       # sign-flip: 3 seeds (headline)
_SF = [int(x) for x in os.environ.get("CBSAFE_SF_SEEDS", "").split(",") if x.strip()]
if _SF:
    SF_SEEDS = _SF
# Backdoor/label-flip default to 1 seed to bound runtime, which the paper footnotes.
# CBSAFE_OTHER_SEEDS overrides it so missing seeds can be topped up without editing
# this file, e.g. CBSAFE_OTHER_SEEDS=1,2 to complete an existing grid.
OTHER_SEEDS = [int(x) for x in
               os.environ.get("CBSAFE_OTHER_SEEDS", "0").split(",") if x.strip()]
ROUNDS = int(os.environ.get("CBSAFE_ROUNDS", "25"))
# Optionally scope attacks (comma-separated), e.g. CBSAFE_ATTACKS=labelflip.
_ATTACKS = [a.strip() for a in os.environ.get("CBSAFE_ATTACKS", "").split(",") if a.strip()] or None


def subdir(ds):
    return {"cifar10": "", "fmnist": "fmnist", "emnist": os.path.join("kaggle", "emnist"),
            "edgeiiot": os.path.join("kaggle", "edgeiiot")}[ds]


def load(ds, seed):
    if ds == "edgeiiot":
        csv = kd.find_edgeiiot_csv()
        if not csv:
            return None
        tr, te, lab = kd.load_edgeiiot(csv, seed=seed)
        return tr, te, lab, 128
    tr, te = data.load_dataset(ds)
    return tr, te, np.array(tr.targets), 64


def available(ds):
    """Cheap probe: is this dataset's source present? (edgeiiot needs its CSV
    added to the notebook; the image sets auto-download.) Avoids retrying prep()
    once per method when a dataset was simply not added."""
    if ds == "edgeiiot":
        return kd.find_edgeiiot_csv() is not None
    return True


def prep(ds, seed, root_size=200):
    got = load(ds, seed)
    if got is None:
        return None
    tr, te, lab, batch = got
    parts = dirichlet_partition(lab, 30, alpha=0.5, seed=seed)
    rng = np.random.default_rng(seed + 99)
    allidx = np.concatenate(parts)
    root = set(rng.choice(allidx, size=min(root_size, len(allidx) // 4), replace=False).tolist())
    parts = [np.array([i for i in p if i not in root]) for p in parts]
    cdl = [DataLoader(Subset(tr, ix.tolist()), batch_size=batch, shuffle=True) for ix in parts]
    sdl = DataLoader(Subset(tr, sorted(root)), batch_size=64, shuffle=True)
    tdl = DataLoader(te, batch_size=512)
    return cdl, tdl, sdl


def main():
    os.makedirs(OUT, exist_ok=True)
    total = done = 0
    for ds in DATASETS:
        if not available(ds):
            print(f"[skip dataset] {ds}: source not found under /kaggle/input "
                  f"(did you add the Edge-IIoTset dataset?) - skipping", flush=True)
            continue
        attacks = ["signflip", "labelflip"] + (["backdoor"] if ds != "edgeiiot" else [])
        if _ATTACKS:
            # Intersect, but let CBSAFE_ATTACKS introduce "none" (the f=0 clean
            # baseline), which is not part of the default attack grid. Filtering
            # alone would silently drop it and the run would do nothing.
            attacks = [a for a in _ATTACKS if a in attacks or a == "none"]
        odir = os.path.join(OUT, subdir(ds))
        os.makedirs(odir, exist_ok=True)
        prepared = {}
        for attack in attacks:
            seeds = SF_SEEDS if attack == "signflip" else OTHER_SEEDS
            for agg, ov, needs_root in METHODS:
                if attack != "signflip" and agg in ("reputation_tf", "hybrid", "fedgt") and ov == 1:
                    pass  # keep detection methods on non-signflip too
                for f in FS:
                    for s in seeds:
                        suf = f"_ov{ov}" if ov != 1 else ""
                        name = f"robust_{attack}_{agg}{suf}_f{int(f*100):02d}_c3_s{s}.csv"
                        path = os.path.join(odir, name)
                        total += 1
                        if os.path.exists(path):
                            done += 1
                            continue
                        if (ds, s) not in prepared:
                            got = prep(ds, s)
                            if got is None:
                                print(f"[skip dataset] {ds}: data not found", flush=True)
                                break
                            prepared[(ds, s)] = got
                        cdl, tdl, sdl = prepared[(ds, s)]
                        cfg = Config(n_clients=30, rounds=ROUNDS, aggregation="cluster",
                                     aggregator=agg, cluster_size=3, attack=attack,
                                     f_malicious=f, seed=s, dataset=ds, overlap=ov,
                                     n_classes=models.n_classes_of(ds))
                        print(f"[run] {ds}/{name}", flush=True)
                        hist = run(cfg, cdl, tdl, server_dl=(sdl if needs_root else None))
                        with open(path, "w", newline="") as fh:
                            w = csv.DictWriter(fh, fieldnames=list(hist[0]))
                            w.writeheader(); w.writerows(hist)
                        done += 1
    print(f"ALL-KAGGLE COMPLETE ({done} present of {total} planned)", flush=True)


if __name__ == "__main__":
    main()
