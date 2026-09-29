"""Run FedGT (its BCJR decoder, recall signal, and KMeans test) across the paper's
configurations at N=30, FedGT's published configuration. Reuses the shared
training/attack/data harness; only the detector is FedGT's own compiled code.

Two protocols:
  --mode accumulate : run the group test every round, accumulate exclusions (matches
                      CB-SAFE+ / COMP-FedGT protocol used elsewhere).
  --mode oneshot    : train all clients until --oneshot_at, run FedGT's one-shot
                      identification ONCE there, exclude, continue (FedGT's own
                      intended deployment; its best-case false-positive count).
Logs per-round accuracy / attackers-caught / honest-FP. Resumable (skips existing).
"""

import _bootstrap  # noqa: F401

import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.federated import data, models
from src.federated.simulation import Config, set_seeds, pick_malicious, local_train, evaluate
from src.federated.models import make_model, flat_params, load_flat_params
from src.aggregation.fedgt_faithful import GroupTestFaithful

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def group_recall(flat, loader, dataset, n_classes):
    m = make_model(dataset).to(DEV); load_flat_params(m, flat); m.eval()
    conf = np.zeros((n_classes, n_classes))
    for x, y in loader:
        pred = m(x.to(DEV)).argmax(1).cpu().numpy()
        for t, p in zip(y.numpy(), pred):
            conf[t, p] += 1
    return float((np.diag(conf) / np.maximum(conf.sum(1), 1)).mean())


def prep(seed, ds, n=30, root=200):
    if ds == "edgeiiot":
        from src.federated import kaggle_datasets as kd
        csvp = kd.find_edgeiiot_csv(os.environ.get("EDGEIIOT_ROOT", "/kaggle/input"))
        train, test, lab = kd.load_edgeiiot(csvp, seed=seed)
        parts = data.dirichlet_partition(lab, n, 0.5, seed)
    else:
        train, test = data.load_dataset(ds)
        parts = data.dirichlet_partition(np.array(train.targets), n, 0.5, seed)
    rng = np.random.default_rng(seed + 99)
    ridx = set(rng.choice(len(train), size=root, replace=False).tolist())
    parts = [np.array([i for i in p if i not in ridx]) for p in parts]
    return (data.client_loaders(train, parts, 64),
            DataLoader(test, batch_size=512),
            DataLoader(Subset(train, sorted(ridx)), batch_size=64, shuffle=True))


def run(ds, f, seed, rounds, gamma, mode, oneshot_at, attack="signflip"):
    n = 30; nc = models.n_classes_of(ds)
    cfg = Config(n_clients=n, rounds=rounds, attack=attack, f_malicious=f, seed=seed,
                 dataset=ds, signflip_gamma=gamma, n_classes=nc)
    set_seeds(seed)
    cdls, tdl, sdl = prep(seed, ds, n)
    malicious = pick_malicious(cfg)
    gt = GroupTestFaithful(n_clients=n)
    groups = [np.where(gt.parity_check_matrix[g])[0].tolist() for g in range(gt.n_tests)]

    global_flat = flat_params(make_model(ds))
    excluded = set()
    hist = []
    for r in range(rounds):
        deltas = {i: local_train(global_flat, cdls[i], cfg, DEV, i in malicious) for i in range(n)}
        do_detect = (mode == "accumulate") or (mode == "oneshot" and r == oneshot_at)
        if do_detect:
            gacc = np.zeros(gt.n_tests); gvec = []
            for g, mem in enumerate(groups):
                gd = np.mean([deltas[i] for i in mem], axis=0); gvec.append(gd)
                gacc[g] = group_recall(global_flat + gd.astype(np.float32), sdl, ds, nc)
            M = np.stack(gvec); M = M - M.mean(0)
            try:
                _, _, vt = np.linalg.svd(M, full_matrices=False); gpca = M @ vt[0]
            except np.linalg.LinAlgError:
                gpca = np.zeros(gt.n_tests)
            tests = gt.perform_clustering_and_testing(gacc, gpca, ss_thres=0.0)
            flagged = gt.perform_gt(tests)
            if mode == "oneshot":
                excluded = set(int(i) for i in flagged)   # single identification
            else:
                excluded |= set(int(i) for i in flagged)  # accumulate
        kept = [i for i in range(n) if i not in excluded]
        global_flat = global_flat + np.mean([deltas[i] for i in (kept if kept else range(n))], axis=0)
        acc = evaluate(global_flat, tdl, DEV, ds)
        hist.append(dict(round=r, acc=round(acc, 4),
                         excluded_malicious=len(excluded & malicious),
                         excluded_honest=len(excluded - malicious), n_malicious=len(malicious)))
        hb = os.environ.get("CBSAFE_HEARTBEAT")
        if hb:
            open(hb, "w").write(f"{ds} f{f} s{seed} {mode}: round {r+1}/{rounds} acc={acc:.4f} "
                                f"exc_mal={len(excluded & malicious)} exc_hon={len(excluded - malicious)}")
    return hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="cifar10,fmnist,emnist")
    p.add_argument("--fs", default="0.1,0.2,0.3")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--gamma", type=float, default=5.0)
    p.add_argument("--mode", default="accumulate")   # accumulate | oneshot
    p.add_argument("--oneshot_at", type=int, default=15)
    p.add_argument("--attack", default="signflip")
    args = p.parse_args()
    odir = os.path.join(_bootstrap.RESULTS, "fedgt_faithful"); os.makedirs(odir, exist_ok=True)
    dsets = [d for d in args.datasets.split(",") if d]
    fs = [float(x) for x in args.fs.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    for ds in dsets:
        for f in fs:
            for s in seeds:
                tag = "" if args.mode == "accumulate" else "_oneshot"
                name = f"faithful_fedgt_{ds}_{args.attack}_f{int(f*100):02d}_N30_s{s}{tag}.csv"
                path = os.path.join(odir, name)
                if os.path.exists(path):
                    print("skip", name, flush=True); continue
                print("RUN", name, flush=True)
                hist = run(ds, f, s, args.rounds, args.gamma, args.mode, args.oneshot_at, args.attack)
                with open(path, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(hist[0])); w.writeheader(); w.writerows(hist)
                print("  done:", hist[-1], flush=True)
    print("FAITHFUL SWEEP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
