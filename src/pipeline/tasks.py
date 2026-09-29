"""The five experiment tasks. Each takes one Job and writes its result file.

  robustness  federated training with an aggregation rule under attack
              (every accuracy, detection, sweep, and scaling result)
  oracle      upper bound: malicious clients excluded from round 1
  fedgt       the published FedGT detector (BCJR decoder) at N=30
  overhead    measured communication and computation cost per KEM
  utility     secure versus plain aggregation, round by round, with dropouts
"""

from __future__ import annotations

import csv
import os
import time
from contextlib import contextmanager

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..federated import models
from ..federated.simulation import Config, evaluate, local_train, pick_malicious, run, set_seeds
from . import data
from .config import dataset_config, method_config

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Detector thresholds that the sensitivity sweep varies (read by the detector at run time)
_DETECTOR_ENV = {"probe_margin": "CBSAFE_PROBE_MARGIN", "min_gap": "CBSAFE_MIN_GAP",
                 "min_floor": "CBSAFE_MIN_FLOOR"}


def write_rows(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


@contextmanager
def detector_overrides(overrides: dict | None):
    """Set detector thresholds for one job and restore the environment afterwards."""
    saved = {}
    for key, value in (overrides or {}).items():
        env = _DETECTOR_ENV[key]
        saved[env] = os.environ.get(env)
        os.environ[env] = str(value)
    try:
        yield
    finally:
        for env, old in saved.items():
            if old is None:
                os.environ.pop(env, None)
            else:
                os.environ[env] = old


def robustness(job) -> None:
    p = job.params
    ds = dataset_config(p["dataset"])
    m = method_config(p["method"])
    uses_root = bool(m.get("root", False))
    client_dls, test_dl, server_dl = data.prepare(
        ds, p["seed"], p.get("n_clients", 30), p.get("root_policy", "per_method"), uses_root,
        root_size=p.get("root_size", 200))
    temporal = p.get("temporal", m.get("temporal", False))
    cfg = Config(
        n_clients=p.get("n_clients", 30), rounds=p["rounds"], aggregation="cluster",
        aggregator=m["aggregator"], cluster_size=p.get("cluster_size", 3),
        trim=p.get("trim", 2), attack=p["attack"], f_malicious=p["f"], seed=p["seed"],
        dataset=p["dataset"], overlap=m.get("overlap", 1), temporal_overlap=temporal,
        signflip_gamma=p.get("gamma", 5.0), attack_duty=p.get("duty", 1.0),
        participation=p.get("participation", 1.0),
        n_classes=models.n_classes_of(p["dataset"]))
    with detector_overrides(p.get("detector")):
        history = run(cfg, client_dls, test_dl, server_dl=server_dl)
    write_rows(job.output, history)


def oracle(job) -> None:
    """Honest clients only, mean-aggregated: the perfect-detection upper bound."""
    p = job.params
    ds = dataset_config(p["dataset"])
    client_dls, test_dl, _ = data.prepare(ds, p["seed"], 30, "none", False)
    cfg = Config(n_clients=30, rounds=p["rounds"], aggregator="mean", cluster_size=3,
                 attack="signflip", f_malicious=p["f"], seed=p["seed"], dataset=p["dataset"],
                 n_classes=models.n_classes_of(p["dataset"]))
    set_seeds(p["seed"])
    malicious = pick_malicious(cfg)
    honest = [i for i in range(30) if i not in malicious]
    gflat = models.flat_params(models.make_model(p["dataset"]).to(DEV))
    history = []
    for r in range(p["rounds"]):
        deltas = [local_train(gflat, client_dls[i], cfg, DEV, malicious=False) for i in honest]
        gflat = gflat + np.mean(np.stack(deltas), axis=0).astype(np.float32)
        acc = evaluate(gflat, test_dl, DEV, p["dataset"])
        history.append(dict(round=r, acc=round(acc, 4), n_malicious=len(malicious),
                            excluded_malicious=len(malicious), excluded_honest=0))
    write_rows(job.output, history)


@torch.no_grad()
def _group_recall(flat, loader, dataset, n_classes):
    m = models.make_model(dataset).to(DEV)
    models.load_flat_params(m, flat)
    m.eval()
    conf = np.zeros((n_classes, n_classes))
    for x, y in loader:
        pred = m(x.to(DEV)).argmax(1).cpu().numpy()
        for t, q in zip(y.numpy(), pred):
            conf[t, q] += 1
    return float((np.diag(conf) / np.maximum(conf.sum(1), 1)).mean())


def fedgt(job) -> None:
    """FedGT's detection pipeline: 12 overlapping group tests over 30 clients, group
    recall on the server root set, KMeans test outcomes, and BCJR decoding."""
    from ..aggregation.fedgt_faithful import GroupTestFaithful
    p = job.params
    n, ds_name = 30, p["dataset"]
    nc = models.n_classes_of(ds_name)
    cfg = Config(n_clients=n, rounds=p["rounds"], attack=p["attack"], f_malicious=p["f"],
                 seed=p["seed"], dataset=ds_name, signflip_gamma=p.get("gamma", 5.0),
                 n_classes=nc)
    set_seeds(p["seed"])
    cdls, tdl, sdl = data.prepare(dataset_config(ds_name), p["seed"], n, "always_exclude",
                                  True, batch_size=64)
    malicious = pick_malicious(cfg)
    gt = GroupTestFaithful(n_clients=n)
    groups = [np.where(gt.parity_check_matrix[g])[0].tolist() for g in range(gt.n_tests)]
    mode, oneshot_at = p.get("mode", "accumulate"), p.get("oneshot_at", 15)

    global_flat = models.flat_params(models.make_model(ds_name))
    excluded: set[int] = set()
    history = []
    for r in range(p["rounds"]):
        deltas = {i: local_train(global_flat, cdls[i], cfg, DEV, i in malicious) for i in range(n)}
        if mode == "accumulate" or (mode == "oneshot" and r == oneshot_at):
            gacc = np.zeros(gt.n_tests)
            gvec = []
            for g, members in enumerate(groups):
                gd = np.mean([deltas[i] for i in members], axis=0)
                gvec.append(gd)
                gacc[g] = _group_recall(global_flat + gd.astype(np.float32), sdl, ds_name, nc)
            mat = np.stack(gvec)
            mat = mat - mat.mean(0)
            try:
                _, _, vt = np.linalg.svd(mat, full_matrices=False)
                gpca = mat @ vt[0]
            except np.linalg.LinAlgError:
                gpca = np.zeros(gt.n_tests)
            flagged = gt.perform_gt(gt.perform_clustering_and_testing(gacc, gpca, ss_thres=0.0))
            if mode == "oneshot":
                excluded = set(int(i) for i in flagged)
            else:
                excluded |= set(int(i) for i in flagged)
        kept = [i for i in range(n) if i not in excluded]
        global_flat = global_flat + np.mean([deltas[i] for i in (kept if kept else range(n))], axis=0)
        acc = evaluate(global_flat, tdl, DEV, ds_name)
        history.append(dict(round=r, acc=round(acc, 4),
                            excluded_malicious=len(excluded & malicious),
                            excluded_honest=len(excluded - malicious), n_malicious=len(malicious)))
    write_rows(job.output, history)


def overhead(job) -> None:
    """Setup and per-round cost per KEM and cluster size, with a model-sized update."""
    from ..aggregation.secure_agg import ClusterSecureAggregator, make_clusters
    p = job.params
    setup = p["setup"]
    n = p.get("n_clients", 30)
    dim = models.param_count(models.SmallCNN())
    rng = np.random.default_rng(0)
    updates = {i: rng.standard_normal(dim).astype(np.float32) * 0.01 for i in range(n)}
    rows = []
    for c in setup["cluster_sizes"]:
        clusters = make_clusters(list(range(n)), c, seed=13)
        dropouts = {cl[0] for cl in clusters[: max(1, n // (10 * c))]}  # ~10% of clusters lose one
        for kem in setup["kems"]:
            t0 = time.perf_counter()
            agg = ClusterSecureAggregator(kem, clusters)
            setup_wall = time.perf_counter() - t0
            _, _, r_norm = agg.aggregate_round(updates, round_idx=1)
            _, _, r_drop = agg.aggregate_round(
                {i: u for i, u in updates.items() if i not in dropouts}, round_idx=2,
                dropouts=dropouts)
            s = agg.stats_setup
            rows.append({
                "kem": kem, "cluster_size": c, "n_clients": n, "dim": dim,
                "setup_up_B": round(s.setup_up), "setup_down_B": round(s.setup_down),
                "setup_wall_s": round(setup_wall, 4),
                "round_up_B": round(r_norm.round_up), "round_down_B": round(r_norm.round_down),
                "round_client_mask_s": round(r_norm.t_client_mask_s / n, 6),
                "round_server_unmask_s": round(r_norm.t_server_unmask_s, 4),
                "drop_round_up_B": round(r_drop.round_up),
                "drop_server_unmask_s": round(r_drop.t_server_unmask_s, 4),
            })
            agg.close()
            print(rows[-1], flush=True)
    write_rows(job.output, rows)


def utility(job) -> None:
    """Plain FedAvg on CIFAR-10; every round the same client updates also go through
    the full secure-aggregation pipeline for each KEM, and the per-cluster secure sums
    are compared with the plain sums. Some rounds drop clients mid-round."""
    from ..aggregation.secure_agg import ClusterSecureAggregator
    from ..federated import data as fl_data
    p = job.params
    setup = p["setup"]
    cfg = Config(n_clients=p.get("n_clients", 30), rounds=p["rounds"], aggregation="fedavg",
                 seed=p["seed"])
    train, test = fl_data.load_cifar10()
    parts = fl_data.dirichlet_partition(np.array(train.targets), cfg.n_clients, cfg.alpha, cfg.seed)
    client_dls = fl_data.client_loaders(train, parts, cfg.batch_size)
    test_dl = DataLoader(test, batch_size=512, num_workers=0)
    drop_rounds = {int(k): v for k, v in (setup.get("drop_rounds") or {}).items()}

    aggs: dict[str, ClusterSecureAggregator] = {}
    equiv_rows: list[dict] = []

    def on_round(r, deltas, clusters):
        if not aggs:  # one-time setup, amortized over training
            for kem in setup["kems"]:
                aggs[kem] = ClusterSecureAggregator(kem, clusters)
        n_drop = drop_rounds.get(r, 0)
        rng = np.random.default_rng(1000 + r)
        dropped = set(rng.choice(list(deltas), size=n_drop, replace=False).tolist()) if n_drop else set()
        alive = {i: d for i, d in deltas.items() if i not in dropped}
        for kem in setup["kems"]:
            sums, _, st = aggs[kem].aggregate_round(alive, round_idx=r, dropouts=dropped)
            err = max(
                float(np.max(np.abs(sums[cid] - np.sum([alive[i] for i in clusters[cid] if i in alive], axis=0))))
                for cid in sums)
            equiv_rows.append({
                "round": r, "kem": kem, "n_dropped": len(dropped),
                "n_failed_clusters": len(st.extras.get("failed_clusters", [])),
                "max_abs_err": err,
                "round_up_B": round(st.round_up), "round_down_B": round(st.round_down),
                "client_mask_s": round(st.t_client_mask_s / len(alive), 4),
                "server_unmask_s": round(st.t_server_unmask_s, 4),
            })

    history = run(cfg, client_dls, test_dl, on_round=on_round)
    write_rows(job.output, history)
    write_rows(os.path.join(os.path.dirname(job.output), setup["equivalence_file"]), equiv_rows)
    for a in aggs.values():
        a.close()


TASKS = {"robustness": robustness, "oracle": oracle, "fedgt": fedgt,
         "overhead": overhead, "utility": utility}
