from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..adversary import attacks
from ..aggregation import robust
from ..aggregation.reputation import ReputationState, defend_round
from ..aggregation.secure_agg import make_clusters
from .models import flat_params, load_flat_params, make_model


@dataclass
class Config:
    n_clients: int = 30
    rounds: int = 40
    local_epochs: int = 1
    lr: float = 0.01
    momentum: float = 0.9
    batch_size: int = 64
    alpha: float = 0.5
    seed: int = 0
    aggregation: str = "fedavg"
    aggregator: str = "mean"
    cluster_size: int = 3
    trim: int = 2
    attack: str = "none"
    f_malicious: float = 0.0
    root_size: int = 200
    dataset: str = "cifar10"
    n_classes: int = 10
    signflip_gamma: float = 5.0
    overlap: int = 1
    participation: float = 1.0


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_malicious(cfg: Config) -> set[int]:
    rng = np.random.default_rng(cfg.seed + 7)
    n_mal = int(round(cfg.f_malicious * cfg.n_clients))
    return set(rng.choice(cfg.n_clients, size=n_mal, replace=False).tolist())


def local_train(global_flat: np.ndarray, loader: DataLoader, cfg: Config,
                device: torch.device, malicious: bool) -> np.ndarray:
    model = make_model(cfg.dataset).to(device)
    load_flat_params(model, global_flat)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(cfg.local_epochs):
        for x, y in loader:
            if malicious and cfg.attack == "labelflip":
                y = attacks.flip_labels(y, cfg.n_classes)
            if malicious and cfg.attack == "backdoor":
                x, y = attacks.poison_batch(x, y)
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
    delta = flat_params(model) - global_flat
    if malicious and cfg.attack == "signflip":
        delta = attacks.signflip_delta(delta, cfg.signflip_gamma)
    return delta


@torch.no_grad()
def mean_loss(global_flat: np.ndarray, loader: DataLoader, device: torch.device,
              dataset: str = "cifar10") -> float:
    model = make_model(dataset).to(device)
    load_flat_params(model, global_flat)
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total = n = 0
    for x, y in loader:
        total += loss_fn(model(x.to(device)), y.to(device)).item()
        n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(global_flat: np.ndarray, loader: DataLoader, device: torch.device,
             dataset: str = "cifar10") -> float:
    model = make_model(dataset).to(device)
    load_flat_params(model, global_flat)
    model.eval()
    correct = total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += len(y)
    return correct / total


@torch.no_grad()
def attack_success_rate(global_flat: np.ndarray, loader: DataLoader,
                        device: torch.device, target: int = attacks.BACKDOOR_TARGET,
                        dataset: str = "cifar10") -> float:
    model = make_model(dataset).to(device)
    load_flat_params(model, global_flat)
    model.eval()
    hits = total = 0
    for x, y in loader:
        keep = y != target
        if keep.sum() == 0:
            continue
        x = attacks.stamp_trigger(x[keep])
        pred = model(x.to(device)).argmax(1).cpu()
        hits += (pred == target).sum().item()
        total += keep.sum().item()
    return hits / max(total, 1)


def aggregate(deltas: dict[int, np.ndarray], cfg: Config,
              clusters: list[list[int]]) -> np.ndarray:
    if cfg.aggregation == "fedavg":
        return np.mean(np.stack(list(deltas.values())), axis=0)
    means = np.stack([np.mean(np.stack([deltas[i] for i in cl]), axis=0) for cl in clusters])
    if cfg.aggregator == "trimmed":
        return robust.trimmed_mean(means, trim=cfg.trim)
    return robust.AGGREGATORS[cfg.aggregator](means)


def run(cfg: Config, client_dls: list[DataLoader], test_dl: DataLoader,
        on_round=None, server_dl: DataLoader | None = None) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(cfg.seed)
    model = make_model(cfg.dataset).to(device)
    global_flat = flat_params(model)
    clusters = make_clusters(list(range(cfg.n_clients)), cfg.cluster_size, cfg.seed + 13)
    malicious = pick_malicious(cfg)
    rep_aggs = ("reputation", "reputation_tf", "hybrid", "hybrid_tf")
    rep = ReputationState(n_clients=cfg.n_clients) if cfg.aggregator in rep_aggs else None
    trustfree = cfg.aggregator in ("reputation_tf", "hybrid_tf")
    comp = cfg.aggregator in ("hybrid", "hybrid_tf")
    fedgt = None
    if cfg.aggregator == "fedgt":
        from ..aggregation.fedgt import FedGTDetector
        fedgt = FedGTDetector(cfg.n_clients, group_size=cfg.cluster_size,
                              deg=4, seed=cfg.seed + 13)

    history: list[dict] = []
    for r in range(cfg.rounds):
        t0 = time.perf_counter()
        excluded = rep.excluded if rep is not None else (fedgt.excluded if fedgt else set())
        active = [i for i in range(cfg.n_clients) if i not in excluded]
        if cfg.participation < 1.0 and len(active) > cfg.cluster_size:
            k = int(round(cfg.participation * len(active)))
            k -= k % cfg.cluster_size
            k = max(cfg.cluster_size, min(k, len(active)))
            rng_r = np.random.default_rng(cfg.seed + 1000 + r)
            active = sorted(int(i) for i in rng_r.choice(active, size=k, replace=False))
        deltas = {
            i: local_train(global_flat, client_dls[i], cfg, device, i in malicious)
            for i in active
        }
        if fedgt is not None:
            base = mean_loss(global_flat, server_dl, device, cfg.dataset) if server_dl is not None else 0.0
            g_means, g_scores = [], []
            for members in fedgt.groups():
                alive_m = [i for i in members if i in deltas]
                if not alive_m:
                    g_means.append(None); g_scores.append(-np.inf); continue
                gm = np.mean(np.stack([deltas[i] for i in alive_m]), axis=0)
                g_means.append(gm)
                if server_dl is None:
                    score = 0.0
                elif os.environ.get("FEDGT_STAT") == "acc":
                    score = -evaluate(global_flat + gm.astype(np.float32), server_dl,
                                      device, cfg.dataset)
                else:
                    score = mean_loss(global_flat + gm.astype(np.float32), server_dl,
                                      device, cfg.dataset) - base
                g_scores.append(score)
            newly = fedgt.decode(np.array(g_scores))
            fedgt.excluded |= (newly - set())
            keep = [gm for gm, s in zip(g_means, g_scores)
                    if gm is not None and s <= np.quantile([x for x in g_scores if np.isfinite(x)], 0.5)]
            delta_agg = np.mean(np.stack(keep), axis=0) if keep else np.mean(
                np.stack([gm for gm in g_means if gm is not None]), axis=0)
        elif rep is not None:
            clusters_r = []
            for p in range(cfg.overlap):
                clusters_r += make_clusters(active, cfg.cluster_size,
                                            cfg.seed + 13 + r + p * 1009)
            means = np.stack(
                [np.mean(np.stack([deltas[i] for i in cl]), axis=0) for cl in clusters_r])
            probe = None
            if server_dl is not None and not trustfree:
                base = mean_loss(global_flat, server_dl, device, cfg.dataset)
                probe = lambda m: mean_loss(
                    global_flat + m.astype(np.float32), server_dl, device,
                    cfg.dataset) - base
            delta_agg = defend_round(rep, means, clusters_r, r, probe=probe, comp=comp)
        elif cfg.aggregator == "fltrust":
            means = np.stack([np.mean(np.stack([deltas[i] for i in cl]), axis=0)
                              for cl in clusters])
            root_update = local_train(global_flat, server_dl, cfg, device, malicious=False)
            delta_agg = robust.fltrust(means, root_update)
        else:
            if on_round is not None:
                on_round(r, deltas, clusters)
            delta_agg = aggregate(deltas, cfg, clusters)
        global_flat = global_flat + delta_agg.astype(np.float32)
        acc = evaluate(global_flat, test_dl, device, cfg.dataset)
        row = {
            "round": r,
            "acc": acc,
            "t_round_s": time.perf_counter() - t0,
            "n_malicious": len(malicious),
        }
        if fedgt is not None:
            row["excluded_malicious"] = len(fedgt.excluded & malicious)
            row["excluded_honest"] = len(fedgt.excluded - malicious)
        if rep is not None:
            row["excluded_malicious"] = len(rep.excluded & malicious)
            row["excluded_honest"] = len(rep.excluded - malicious)
            info = rep.last_info
            dirty = [j for j, cl in enumerate(clusters_r) if any(i in malicious for i in cl)]
            hit = [j for j in info.get("flagged", []) if j in dirty]
            row["n_dirty"] = len(dirty)
            row["n_flagged"] = len(info.get("flagged", []))
            row["dirty_flagged"] = len(hit)
            susp_m = [rep.suspicion(i) for i in rep.rounds if i in malicious]
            susp_h = [rep.suspicion(i) for i in rep.rounds if i not in malicious]
            row["susp_mal"] = round(float(np.mean(susp_m)), 3) if susp_m else 0.0
            row["susp_hon"] = round(float(np.mean(susp_h)), 3) if susp_h else 0.0
        if cfg.attack == "backdoor":
            row["asr"] = attack_success_rate(global_flat, test_dl, device,
                                             dataset=cfg.dataset)
        history.append(row)
        _hb = os.environ.get("CBSAFE_HEARTBEAT")
        if _hb:
            with open(_hb, "w") as _f:
                _f.write(f"round {r + 1}/{cfg.rounds} acc={acc:.4f} "
                         f"exc_mal={row.get('excluded_malicious', '-')} "
                         f"exc_hon={row.get('excluded_honest', '-')}\n")
        print(f"[{cfg.attack}|{cfg.aggregation}/{cfg.aggregator}|f={cfg.f_malicious}] "
              f"round {r + 1}/{cfg.rounds} acc={acc:.4f}"
              + (f" asr={row['asr']:.4f}" if "asr" in row else ""),
              flush=True)
    return history
