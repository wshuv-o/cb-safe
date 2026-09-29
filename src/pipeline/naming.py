from __future__ import annotations

import os

from .config import REPO, dataset_config

RESULTS = os.environ.get("CBSAFE_OUT") or os.path.join(REPO, "results")


def default_name(p: dict) -> str:
    n = p.get("n_clients", 30)
    gamma = float(p.get("gamma", 5.0))
    duty = float(p.get("duty", 1.0))
    ntag = f"_N{n}" if n != 30 else ""
    gtag = f"_g{int(round(gamma * 10)):03d}" if gamma != 5.0 else ""
    dtag = f"_d{int(round(duty * 100)):03d}" if duty != 1.0 else ""
    return (f"robust_{p['attack']}_{p['method']}{ntag}_f{int(p['f'] * 100):02d}"
            f"_c{p.get('cluster_size', 3)}{gtag}{dtag}_s{p['seed']}.csv")


def output_path(spec: dict, p: dict) -> str:
    out = spec.get("output") or {}
    fields = {**p, "f100": int(round(p.get("f", 0) * 100))}
    name = out["name"].format(**fields) if "name" in out else default_name(p)
    subdir = out.get("dir")
    if subdir is None:
        subdir = dataset_config(p["dataset"]).get("results_dir", "") if "dataset" in p else ""
    return os.path.join(RESULTS, subdir, name)


def is_complete(path: str, rounds: int | None) -> bool:
    """A result counts as done only if it exists and covers the requested rounds, so
    shorter earlier runs are redone at the configured horizon."""
    if not os.path.exists(path):
        return False
    if not rounds:
        return True
    with open(path, encoding="utf-8") as fh:
        return sum(1 for _ in fh) - 1 >= rounds
