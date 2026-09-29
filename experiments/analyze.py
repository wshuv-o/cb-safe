"""Aggregate results CSVs into the paper's summary tables (markdown + csv).

Also reproduces, analytically, the cluster-contamination statistics for each
robustness config (same seeded RNG as the simulation), so the observed robustness
can be compared against the (1-f)^c clean-cluster model.
"""

import _bootstrap  # noqa: F401

import glob
import os
import re

import numpy as np
import pandas as pd

from src.aggregation.secure_agg import make_clusters
from src.federated.simulation import Config, pick_malicious

R = _bootstrap.RESULTS


def contamination(f: float, cluster_size: int, n_clients: int = 30, seed: int = 0) -> dict:
    cfg = Config(n_clients=n_clients, f_malicious=f, seed=seed, cluster_size=cluster_size)
    mal = pick_malicious(cfg)
    clusters = make_clusters(list(range(n_clients)), cluster_size, seed + 13)
    dirty = sum(1 for cl in clusters if any(i in mal for i in cl))
    return {
        "k_clusters": len(clusters),
        "dirty_clusters": dirty,
        "clean_frac_observed": 1 - dirty / len(clusters),
        "clean_frac_theory": (1 - f) ** cluster_size,
    }


def robustness_table() -> pd.DataFrame | None:
    rows = []
    for path in glob.glob(os.path.join(R, "robust_*.csv")):
        m = re.match(r"robust_(\w+)_(\w+)_f(\d+)_c(\d+)_s(\d+)\.csv", os.path.basename(path))
        if not m:
            continue
        attack, agg, f, c, seed = m[1], m[2], int(m[3]) / 100, int(m[4]), int(m[5])
        df = pd.read_csv(path)
        tail = df.tail(5)
        row = {
            "attack": attack, "aggregator": agg, "f": f, "cluster_size": c, "seed": seed,
            "final_acc": round(tail["acc"].mean(), 4),
            "final_acc_std": round(tail["acc"].std(), 4),
            "rounds": len(df),
        }
        if "asr" in df.columns:
            row["final_asr"] = round(tail["asr"].mean(), 4)
        row.update({k: round(v, 4) if isinstance(v, float) else v
                    for k, v in contamination(f, c, seed=seed).items()})
        rows.append(row)
    if not rows:
        return None
    out = pd.DataFrame(rows).sort_values(["attack", "f", "aggregator"])
    out.to_csv(os.path.join(R, "summary_robustness.csv"), index=False)
    return out


def overhead_table() -> pd.DataFrame | None:
    path = os.path.join(R, "overhead.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # break-even: rounds after which per-round (KEM-independent) cost dominates setup delta
    df["setup_total_B"] = df["setup_up_B"] + df["setup_down_B"]
    df["round_total_B"] = df["round_up_B"] + df["round_down_B"]
    return df


def utility_summary() -> None:
    upath = os.path.join(R, "utility_acc.csv")
    epath = os.path.join(R, "secure_equivalence.csv")
    if os.path.exists(upath):
        acc = pd.read_csv(upath)
        print(f"\n== Utility: final acc (mean last 5 rounds) = {acc.tail(5)['acc'].mean():.4f} "
              f"over {len(acc)} rounds ==")
    if os.path.exists(epath):
        eq = pd.read_csv(epath)
        g = eq.groupby("kem").agg(max_err=("max_abs_err", "max"),
                                  mean_round_up_B=("round_up_B", "mean"),
                                  mean_mask_s=("client_mask_s", "mean"),
                                  mean_unmask_s=("server_unmask_s", "mean"),
                                  drop_rounds=("n_dropped", lambda s: int((s > 0).sum())))
        print("\n== Cryptographic equivalence (secure sum vs plain sum, all rounds) ==")
        print(g.to_string())


def main() -> None:
    utility_summary()
    ov = overhead_table()
    if ov is not None:
        print("\n== Overhead (per client, measured) ==")
        cols = ["kem", "cluster_size", "setup_total_B", "setup_wall_s",
                "round_total_B", "round_client_mask_s", "round_server_unmask_s"]
        print(ov[cols].to_string(index=False))
    rt = robustness_table()
    if rt is not None:
        print("\n== Robustness sweep ==")
        print(rt.to_string(index=False))
    print("\nwrote summary_robustness.csv" if rt is not None else "")


if __name__ == "__main__":
    main()
