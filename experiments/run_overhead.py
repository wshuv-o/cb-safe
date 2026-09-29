"""Communication/computation overhead of CB-SAFE per KEM, measured (not modeled).

For each KEM and cluster size: one-time setup cost (keygen + pairwise encapsulation,
bytes and seconds) and steady-state per-round cost (masked submission, Shamir shares,
unmasking traffic) with a real model-sized vector, plus a 10%-dropout round.

Key claim this quantifies: per-round bytes are KEM-INDEPENDENT (the KEM appears only
in setup), so HQC's larger ciphertexts are a one-time, amortizable cost.
"""

import _bootstrap  # noqa: F401

import csv
import os
import time

import numpy as np

from src.aggregation.secure_agg import ClusterSecureAggregator, make_clusters
from src.federated.models import SmallCNN, param_count

N_CLIENTS = 30
KEMS = ["hqc-128", "hqc-192", "hqc-256", "mlkem-512", "mlkem-768", "mlkem-1024"]
CLUSTER_SIZES = [3, 5]


def main() -> None:
    dim = param_count(SmallCNN())
    print(f"model dimension d = {dim}")
    rng = np.random.default_rng(0)
    updates = {i: rng.standard_normal(dim).astype(np.float32) * 0.01 for i in range(N_CLIENTS)}

    out = os.path.join(_bootstrap.RESULTS, "overhead.csv")
    rows = []
    for c in CLUSTER_SIZES:
        clusters = make_clusters(list(range(N_CLIENTS)), c, seed=13)
        dropouts = {cl[0] for cl in clusters[: max(1, N_CLIENTS // (10 * c))]}  # ~10% of clusters lose 1
        for kem in KEMS:
            t0 = time.perf_counter()
            agg = ClusterSecureAggregator(kem, clusters)
            setup_wall = time.perf_counter() - t0
            _, _, r_norm = agg.aggregate_round(updates, round_idx=1)
            _, _, r_drop = agg.aggregate_round(
                {i: u for i, u in updates.items() if i not in dropouts}, round_idx=2,
                dropouts=dropouts)
            s = agg.stats_setup
            rows.append({
                "kem": kem, "cluster_size": c, "n_clients": N_CLIENTS, "dim": dim,
                "setup_up_B": round(s.setup_up), "setup_down_B": round(s.setup_down),
                "setup_wall_s": round(setup_wall, 4),
                "round_up_B": round(r_norm.round_up), "round_down_B": round(r_norm.round_down),
                "round_client_mask_s": round(r_norm.t_client_mask_s / N_CLIENTS, 6),
                "round_server_unmask_s": round(r_norm.t_server_unmask_s, 4),
                "drop_round_up_B": round(r_drop.round_up),
                "drop_server_unmask_s": round(r_drop.t_server_unmask_s, 4),
            })
            agg.close()
            print(rows[-1], flush=True)

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
