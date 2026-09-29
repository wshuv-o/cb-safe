"""Utility + cryptographic-equivalence experiment (the privacy-correctness claim).

Trains plain FedAvg on CIFAR-10 (Dirichlet non-IID, 30 clients, 40 rounds). Every
round, the SAME client deltas are also pushed through the full CB-SAFE cryptographic
pipeline twice — once with code-based HQC-128, once with lattice ML-KEM-768 — and the
per-cluster secure sums are compared against the plain sums. At rounds 10 and 20,
three random clients drop out mid-round to exercise Shamir/seed-reveal recovery.

Outputs:
  results/utility_acc.csv          round, acc, t_round_s
  results/secure_equivalence.csv   per round x KEM: max |secure - plain| error,
                                   per-round comm bytes, mask/unmask seconds, dropouts
"""

import _bootstrap  # noqa: F401

import csv
import os

import numpy as np

from src.aggregation.secure_agg import ClusterSecureAggregator
from src.federated import data
from src.federated.simulation import Config, run
from torch.utils.data import DataLoader

KEMS = ["hqc-128", "mlkem-768"]
DROP_ROUNDS = {10: 3, 20: 3}  # round -> number of dropped clients


def main() -> None:
    cfg = Config(n_clients=30, rounds=40, aggregation="fedavg", seed=0)
    train, test = data.load_cifar10()
    parts = data.dirichlet_partition(np.array(train.targets), cfg.n_clients, cfg.alpha, cfg.seed)
    client_dls = data.client_loaders(train, parts, cfg.batch_size)
    test_dl = DataLoader(test, batch_size=512, num_workers=0)

    aggs: dict[str, ClusterSecureAggregator] = {}
    equiv_rows: list[dict] = []

    def on_round(r: int, deltas: dict[int, np.ndarray], clusters: list[list[int]]) -> None:
        if not aggs:  # one-time setup (amortized), built on first round
            for kem in KEMS:
                aggs[kem] = ClusterSecureAggregator(kem, clusters)
                s = aggs[kem].stats_setup
                print(f"setup[{kem}]: up={s.setup_up:.0f}B down={s.setup_down:.0f}B "
                      f"wall={s.t_setup_s:.2f}s", flush=True)
        n_drop = DROP_ROUNDS.get(r, 0)
        rng = np.random.default_rng(1000 + r)
        dropped = set(rng.choice(list(deltas), size=n_drop, replace=False).tolist()) if n_drop else set()
        alive = {i: d for i, d in deltas.items() if i not in dropped}
        for kem in KEMS:
            sums, alive_counts, st = aggs[kem].aggregate_round(alive, round_idx=r, dropouts=dropped)
            err = max(
                float(np.max(np.abs(sums[cid] - np.sum([alive[i] for i in clusters[cid] if i in alive], axis=0))))
                for cid in sums
            )
            equiv_rows.append({
                "round": r, "kem": kem, "n_dropped": len(dropped),
                "n_failed_clusters": len(st.extras.get("failed_clusters", [])),
                "max_abs_err": err,
                "round_up_B": round(st.round_up), "round_down_B": round(st.round_down),
                "client_mask_s": round(st.t_client_mask_s / len(alive), 4),
                "server_unmask_s": round(st.t_server_unmask_s, 4),
            })
            print(f"  equiv[{kem}] r={r} err={err:.2e} drop={len(dropped)}", flush=True)

    history = run(cfg, client_dls, test_dl, on_round=on_round)

    with open(os.path.join(_bootstrap.RESULTS, "utility_acc.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    with open(os.path.join(_bootstrap.RESULTS, "secure_equivalence.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(equiv_rows[0]))
        w.writeheader()
        w.writerows(equiv_rows)
    for kem, a in aggs.items():
        a.close()
    print("done")


if __name__ == "__main__":
    main()
