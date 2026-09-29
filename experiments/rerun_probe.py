"""Diagnostic: re-run ONE (N, method, seed) with byte-identical data to measure
GPU run-to-run nondeterminism. Writes to a distinct *_rerun<tag>.csv so the
original stored result is never overwritten. Compares to the stored CSV."""

import _bootstrap  # noqa: F401

import argparse
import csv
import os

from run_scale import prep  # identical partition/root/test as the main sweep
from src.federated import models
from src.federated.simulation import Config, run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agg", default="hybrid")
    p.add_argument("--ov", type=int, default=4)
    p.add_argument("--N", type=int, default=500)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--tag", default="A")
    p.add_argument("--dataset", default="fmnist")
    p.add_argument("--attack", default="signflip")
    p.add_argument("--f", type=float, default=0.2)
    args = p.parse_args()

    needs_root = args.agg in ("hybrid", "fedgt", "fltrust")
    client_dls, test_dl, server_dl = prep(args.dataset, args.N, args.seed)
    cfg = Config(n_clients=args.N, rounds=args.rounds, aggregation="cluster",
                 aggregator=args.agg, cluster_size=3, attack=args.attack,
                 f_malicious=args.f, seed=args.seed, dataset=args.dataset,
                 overlap=args.ov, participation=1.0,
                 n_classes=models.n_classes_of(args.dataset))
    hist = run(cfg, client_dls, test_dl, server_dl=(server_dl if needs_root else None))

    suf = f"_ov{args.ov}" if args.ov != 1 else ""
    base = f"robust_{args.attack}_{args.agg}{suf}_N{args.N}_f{int(args.f*100):02d}_c3_s{args.seed}"
    odir = os.path.join(_bootstrap.RESULTS, "scale")
    out = os.path.join(odir, f"{base}_rerun{args.tag}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hist[0]))
        w.writeheader(); w.writerows(hist)

    stored = os.path.join(odir, f"{base}.csv")
    new_acc = float(hist[-1]["acc"])
    new_em = int(hist[-1].get("excluded_malicious", 0))
    new_eh = int(hist[-1].get("excluded_honest", 0))
    line = f"RERUN{args.tag} {base}: acc={new_acc:.4f} exc_mal={new_em} exc_hon={new_eh}"
    if os.path.exists(stored):
        r = list(csv.DictReader(open(stored)))
        old_acc = float(r[-1]["acc"])
        line += f"  ||  STORED acc={old_acc:.4f}  delta={new_acc-old_acc:+.4f}"
    print(line, flush=True)
    with open(os.path.join(odir, "_rerun_log.txt"), "a") as fh:
        fh.write(line + "\n")


if __name__ == "__main__":
    main()
