"""FedGT head-to-head grid (the make-or-break Q1 comparison): FedGT vs CB-SAFE+
under identical secure-aggregation constraints on the sign-flip (laundering)
attack. Runs the FedGT arm; the CB-SAFE+ (reputation) arm already exists from
Phase B. Local datasets only (cifar10, fmnist); Edge-IIoTset/EMNIST FedGT run on
Kaggle where their data lives. Resumable; supports `reverse` for a 2nd worker."""

import _bootstrap  # noqa: F401

import os
import subprocess
import sys

DATASETS = ["cifar10", "fmnist"]
FS = [0.3, 0.2, 0.1]
SEEDS = [0, 1, 2]


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    jobs = [(d, f, s) for d in DATASETS for f in FS for s in SEEDS]
    tag = "fwd"
    if "reverse" in sys.argv:
        jobs = list(reversed(jobs)); tag = "rev"
    for n, (d, f, s) in enumerate(jobs, 1):
        name = f"robust_signflip_fedgt_f{int(f * 100):02d}_c3_s{s}.csv"
        outdir = _bootstrap.RESULTS if d == "cifar10" else os.path.join(_bootstrap.RESULTS, d)
        if os.path.exists(os.path.join(outdir, name)):
            print(f"[{tag} {n}/{len(jobs)}] skip {d}/{name}", flush=True)
            continue
        print(f"[{tag} {n}/{len(jobs)}] run {d} f={f} s={s}", flush=True)
        rc = subprocess.call([
            sys.executable, os.path.join(here, "run_robustness.py"),
            "--attack", "signflip", "--f", str(f), "--aggregator", "fedgt",
            "--dataset", d, "--rounds", "30", "--seed", str(s), "--cluster-size", "3"])
        if rc != 0:
            print(f"FAIL {d}/{name} rc={rc}", flush=True)
    print("FEDGT GRID COMPLETE", flush=True)


if __name__ == "__main__":
    main()
