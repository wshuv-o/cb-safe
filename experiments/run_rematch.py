"""FedGT rematch: CB-SAFE+ with the ADAPTIVE exclusion threshold, both root-based
(reputation) and trust-free (reputation_tf), vs FedGT (already run). Sign-flip,
cifar10 + fmnist, f in {.1,.2,.3}, 3 seeds. Resumable; `reverse` for a 2nd/3rd
worker."""

import _bootstrap  # noqa: F401

import os
import subprocess
import sys

AGGS = ["reputation", "reputation_tf"]
DATASETS = ["cifar10", "fmnist"]
FS = [0.3, 0.2, 0.1]
SEEDS = [0, 1, 2]


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    jobs = [(a, d, f, s) for a in AGGS for d in DATASETS for f in FS for s in SEEDS]
    tag = "fwd"
    if "reverse" in sys.argv:
        jobs = list(reversed(jobs)); tag = "rev"
    for n, (a, d, f, s) in enumerate(jobs, 1):
        name = f"robust_signflip_{a}_f{int(f * 100):02d}_c3_s{s}.csv"
        outdir = _bootstrap.RESULTS if d == "cifar10" else os.path.join(_bootstrap.RESULTS, d)
        if os.path.exists(os.path.join(outdir, name)):
            print(f"[{tag} {n}/{len(jobs)}] skip {d}/{name}", flush=True)
            continue
        print(f"[{tag} {n}/{len(jobs)}] run {a} {d} f={f} s={s}", flush=True)
        rc = subprocess.call([
            sys.executable, os.path.join(here, "run_robustness.py"),
            "--attack", "signflip", "--f", str(f), "--aggregator", a,
            "--dataset", d, "--rounds", "30", "--seed", str(s), "--cluster-size", "3"])
        if rc != 0:
            print(f"FAIL {a}/{d}/{name} rc={rc}", flush=True)
    print("REMATCH GRID COMPLETE", flush=True)


if __name__ == "__main__":
    main()
