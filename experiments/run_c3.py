"""C3 grid: trust-anchor-free CB-SAFE+ (reputation_tf, NO server root set) under
sign-flip. The beyond-FedGT lever -- if this matches root-based CB-SAFE+/FedGT
without any trusted data, that is a capability FedGT (which needs a per-group
maliciousness signal) cannot match. Resumable; `reverse` for a 2nd worker."""

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
        name = f"robust_signflip_reputation_tf_f{int(f * 100):02d}_c3_s{s}.csv"
        outdir = _bootstrap.RESULTS if d == "cifar10" else os.path.join(_bootstrap.RESULTS, d)
        if os.path.exists(os.path.join(outdir, name)):
            print(f"[{tag} {n}/{len(jobs)}] skip {d}/{name}", flush=True)
            continue
        print(f"[{tag} {n}/{len(jobs)}] run {d} f={f} s={s}", flush=True)
        rc = subprocess.call([
            sys.executable, os.path.join(here, "run_robustness.py"),
            "--attack", "signflip", "--f", str(f), "--aggregator", "reputation_tf",
            "--dataset", d, "--rounds", "30", "--seed", str(s), "--cluster-size", "3"])
        if rc != 0:
            print(f"FAIL {d}/{name} rc={rc}", flush=True)
    print("C3 GRID COMPLETE", flush=True)


if __name__ == "__main__":
    main()
