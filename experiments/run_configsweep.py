"""Multi-config sweep to beat FedGT (fairly): improve CB-SAFE+ across its own
design space while FedGT stays on the same test budget / attack / seeds.

Configs (CIFAR-10 battleground first), ordered most-promising first:
  reputation    ov4  -- root-based, adaptive threshold, 4 tests/round (matches
                        FedGT's overlapping-group test budget)
  reputation_tf ov4  -- trust-free (no root data) + matched budget (the dream:
                        beat FedGT on an axis it can't touch)
  reputation    ov1  -- root-based, adaptive, 1 test/round (baseline)
  reputation_tf ov1  -- trust-free, 1 test/round

All under sign-flip, f in {.1,.2,.3}, 3 seeds. Resumable; `reverse` for a 2nd
worker. FedGT numbers already exist for the comparison.
"""

import _bootstrap  # noqa: F401

import os
import subprocess
import sys

CONFIGS = [("reputation", 4), ("reputation_tf", 4), ("reputation", 1), ("reputation_tf", 1)]
DATASETS = ["cifar10"]           # extend winners to fmnist in a follow-up
FS = [0.2, 0.3, 0.1]             # f=0.2 first: the cell we lost
SEEDS = [0, 1, 2]


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    jobs = [(a, ov, d, f, s) for (a, ov) in CONFIGS for d in DATASETS
            for f in FS for s in SEEDS]
    tag = "fwd"
    if "reverse" in sys.argv:
        jobs = list(reversed(jobs)); tag = "rev"
    for n, (a, ov, d, f, s) in enumerate(jobs, 1):
        suf = f"_ov{ov}" if ov != 1 else ""
        name = f"robust_signflip_{a}{suf}_f{int(f * 100):02d}_c3_s{s}.csv"
        outdir = _bootstrap.RESULTS if d == "cifar10" else os.path.join(_bootstrap.RESULTS, d)
        if os.path.exists(os.path.join(outdir, name)):
            print(f"[{tag} {n}/{len(jobs)}] skip {d}/{name}", flush=True)
            continue
        print(f"[{tag} {n}/{len(jobs)}] run {a} ov{ov} {d} f={f} s={s}", flush=True)
        rc = subprocess.call([
            sys.executable, os.path.join(here, "run_robustness.py"),
            "--attack", "signflip", "--f", str(f), "--aggregator", a,
            "--overlap", str(ov), "--dataset", d, "--rounds", "30",
            "--seed", str(s), "--cluster-size", "3"])
        if rc != 0:
            print(f"FAIL {a}{suf}/{d}/{name} rc={rc}", flush=True)
    print("CONFIGSWEEP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
