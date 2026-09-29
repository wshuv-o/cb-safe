"""Sequential robustness sweep. Runs highest-information configs first and writes a
CSV per run, so partial completion still yields usable results. Skips runs whose
output CSV already exists (safe to re-launch)."""

import _bootstrap  # noqa: F401

import os
import subprocess
import sys

ATTACKS = ["signflip", "backdoor", "labelflip"]
FS = [0.3, 0.2, 0.1]
AGGREGATORS = ["mean", "trimmed", "median", "krum"]
CLUSTER_SIZE = 3
ROUNDS = 30


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    jobs = [(a, f, g) for a in ATTACKS for f in FS for g in AGGREGATORS]
    for n, (attack, f, agg) in enumerate(jobs, 1):
        name = f"robust_{attack}_{agg}_f{int(f * 100):02d}_c{CLUSTER_SIZE}_s0.csv"
        if os.path.exists(os.path.join(_bootstrap.RESULTS, name)):
            print(f"[{n}/{len(jobs)}] skip (exists): {name}", flush=True)
            continue
        print(f"[{n}/{len(jobs)}] running: attack={attack} f={f} agg={agg}", flush=True)
        rc = subprocess.call([
            sys.executable, os.path.join(here, "run_robustness.py"),
            "--attack", attack, "--f", str(f), "--aggregator", agg,
            "--cluster-size", str(CLUSTER_SIZE), "--rounds", str(ROUNDS),
        ])
        if rc != 0:
            print(f"FAILED (rc={rc}): {name}", flush=True)
    print("sweep complete")


if __name__ == "__main__":
    main()
