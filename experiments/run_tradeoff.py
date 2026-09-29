"""Supplementary sweep tracing the privacy-robustness trade-off empirically:
fixed attack (signflip) and rule (median), varying cluster size c and malicious
fraction f. c=1 is the no-privacy classical robust-aggregation baseline; larger c
buys anonymity and pays in contaminated cluster sums per (1-f)^c.

Run AFTER run_sweep.py (GPU). Skips existing CSVs, like run_sweep."""

import _bootstrap  # noqa: F401

import os
import subprocess
import sys

JOBS = [
    # (cluster_size, f) — c=3 rows come from the main sweep
    (1, 0.30), (1, 0.20), (1, 0.10), (1, 0.05),
    (5, 0.30), (5, 0.20), (5, 0.10), (5, 0.05),
    (3, 0.05),
]


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    for n, (c, f) in enumerate(JOBS, 1):
        name = f"robust_signflip_median_f{int(f * 100):02d}_c{c}_s0.csv"
        if os.path.exists(os.path.join(_bootstrap.RESULTS, name)):
            print(f"[{n}/{len(JOBS)}] skip (exists): {name}", flush=True)
            continue
        print(f"[{n}/{len(JOBS)}] running: c={c} f={f}", flush=True)
        rc = subprocess.call([
            sys.executable, os.path.join(here, "run_robustness.py"),
            "--attack", "signflip", "--f", str(f), "--aggregator", "median",
            "--cluster-size", str(c), "--rounds", "30",
        ])
        if rc != 0:
            print(f"FAILED (rc={rc}): {name}", flush=True)
    print("tradeoff sweep complete")


if __name__ == "__main__":
    main()
