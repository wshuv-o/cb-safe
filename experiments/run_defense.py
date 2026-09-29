"""CB-SAFE+ defense evaluation: reputation aggregator vs all three attacks in the
regime where every static rule collapsed (f = 0.2, 0.3). Skips existing CSVs."""

import _bootstrap  # noqa: F401

import os
import subprocess
import sys

JOBS = [
    ("signflip", 0.30), ("signflip", 0.20),
    ("backdoor", 0.30), ("backdoor", 0.20),
    ("labelflip", 0.30), ("labelflip", 0.20),
]


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    for n, (attack, f) in enumerate(JOBS, 1):
        name = f"robust_{attack}_reputation_f{int(f * 100):02d}_c3_s0.csv"
        if os.path.exists(os.path.join(_bootstrap.RESULTS, name)):
            print(f"[{n}/{len(JOBS)}] skip (exists): {name}", flush=True)
            continue
        print(f"[{n}/{len(JOBS)}] running: attack={attack} f={f} reputation", flush=True)
        rc = subprocess.call([
            sys.executable, os.path.join(here, "run_robustness.py"),
            "--attack", attack, "--f", str(f), "--aggregator", "reputation",
            "--cluster-size", "3", "--rounds", "30",
        ])
        if rc != 0:
            print(f"FAILED (rc={rc}): {name}", flush=True)
    print("defense eval complete")


if __name__ == "__main__":
    main()
