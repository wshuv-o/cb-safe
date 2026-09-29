"""Hybrid grid: temporal group-testing (per-round COMP decode over overlapping
groups + accumulation across re-randomized rounds) vs FedGT. This is the honest
shot at BEATING (not just tying) FedGT. root-probe (hybrid) and trust-free
(hybrid_tf), overlap=4, sign-flip, cifar10, f in {.1,.2,.3}, 3 seeds. Resumable;
`reverse` for a 2nd worker."""

import _bootstrap  # noqa: F401

import os
import subprocess
import sys

CONFIGS = [("hybrid", 4), ("hybrid_tf", 4)]
FS = [0.2, 0.3, 0.1]
SEEDS = [0, 1, 2]


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    jobs = [(a, ov, f, s) for (a, ov) in CONFIGS for f in FS for s in SEEDS]
    tag = "fwd"
    if "reverse" in sys.argv:
        jobs = list(reversed(jobs)); tag = "rev"
    for n, (a, ov, f, s) in enumerate(jobs, 1):
        suf = f"_ov{ov}" if ov != 1 else ""
        name = f"robust_signflip_{a}{suf}_f{int(f * 100):02d}_c3_s{s}.csv"
        if os.path.exists(os.path.join(_bootstrap.RESULTS, name)):
            print(f"[{tag} {n}/{len(jobs)}] skip {name}", flush=True)
            continue
        print(f"[{tag} {n}/{len(jobs)}] run {a} ov{ov} f={f} s={s}", flush=True)
        rc = subprocess.call([
            sys.executable, os.path.join(here, "run_robustness.py"),
            "--attack", "signflip", "--f", str(f), "--aggregator", a,
            "--overlap", str(ov), "--dataset", "cifar10", "--rounds", "30",
            "--seed", str(s), "--cluster-size", "3"])
        if rc != 0:
            print(f"FAIL {a}{suf} f={f} s={s} rc={rc}", flush=True)
    print("HYBRID GRID COMPLETE", flush=True)


if __name__ == "__main__":
    main()
