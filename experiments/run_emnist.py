"""Full local EMNIST grid at the best configuration (c=3, adaptive threshold,
overlap where used, hybrid included). Fills the EMNIST columns of the wide tables.
Resumable; `reverse` for a 2nd worker. Results -> results/kaggle/emnist/ (where the
table/analysis scripts already look)."""

import _bootstrap  # noqa: F401

import os
import subprocess
import sys

# sign-flip: full method set, 3 seeds (Tables I & II)
SF_METHODS = [("mean", 1), ("trimmed", 1), ("median", 1), ("krum", 1),
              ("reputation", 1), ("reputation", 4), ("reputation_tf", 1),
              ("hybrid", 4), ("fedgt", 1)]
# backdoor + label-flip: core rules, 1 seed (Table IV; footnoted single-seed)
OTHER_METHODS = [("mean", 1), ("median", 1), ("krum", 1), ("reputation", 1)]
FS = [0.2, 0.3, 0.1]


def jobs():
    js = []
    for a, ov in SF_METHODS:
        for f in FS:
            for s in (0, 1, 2):
                js.append(("signflip", a, ov, f, s))
    for attack in ("backdoor", "labelflip"):
        for a, ov in OTHER_METHODS:
            for f in FS:
                js.append((attack, a, ov, f, 0))
    return js


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(_bootstrap.RESULTS, "kaggle", "emnist")
    js = jobs()
    if "reverse" in sys.argv:
        js = list(reversed(js))
    for n, (attack, a, ov, f, s) in enumerate(js, 1):
        suf = f"_ov{ov}" if ov != 1 else ""
        name = f"robust_{attack}_{a}{suf}_f{int(f * 100):02d}_c3_s{s}.csv"
        if os.path.exists(os.path.join(outdir, name)):
            print(f"[{n}/{len(js)}] skip {name}", flush=True)
            continue
        print(f"[{n}/{len(js)}] run {attack} {a} ov{ov} f={f} s={s}", flush=True)
        rc = subprocess.call([
            sys.executable, os.path.join(here, "run_robustness.py"),
            "--attack", attack, "--f", str(f), "--aggregator", a, "--overlap", str(ov),
            "--dataset", "emnist", "--rounds", "25", "--seed", str(s), "--cluster-size", "3"])
        if rc != 0:
            print(f"FAIL {attack}/{a}{suf} f={f} s={s} rc={rc}", flush=True)
    print("EMNIST GRID COMPLETE", flush=True)


if __name__ == "__main__":
    main()
