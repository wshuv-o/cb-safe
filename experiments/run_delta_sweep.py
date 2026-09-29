"""Hyperparameter sensitivity for the CB-SAFE+ detector thresholds.

delta (the loss-probe flag margin) is stated in absolute cross-entropy units and is
used unchanged across four datasets, so this sweep tests whether the reported results
depend on its exact value.

This sweeps the three that matter, one at a time around the published operating
point, on CIFAR-10 sign-flip at f=0.2 (the configuration where the components
actually separate -- on the other datasets every variant agrees within 0.3 points,
so a sweep there would measure nothing):

  CBSAFE_PROBE_MARGIN  delta, the loss-probe flag margin      (published 0.25)
  CBSAFE_MIN_GAP       smallest suspicion gap counted as real (published 0.15)
  CBSAFE_MIN_FLOOR     floor a flagged group must sit above   (published 0.25)

Each point is 3 seeds at 50 rounds. Results land in results/delta_sweep/ under a
name carrying the varied knob, so build_delta_table.py can read them back.

    python experiments/run_delta_sweep.py --dry-run
    python experiments/run_delta_sweep.py --workers 4
"""

import _bootstrap  # noqa: F401

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

RESULTS = _bootstrap.RESULTS
HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(HERE, "run_robustness.py")
OUTDIR = os.path.join(RESULTS, "delta_sweep")
LOGDIR = os.path.join(RESULTS, "battery_logs")

SEEDS = [0, 1, 2]
ROUNDS = 50
# (env var, short tag, published value, sweep points)
KNOBS = [
    ("CBSAFE_PROBE_MARGIN", "pm", 0.25, [0.10, 0.15, 0.25, 0.40, 0.60]),
    ("CBSAFE_MIN_GAP", "mg", 0.15, [0.05, 0.10, 0.15, 0.25, 0.35]),
    ("CBSAFE_MIN_FLOOR", "mf", 0.25, [0.10, 0.20, 0.25, 0.35, 0.50]),
]


def out_path(tag, value, seed):
    return os.path.join(OUTDIR, f"sweep_{tag}{int(round(value * 100)):03d}_s{seed}.csv")


def jobs():
    out = []
    for env, tag, published, points in KNOBS:
        for v in points:
            for s in SEEDS:
                out.append((env, tag, v, s, published))
    return out


def is_done(tag, v, s):
    p = out_path(tag, v, s)
    if not os.path.exists(p):
        return False
    with open(p, encoding="utf-8") as fh:
        return sum(1 for _ in fh) - 1 >= ROUNDS


def run_one(a):
    env_name, tag, v, seed, _pub = a
    dst = out_path(tag, v, seed)
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", PYTHONIOENCODING="utf-8")
    env[env_name] = str(v)
    # Each run gets a private CBSAFE_OUT. The driver names its output from
    # (attack, aggregator, f, seed) only, so every sweep point for a given seed
    # would otherwise write the same filename -- workers would race, and the first
    # one would overwrite results/robust_signflip_hybrid_ov4_f20_c3_s{seed}.csv,
    # which Table I reports. Redirecting keeps the published tree untouched.
    scratch = os.path.join(OUTDIR, f"_work_{tag}{int(round(v * 100)):03d}_s{seed}")
    env["CBSAFE_OUT"] = scratch
    cmd = [sys.executable, DRIVER, "--attack", "signflip", "--f", "0.2",
           "--aggregator", "hybrid", "--overlap", "4", "--dataset", "cifar10",
           "--rounds", str(ROUNDS), "--seed", str(seed), "--cluster-size", "3"]
    log = os.path.join(LOGDIR, os.path.basename(dst).replace(".csv", ".log"))
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    produced = os.path.join(scratch, f"robust_signflip_hybrid_ov4_f20_c3_s{seed}.csv")
    if rc == 0 and os.path.exists(produced):
        os.replace(produced, dst)
        shutil.rmtree(scratch, ignore_errors=True)
    return dst, rc, time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    pending = [a for a in jobs() if not is_done(a[1], a[2], a[3])]
    print(f"delta sweep: {len(jobs())} configs | {len(jobs()) - len(pending)} done "
          f"| {len(pending)} pending | workers={args.workers}", flush=True)
    if args.dry_run:
        for env, tag, v, s, pub in pending:
            star = "  <- published" if abs(v - pub) < 1e-9 else ""
            print(f"  PENDING {tag}={v} seed={s}{star}")
        return
    if not pending:
        print("nothing to do.")
        return

    ok = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, a): a for a in pending}
        for i, fut in enumerate(as_completed(futs), 1):
            dst, rc, dt = fut.result()
            ok, fail = (ok + 1, fail) if rc == 0 else (ok, fail + 1)
            print(f"[{i}/{len(pending)}] {'OK ' if rc == 0 else 'FAIL'} {dt/60:.1f}m "
                  f"{os.path.basename(dst)}", flush=True)
    print(f"DONE: {ok} ok, {fail} failed, {(time.time()-t0)/60:.1f} min wall", flush=True)


if __name__ == "__main__":
    main()
