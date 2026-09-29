"""Experiment battery: the no-attack rows and 50-round sweeps, produced with the
shared driver (run_robustness.py). Resumable and parallel.

Phases:
  f0      - clean (f=0) baseline row for every rule x {cifar10,fmnist,emnist} x 3 seeds
  dial50  - cluster-size dial (median, c in {1,3,5}) re-run at 50 rounds (CIFAR)
  gamma50 - gamma-sweep off-points re-run at 50 rounds (CIFAR, f=0.2)

Usage:
  python run_battery.py --dry-run                 # list pending + count
  python run_battery.py --workers 10              # run all pending, 10 parallel
  python run_battery.py --only f0 --workers 10    # one phase
Env:
  CBSAFE_OUT can redirect the results dir (defaults to <repo>/results).
"""

import _bootstrap  # noqa: F401  (sets paths, RESULTS, env)

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

RESULTS = _bootstrap.RESULTS
HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(HERE, "run_robustness.py")
LOGDIR = os.path.join(RESULTS, "battery_logs")

SEEDS = [0, 1, 2]
# rule -> (aggregator, overlap); CB-SAFE+ is hybrid at overlap 4 (matches the main tables)
RULES = [("mean", 1), ("median", 1), ("trimmed", 1), ("krum", 1), ("bulyan", 1),
         ("geomedian", 1), ("fltrust", 1), ("fedgt", 1), ("hybrid", 4)]
DATASETS = ["cifar10", "fmnist", "emnist"]


def out_path(a):
    """Mirror run_robustness.py's output path so we can skip finished runs."""
    ov = f"_ov{a['overlap']}" if a["overlap"] != 1 else ""
    gtag = f"_g{int(round(a['gamma'] * 10)):03d}" if a["gamma"] != 5.0 else ""
    dtag = f"_d{int(round(a['duty'] * 100)):03d}" if a["duty"] != 1.0 else ""
    name = (f"robust_{a['attack']}_{a['aggregator']}{ov}"
            f"_f{int(a['f'] * 100):02d}_c{a['c']}{gtag}{dtag}_s{a['seed']}.csv")
    if gtag:
        outdir = os.path.join(RESULTS, "gamma_sweep")
    elif dtag:
        outdir = os.path.join(RESULTS, "duty_sweep")
    else:
        sub = {"fmnist": "fmnist", "emnist": os.path.join("kaggle", "emnist")}.get(a["dataset"])
        outdir = os.path.join(RESULTS, sub) if sub else RESULTS
    return os.path.join(outdir, name)


def job(attack, aggregator, f, c, seed, dataset, overlap=1, gamma=5.0, duty=1.0, rounds=50):
    return dict(attack=attack, aggregator=aggregator, f=f, c=c, seed=seed,
                dataset=dataset, overlap=overlap, gamma=gamma, duty=duty, rounds=rounds)


def rows_in(path):
    """Number of data rows (excludes header); -1 if unreadable."""
    try:
        with open(path, encoding="utf-8") as fh:
            return sum(1 for _ in fh) - 1
    except OSError:
        return -1


def is_done(a):
    """Done only if the CSV exists AND already has >= the target number of rounds,
    so shorter-horizon (e.g. 30-round) runs are correctly re-run at 50."""
    dst = out_path(a)
    return os.path.exists(dst) and rows_in(dst) >= a["rounds"]


def build_jobs(only):
    jobs = []
    if only in (None, "f0"):
        for agg, ov in RULES:
            for d in DATASETS:
                for s in SEEDS:
                    jobs.append(job("none", agg, 0.0, 3, s, d, overlap=ov))
    if only in (None, "dial50"):
        for c in (1, 3, 5):
            for f in (0.05, 0.1, 0.2, 0.3):
                for s in SEEDS:
                    jobs.append(job("signflip", "median", f, c, s, "cifar10"))
    if only in (None, "gamma50"):
        for agg, ov in [("hybrid", 4), ("median", 1), ("bulyan", 1), ("krum", 1)]:
            for g in (2.0, 3.0, 8.0, 15.0):
                for s in SEEDS:
                    jobs.append(job("signflip", agg, 0.2, 3, s, "cifar10", overlap=ov, gamma=g))
    return jobs


def cmd(a):
    c = [sys.executable, DRIVER,
         "--attack", a["attack"], "--f", str(a["f"]), "--aggregator", a["aggregator"],
         "--dataset", a["dataset"], "--rounds", str(a["rounds"]), "--seed", str(a["seed"]),
         "--cluster-size", str(a["c"])]
    if a["overlap"] != 1:
        c += ["--overlap", str(a["overlap"])]
    if a["gamma"] != 5.0:
        c += ["--signflip-gamma", str(a["gamma"])]
    if a["duty"] != 1.0:
        c += ["--attack-duty", str(a["duty"])]
    return c


def run_one(a):
    dst = out_path(a)
    os.makedirs(LOGDIR, exist_ok=True)
    log = os.path.join(LOGDIR, os.path.basename(dst).replace(".csv", ".log"))
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", PYTHONIOENCODING="utf-8")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd(a), stdout=fh, stderr=subprocess.STDOUT, env=env)
    return dst, rc, time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--only", choices=["f0", "dial50", "gamma50"], default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    jobs = build_jobs(args.only)
    pending = [a for a in jobs if not is_done(a)]
    done = len(jobs) - len(pending)
    print(f"battery: {len(jobs)} configs | {done} already done | {len(pending)} pending "
          f"| workers={args.workers}", flush=True)
    if args.dry_run:
        for a in pending:
            print("  PENDING", os.path.relpath(out_path(a), RESULTS))
        return
    if not pending:
        print("nothing to do."); return

    ok = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, a): a for a in pending}
        for i, fut in enumerate(as_completed(futs), 1):
            dst, rc, dt = fut.result()
            tag = "OK " if rc == 0 else "FAIL"
            if rc == 0:
                ok += 1
            else:
                fail += 1
            print(f"[{i}/{len(pending)}] {tag} {dt/60:.1f}m {os.path.relpath(dst, RESULTS)}"
                  + ("" if rc == 0 else "  (see battery_logs/)"), flush=True)
    print(f"DONE: {ok} ok, {fail} failed, {(time.time()-t0)/60:.1f} min wall "
          f"(workers={args.workers})", flush=True)


if __name__ == "__main__":
    main()
