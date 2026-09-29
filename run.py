"""Run an experiment defined by a config file.

    python run.py configs/experiments/main_grid.yaml
    python run.py configs/experiments/main_grid.yaml --dry-run
    python run.py configs/experiments/main_grid.yaml --where dataset=fmnist attack=signflip
    python run.py configs/experiments/main_grid.yaml --workers 4 --shard 0

Each job writes one CSV under results/ (or CBSAFE_OUT). Finished jobs are skipped, so
an interrupted run resumes where it stopped, and several shards can run in parallel.
"""

import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OQS_INSTALL_PATH", os.path.expanduser("~/_oqs"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="+", help="experiment config file(s)")
    ap.add_argument("--dry-run", action="store_true", help="list pending result files and exit")
    ap.add_argument("--where", nargs="*", default=[], metavar="KEY=VALUE",
                    help="run only jobs whose parameters match, e.g. dataset=cifar10 seed=0")
    ap.add_argument("--workers", type=int, default=1, help="total number of shards")
    ap.add_argument("--shard", type=int, default=0, help="index of this shard")
    ap.add_argument("--force", action="store_true", help="rerun jobs whose result already exists")
    args = ap.parse_args()

    from src.pipeline.runner import run_experiment
    where = dict(w.split("=", 1) for w in args.where)
    status = 0
    for path in args.config:
        status |= run_experiment(path, args.workers, args.shard, where, args.dry_run, args.force)
    return status


if __name__ == "__main__":
    sys.exit(main())
