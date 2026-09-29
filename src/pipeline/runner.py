from __future__ import annotations

import os
import time
import traceback

from .config import load_experiment
from .naming import is_complete
from .tasks import TASKS


def select(jobs, where: dict[str, str]):
    """Keep jobs whose parameters match every key=value filter (string comparison)."""
    return [j for j in jobs if all(str(j.params.get(k)) == v for k, v in where.items())]


def run_experiment(path: str, workers: int = 1, shard: int = 0, where: dict | None = None,
                   dry_run: bool = False, force: bool = False) -> int:
    spec, jobs = load_experiment(path)
    if where:
        jobs = select(jobs, where)
    mine = [j for i, j in enumerate(jobs) if i % workers == shard]
    todo = [j for j in mine if force or not is_complete(j.output, j.params.get("rounds"))]
    name = os.path.splitext(os.path.basename(path))[0]
    print(f"{name}: {len(jobs)} jobs, shard {shard}/{workers}: {len(mine)}, "
          f"pending {len(todo)}", flush=True)
    if dry_run:
        for j in todo:
            print("  " + os.path.relpath(j.output), flush=True)
        return 0

    task = TASKS[spec["task"]]
    failures = 0
    for k, job in enumerate(todo, 1):
        t0 = time.time()
        print(f"[{k}/{len(todo)}] {os.path.relpath(job.output)}", flush=True)
        try:
            task(job)
        except Exception:  # keep a long batch going; report at the end
            failures += 1
            print(traceback.format_exc(), flush=True)
            continue
        print(f"    done in {time.time() - t0:.0f}s", flush=True)
    print(f"{name}: finished, {failures} failed", flush=True)
    return 1 if failures else 0
