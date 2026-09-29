"""Experiment configuration: load YAML files and expand them into jobs.

An experiment file (configs/experiments/*.yaml) names a task and a grid. Each grid
block is a Cartesian product over its list-valued keys; blocks are concatenated, so
an experiment can combine, for example, the sign-flip grid over every method with a
smaller backdoor grid. Scalar keys at the top level are defaults for every job and can
be overridden inside a block.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field
from typing import Any

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(REPO, "configs")

# Keys that describe the whole experiment rather than one job.
_EXPERIMENT_KEYS = {"task", "description", "grid", "variants", "output"}


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def dataset_config(name: str) -> dict:
    cfg = load_yaml(os.path.join(CONFIG_DIR, "datasets", f"{name}.yaml"))
    cfg.setdefault("name", name)
    return cfg


def method_config(name: str) -> dict:
    methods = load_yaml(os.path.join(CONFIG_DIR, "methods.yaml"))
    if name not in methods:
        raise KeyError(f"unknown method '{name}'; defined in configs/methods.yaml: "
                       f"{', '.join(sorted(methods))}")
    return dict(methods[name], name=name)


@dataclass
class Job:
    """One run: every parameter needed to reproduce a single result file."""
    task: str
    params: dict[str, Any]
    output: str                       # path of the result file this job writes
    variant: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)


def _as_list(v: Any) -> list:
    return v if isinstance(v, list) else [v]


def _expand_block(defaults: dict, block: dict) -> list[dict]:
    merged = {**defaults, **block}
    keys = sorted(merged)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(_as_list(merged[k]) for k in keys))]


def load_experiment(path: str) -> tuple[dict, list[Job]]:
    """Return the experiment spec and its expanded list of jobs, in file order."""
    spec = load_yaml(path)
    if "task" not in spec:
        raise ValueError(f"{path}: missing 'task'")
    from . import naming  # local import: naming depends on this module's helpers

    defaults = {k: v for k, v in spec.items() if k not in _EXPERIMENT_KEYS}
    blocks = spec.get("grid") or [{}]
    variants = spec.get("variants") or [{}]
    jobs: list[Job] = []
    for block in blocks:
        for params in _expand_block(defaults, block):
            for variant in variants:
                p = {**params, **{k: v for k, v in variant.items() if k != "tag"}}
                if "tag" in variant:
                    p["variant"] = variant["tag"]
                jobs.append(Job(task=spec["task"], params=p,
                                output=naming.output_path(spec, p), variant=variant))
    return spec, jobs
