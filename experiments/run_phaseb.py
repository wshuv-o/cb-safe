"""Phase B: scale the evidence to journal standard. Resumable (skips existing CSVs).

Order (highest information first):
  B1  CIFAR-10 seeds 1,2 for the headline table: signflip + backdoor, 5 rules x 3 f
  B2  FashionMNIST seed 0: baseline + signflip grid (second-dataset claim)
  B3  CIFAR-10 seeds 1,2 for the c-dial (signflip/median, c in {1,5} x f + c3 f05)
  B4  FashionMNIST seeds 1,2 signflip grid
  B5  100-client demonstration (baseline, median, CB-SAFE+ at f=0.2, c=5)
  B6  CIFAR-10 baseline (attack=none) seeds 1,2 for stats normalization
"""

import _bootstrap  # noqa: F401

import os
import subprocess
import sys

RULES = ["mean", "trimmed", "median", "krum", "reputation"]
FS = [0.3, 0.2, 0.1]


def job(attack, f, agg, c=3, seed=0, dataset="cifar10", n_clients=30, rounds=30):
    return {"attack": attack, "f": f, "agg": agg, "c": c, "seed": seed,
            "dataset": dataset, "n": n_clients, "rounds": rounds}


def jobs():
    out = []
    # B1: replicate the headline grid on seeds 1,2
    for seed in (1, 2):
        for attack in ("signflip", "backdoor"):
            for f in FS:
                for agg in RULES:
                    out.append(job(attack, f, agg, seed=seed))
    # B2: FashionMNIST seed 0
    out.append(job("none", 0.0, "mean", dataset="fmnist"))
    for f in FS:
        for agg in ("mean", "median", "krum", "reputation"):
            out.append(job("signflip", f, agg, dataset="fmnist"))
    # B3: c-dial seeds 1,2
    for seed in (1, 2):
        for c, fs in ((1, [0.3, 0.2, 0.1, 0.05]), (5, [0.3, 0.2, 0.1, 0.05]),
                      (3, [0.05])):
            for f in fs:
                out.append(job("signflip", f, "median", c=c, seed=seed))
    # B4: FashionMNIST seeds 1,2
    for seed in (1, 2):
        out.append(job("none", 0.0, "mean", dataset="fmnist", seed=seed))
        for f in FS:
            for agg in ("mean", "median", "krum", "reputation"):
                out.append(job("signflip", f, agg, dataset="fmnist", seed=seed))
    # B5: 100 clients (c=5 -> k=20 cluster sums)
    out.append(job("none", 0.0, "mean", c=5, n_clients=100))
    out.append(job("signflip", 0.2, "median", c=5, n_clients=100))
    out.append(job("signflip", 0.2, "reputation", c=5, n_clients=100))
    # B6: CIFAR baseline seeds 1,2
    for seed in (1, 2):
        out.append(job("none", 0.0, "mean", seed=seed))
    return out


def csv_path(j):
    name = (f"robust_{j['attack']}_{j['agg']}_f{int(j['f'] * 100):02d}"
            f"_c{j['c']}_s{j['seed']}.csv")
    d = _bootstrap.RESULTS
    if j["dataset"] != "cifar10":
        d = os.path.join(d, j["dataset"])
    if j["n"] != 30:
        d = os.path.join(d, f"n{j['n']}")
    return os.path.join(d, name)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    all_jobs = jobs()
    # a second worker launched with "reverse" eats the queue back-to-front so two
    # processes meet in the middle; both skip existing CSVs (see os.path.exists)
    tag = "fwd"
    if "reverse" in sys.argv:
        all_jobs = list(reversed(all_jobs))
        tag = "rev"
    # optional dataset filter, e.g. "only=fmnist", so a CPU-only third worker can
    # take a cleanly disjoint slice from the two GPU workers
    only = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("only=")), None)
    if only:
        all_jobs = [j for j in all_jobs if j["dataset"] == only]
        tag = f"{only}"
    for i, j in enumerate(all_jobs, 1):
        if os.path.exists(csv_path(j)):
            print(f"[{tag} {i}/{len(all_jobs)}] skip: {os.path.basename(csv_path(j))}", flush=True)
            continue
        print(f"[{tag} {i}/{len(all_jobs)}] run: {j}", flush=True)
        rc = subprocess.call([
            sys.executable, os.path.join(here, "run_robustness.py"),
            "--attack", j["attack"], "--f", str(j["f"]), "--aggregator", j["agg"],
            "--cluster-size", str(j["c"]), "--rounds", str(j["rounds"]),
            "--seed", str(j["seed"]), "--dataset", j["dataset"],
            "--n-clients", str(j["n"]),
        ])
        if rc != 0:
            print(f"FAILED (rc={rc}): {j}", flush=True)
    print("PHASE B COMPLETE", flush=True)


if __name__ == "__main__":
    main()
