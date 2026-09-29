"""Controlled detector swap: FedGT decoder vs CB-SAFE+ decoder over the SAME
cluster secure-aggregation and grouping (overlap=4), N=500, sign-flip f=0.2,
seeds {0,2,3}. Per-round accuracy / attackers-caught / honest-FP, seed-averaged.
Only the decoder differs, so any gap is attributable to the detector, not HQC
or the grouping."""

import _bootstrap  # noqa: F401
import csv
import os

import numpy as np

D = os.path.join(_bootstrap.RESULTS, "scale50")
SEEDS = [0, 2, 3]


def load(stem):
    per = []
    for s in SEEDS:
        p = os.path.join(D, f"robust_signflip_{stem}_N500_f20_c3_s{s}.csv")
        if not os.path.exists(p):
            return None
        r = list(csv.DictReader(open(p)))
        if len(r) < 50:
            return None
        per.append(r)
    nr = min(len(x) for x in per)
    acc = [np.mean([float(per[k][i]["acc"]) for k in range(3)]) for i in range(nr)]
    em = [np.mean([int(per[k][i]["excluded_malicious"]) for k in range(3)]) for i in range(nr)]
    eh = [np.mean([int(per[k][i]["excluded_honest"]) for k in range(3)]) for i in range(nr)]
    return acc, em, eh


h = load("hybrid_ov4")   # CB-SAFE+ decoder (COMP + temporal, re-randomized groups)
f = load("fedgt")        # FedGT decoder (static one-shot COMP), same deg=4 groups + same root-loss signal

print("Controlled swap @ N=500, overlap=4, seeds {0,2,3}  (acc | caught/100 | honestFP/400)")
print(f"{'round':>5} | {'CB-SAFE+ acc':>12} {'caught':>6} {'FP':>4} | {'FedGT acc':>9} {'caught':>6} {'FP':>4}")
for i in range(0, 50, 3):
    if h:
        hs = f"{h[0][i]:>12.3f} {h[1][i]:>6.0f} {h[2][i]:>4.0f}"
    else:
        hs = f"{'run':>12} {'':>6} {'':>4}"
    if f:
        fs = f"{f[0][i]:>9.3f} {f[1][i]:>6.0f} {f[2][i]:>4.0f}"
    else:
        fs = f"{'run':>9} {'':>6} {'':>4}"
    print(f"r{i+1:>4} | {hs} | {fs}")

if h and f:
    print(f"\nFINAL (r50): CB-SAFE+ acc={h[0][-1]:.3f} caught={h[1][-1]:.0f} FP={h[2][-1]:.0f}"
          f"  ||  FedGT acc={f[0][-1]:.3f} caught={f[1][-1]:.0f} FP={f[2][-1]:.0f}")
    print(f"peak honest-FP over the run: CB-SAFE+={max(h[2]):.0f}  FedGT={max(f[2]):.0f}")
