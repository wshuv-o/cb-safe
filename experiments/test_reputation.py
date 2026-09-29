"""CPU smoke test for the CB-SAFE+ reputation defense (synthetic updates)."""

import _bootstrap  # noqa: F401

import numpy as np

from src.aggregation.reputation import ReputationState, defend_round
from src.aggregation.secure_agg import make_clusters

rng = np.random.default_rng(0)
N, c, dim, f, gamma = 30, 3, 2_000, 0.3, 5.0
malicious = set(rng.choice(N, size=int(f * N), replace=False).tolist())
h = rng.standard_normal(dim) * 0.02  # honest direction this round

rep = ReputationState()
for r in range(15):
    active = [i for i in range(N) if i not in rep.excluded]
    deltas = {}
    for i in active:
        honest = h + rng.standard_normal(dim) * 0.005
        deltas[i] = -gamma * honest if i in malicious else honest
    clusters = make_clusters(active, c, seed=13 + r)
    means = np.stack([np.mean([deltas[i] for i in cl], axis=0) for cl in clusters])
    server_ref = h + rng.standard_normal(dim) * 0.01  # server's root-data direction
    agg = defend_round(rep, means, clusters, r, ref=server_ref)
    cos_clean = float(agg @ h) / (np.linalg.norm(agg) * np.linalg.norm(h))
    tp = len(rep.excluded & malicious)
    fp = len(rep.excluded - malicious)
    print(f"r={r:2d} aggregate·honest cos={cos_clean:+.3f} "
          f"excluded: {tp}/{len(malicious)} malicious, {fp} honest")

assert len(rep.excluded & malicious) >= 7, "defense failed to identify most attackers"
assert len(rep.excluded - malicious) <= 2, "too many honest clients excluded"
print("REPUTATION DEFENSE SMOKE TEST PASSED")
