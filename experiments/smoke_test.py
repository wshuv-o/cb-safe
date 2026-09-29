"""Fast correctness checks for the crypto + aggregation stack (no training)."""

import _bootstrap  # noqa: F401

import numpy as np

from src.aggregation import robust
from src.aggregation.secure_agg import ClusterSecureAggregator, make_clusters
from src.crypto import masking, shamir

rng = np.random.default_rng(42)

# --- quantization roundtrip ---
x = (rng.standard_normal(10_000) * 0.05).astype(np.float32)
err = np.max(np.abs(masking.dequantize(masking.quantize(x)) - x.astype(np.float64)))
assert err <= 0.5 / masking.SCALE + 1e-12, err
print(f"quantize roundtrip ok, max err {err:.2e}")

# --- Shamir ---
for c, t in [(3, 2), (5, 3)]:
    sec = shamir.new_secret()
    shares = shamir.split(sec, n=c, t=t)
    assert shamir.reconstruct(shares[:t]) == sec
    assert shamir.reconstruct(shares[-t:]) == sec
print("shamir ok (3-of-2, 5-of-3 subsets)")

# --- robust aggregators sanity ---
cm = np.vstack([np.ones((8, 4)), np.full((2, 4), 100.0)])  # 2 of 10 clusters poisoned
assert np.allclose(robust.median(cm), 1.0)
assert np.allclose(robust.trimmed_mean(cm, trim=2), 1.0)
assert np.allclose(robust.multi_krum(cm, n_byzantine=2), 1.0)
print("robust aggregators ok")

# --- full secure aggregation, both KEM families, with dropouts ---
N, dim = 9, 5_000
updates = {i: (rng.standard_normal(dim) * 0.02).astype(np.float32) for i in range(N)}
for kem in ["hqc-128", "mlkem-512"]:
    for c in [3]:
        clusters = make_clusters(list(range(N)), c, seed=1)
        agg = ClusterSecureAggregator(kem, clusters)
        # no dropout
        sums, alive, st = agg.aggregate_round(updates, round_idx=0)
        worst = max(
            np.max(np.abs(sums[cid] - np.sum([updates[i] for i in cl], axis=0)))
            for cid, cl in enumerate(clusters))
        assert worst < 1e-3, worst
        # one dropout per first two clusters
        dropped = {clusters[0][0], clusters[1][1]}
        alive_updates = {i: u for i, u in updates.items() if i not in dropped}
        sums, alive, st = agg.aggregate_round(alive_updates, round_idx=1, dropouts=dropped)
        worst_d = max(
            np.max(np.abs(sums[cid] - np.sum([updates[i] for i in cl if i not in dropped], axis=0)))
            for cid, cl in enumerate(clusters))
        assert worst_d < 1e-3, worst_d
        # masks must actually hide the update: a single submission != quantized update
        print(f"{kem} c={c}: no-drop err {worst:.2e}, drop err {worst_d:.2e}, "
              f"setup up/client {agg.stats_setup.setup_up:.0f}B, "
              f"round up/client {st.round_up:.0f}B ok")
        agg.close()

print("ALL SMOKE TESTS PASSED")
