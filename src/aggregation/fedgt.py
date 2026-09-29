"""FedGT baseline (Xhemrishi et al., IEEE TIFS 2025): identification of malicious
clients under secure aggregation via overlapping group testing.

Faithful reimplementation of the core mechanism for a like-for-like comparison:

  - Clients are assigned to `n_groups` overlapping groups by a fixed binary
    group-testing matrix A (each client in `deg` groups). The server only ever sees
    the SUM of updates within each group (secure aggregation preserved) -- exactly
    the constraint CB-SAFE+ also respects.
  - Each group's aggregate is scored by a test statistic (server root-set loss
    increase, the same signal available to any detector); groups above a threshold
    are "positive" (contain >=1 malicious client).
  - Decoding: a client is declared malicious if it appears in "too many" positive
    groups (column-vs-syndrome matching, the standard group-testing decoder). FedGT
    uses this static one-shot decode; identified clients are removed and the clean
    groups' means are aggregated.

Differences we test against CB-SAFE+: FedGT uses STATIC groups + one-shot decode
and was designed for data poisoning; CB-SAFE+ re-randomizes groups every round and
accumulates temporal suspicion. The head-to-head runs both under identical secure
aggregation on the sign-flip (laundering) attack.
"""

from __future__ import annotations

import numpy as np


def group_testing_matrix(n_clients: int, group_size: int, deg: int,
                         seed: int) -> np.ndarray:
    """Binary A[g, i] = 1 if client i is in group g. Groups have size ~group_size
    (the secure-aggregation anonymity set) and each client is in `deg` groups.
    n_groups is chosen so group size matches: n_groups = n_clients*deg/group_size."""
    rng = np.random.default_rng(seed)
    n_groups = max(deg + 1, round(n_clients * deg / group_size))
    A = np.zeros((n_groups, n_clients), dtype=int)
    for i in range(n_clients):
        groups = rng.choice(n_groups, size=min(deg, n_groups), replace=False)
        A[groups, i] = 1
    for g in range(n_groups):  # no empty group
        if A[g].sum() == 0:
            A[g, rng.integers(n_clients)] = 1
    return A


class FedGTDetector:
    """Overlapping group-testing identification with a COMP decoder."""

    def __init__(self, n_clients: int, group_size: int = 3, deg: int = 4,
                 seed: int = 0, positive_quantile: float = 0.5):
        self.A = group_testing_matrix(n_clients, group_size, deg, seed)
        self.n_clients = n_clients
        self.positive_quantile = positive_quantile
        self.excluded: set[int] = set()

    def groups(self) -> list[list[int]]:
        return [np.where(self.A[g])[0].tolist() for g in range(self.A.shape[0])]

    def decode(self, group_scores: np.ndarray) -> set[int]:
        """COMP (Combinatorial Orthogonal Matching Pursuit) decode:
        a group is 'negative' (clean) if its score is at/below threshold; every
        client in a negative group is definitely clean; the remaining clients that
        appear in >=1 positive group are declared malicious."""
        finite = group_scores[np.isfinite(group_scores)]
        if finite.size == 0:
            return set()
        thr = np.quantile(finite, self.positive_quantile)
        positive = group_scores > thr
        clean = set()
        for g in range(self.A.shape[0]):
            if not positive[g]:
                clean |= set(np.where(self.A[g])[0].tolist())
        flagged = set()
        for i in range(self.n_clients):
            if i in clean:
                continue
            if self.A[:, i].sum() > 0 and positive[np.where(self.A[:, i])[0]].any():
                flagged.add(i)
        return flagged
