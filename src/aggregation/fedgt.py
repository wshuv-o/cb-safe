from __future__ import annotations

import numpy as np


def group_testing_matrix(n_clients: int, group_size: int, deg: int,
                         seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_groups = max(deg + 1, round(n_clients * deg / group_size))
    A = np.zeros((n_groups, n_clients), dtype=int)
    for i in range(n_clients):
        groups = rng.choice(n_groups, size=min(deg, n_groups), replace=False)
        A[groups, i] = 1
    for g in range(n_groups):
        if A[g].sum() == 0:
            A[g, rng.integers(n_clients)] = 1
    return A


class FedGTDetector:

    def __init__(self, n_clients: int, group_size: int = 3, deg: int = 4,
                 seed: int = 0, positive_quantile: float = 0.5):
        self.A = group_testing_matrix(n_clients, group_size, deg, seed)
        self.n_clients = n_clients
        self.positive_quantile = positive_quantile
        self.excluded: set[int] = set()

    def groups(self) -> list[list[int]]:
        return [np.where(self.A[g])[0].tolist() for g in range(self.A.shape[0])]

    def decode(self, group_scores: np.ndarray) -> set[int]:
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
