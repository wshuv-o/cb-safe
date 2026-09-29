"""Robust aggregation rules applied ACROSS cluster means.

This is where CB-SAFE resolves the hide-versus-inspect tension: individual updates
are hidden inside cluster sums by secure aggregation (the server never sees them),
and robustness operates only on the k visible cluster means. A cluster mean is
contaminated if any member is malicious, so with malicious fraction f and cluster
size c the probability a cluster stays clean is (1-f)**c — the quantitative
privacy-robustness trade-off the experiments sweep.
"""

from __future__ import annotations

import numpy as np


def mean(cluster_means: np.ndarray) -> np.ndarray:
    return cluster_means.mean(axis=0)


def median(cluster_means: np.ndarray) -> np.ndarray:
    return np.median(cluster_means, axis=0)


def trimmed_mean(cluster_means: np.ndarray, trim: int = 2) -> np.ndarray:
    """Coordinate-wise trimmed mean: drop `trim` extremes per side, average the rest."""
    k = cluster_means.shape[0]
    if 2 * trim >= k:
        raise ValueError(f"trim={trim} too large for k={k} clusters")
    s = np.sort(cluster_means, axis=0)
    return s[trim : k - trim].mean(axis=0)


def _pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    """Pairwise squared L2 distances via the Gram identity ||a-b||^2 = ||a||^2 +
    ||b||^2 - 2 a.b -- O(k^2) memory instead of the O(k^2 d) difference tensor, which
    OOMs at large cluster counts (23.6 GiB at k=166, d=229k). Mathematically identical."""
    sq = np.einsum("ij,ij->i", X, X)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    return np.maximum(d2, 0.0)


def multi_krum(cluster_means: np.ndarray, n_byzantine: int = 2, n_select: int | None = None) -> np.ndarray:
    """Multi-Krum over cluster means: score by sum of closest k-b-2 squared distances,
    average the n_select lowest-scored vectors."""
    k = cluster_means.shape[0]
    n_select = n_select or max(1, k - 2 * n_byzantine)
    d2 = _pairwise_sq_dists(cluster_means)
    n_close = max(1, k - n_byzantine - 2)
    scores = np.array([np.sort(d2[i][np.arange(k) != i])[:n_close].sum() for i in range(k)])
    chosen = np.argsort(scores)[:n_select]
    return cluster_means[chosen].mean(axis=0)


def bulyan(cluster_means: np.ndarray, n_byzantine: int = 2) -> np.ndarray:
    """Bulyan (El Mhamdi et al., ICML 2018): iteratively pick a Multi-Krum-selected
    set of size m = k - 2b, then take a coordinate-wise trimmed mean over that set
    (trim b per side). Implemented from the paper's specification over cluster means."""
    k = cluster_means.shape[0]
    b = min(n_byzantine, (k - 3) // 2 if k >= 3 else 0)
    m = max(1, k - 2 * b)
    remaining = list(range(k))
    selected = []
    while len(selected) < m and len(remaining) > 0:
        sub = cluster_means[remaining]
        ks = len(remaining)
        d2 = _pairwise_sq_dists(sub)
        n_close = max(1, ks - b - 2)
        scores = [np.sort(d2[i][np.arange(ks) != i])[:n_close].sum() for i in range(ks)]
        pick = int(np.argmin(scores))
        selected.append(remaining.pop(pick))
    sel = cluster_means[selected]
    beta = max(0, (len(sel) - m + 2 * b) // 2) if len(sel) > 2 * b else 0
    if 2 * beta >= len(sel):
        return sel.mean(axis=0)
    s = np.sort(sel, axis=0)
    return s[beta: len(sel) - beta].mean(axis=0) if beta else sel.mean(axis=0)


def geometric_median(cluster_means: np.ndarray, iters: int = 100,
                     eps: float = 1e-7) -> np.ndarray:
    """Geometric median (RFA; Pillutla et al.) via Weiszfeld's algorithm over the
    cluster means -- the smoothed-Weiszfeld robust aggregate."""
    y = cluster_means.mean(axis=0)
    for _ in range(iters):
        d = np.linalg.norm(cluster_means - y, axis=1)
        d = np.maximum(d, eps)
        w = 1.0 / d
        y_new = (w[:, None] * cluster_means).sum(axis=0) / w.sum()
        if np.linalg.norm(y_new - y) < eps:
            break
        y = y_new
    return y


def fltrust(cluster_means: np.ndarray, root_update: np.ndarray) -> np.ndarray:
    """FLTrust (Cao et al., NDSS 2021): trust score = ReLU(cosine(update, root
    update)); each update is length-normalized to the root's norm; aggregate is the
    trust-weighted average. Here 'updates' are the cluster means and the root update
    is the server's own update on a small root dataset."""
    r = root_update
    rn = np.linalg.norm(r) + 1e-12
    num = np.zeros_like(r)
    den = 0.0
    for g in cluster_means:
        gn = np.linalg.norm(g) + 1e-12
        ts = max(0.0, float(g @ r) / (gn * rn))          # ReLU(cos)
        num += ts * (rn / gn) * g                          # normalize to ||root||
        den += ts
    return num / den if den > 1e-12 else np.zeros_like(r)


AGGREGATORS = {
    "mean": mean,
    "median": median,
    "trimmed": trimmed_mean,
    "krum": multi_krum,
    "bulyan": bulyan,
    "geomedian": geometric_median,
}
