"""CB-SAFE: KEM-agnostic cluster-based secure aggregation with dropout recovery.

Protocol (per cluster of size c, Bonawitz-style double masking):

  Setup (once, amortized across all rounds):
    - every client generates a KEM keypair and publishes pk (HQC or ML-KEM/Kyber —
      the KEM is pluggable and this file never depends on which one)
    - for each unordered pair (i, j), i < j within a cluster, client i encapsulates
      to pk_j; the ciphertext travels i -> server -> j; both ends now hold the
      long-lived pairwise shared secret ss_ij

  Round r:
    - client i draws a fresh self-mask seed b_i and Shamir-shares it t-of-c among
      its cluster (including itself)
    - client i submits  y_i = q(x_i) + PRG(b_i) + sum_{j>i} m_ij - sum_{j<i} m_ij
      (mod 2**64), where m_ij = PRG(HKDF(ss_ij, r)) — fresh masks each round with
      NO new encapsulation
    - the server sums each cluster's submissions; pairwise masks cancel
    - unmasking: for every SURVIVOR i the server reconstructs b_i from t shares and
      removes PRG(b_i); for every DROPPED d the survivors reveal their per-round
      pairwise seeds with d and the server removes the survivors' orphaned halves
    - the server obtains only the cluster sum of quantized updates; anonymity set =
      number of alive members in the cluster

All message sizes are accounted from the real byte objects, and crypto wall-times
are measured, so overhead numbers in the paper come from this code path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..crypto import masking, shamir
from ..crypto.kem import KEM


@dataclass
class CommStats:
    """Byte counts, per client (averaged) unless stated otherwise."""
    setup_up: float = 0.0
    setup_down: float = 0.0
    round_up: float = 0.0
    round_down: float = 0.0
    t_setup_s: float = 0.0        # total crypto time in setup, all clients
    t_client_mask_s: float = 0.0  # total client-side masking time this round
    t_server_unmask_s: float = 0.0
    extras: dict = field(default_factory=dict)


def make_clusters(client_ids: list[int], cluster_size: int, seed: int) -> list[list[int]]:
    """Random partition into clusters of `cluster_size`. A remainder smaller than
    `cluster_size` is merged into the previous cluster so no cluster ever falls
    below the anonymity floor (a size-1 cluster would expose an individual sum)."""
    rng = np.random.default_rng(seed)
    ids = list(client_ids)
    rng.shuffle(ids)
    out = [ids[i : i + cluster_size] for i in range(0, len(ids), cluster_size)]
    if len(out) > 1 and len(out[-1]) < cluster_size:
        out[-2].extend(out.pop())
    return out


class ClusterSecureAggregator:
    """Simulates all parties faithfully; the server role only ever touches what a
    real server would receive (masked submissions, shares, revealed seeds)."""

    def __init__(self, kem_name: str, clusters: list[list[int]]):
        self.kem_name = kem_name
        self.clusters = clusters
        self.n_clients = sum(len(c) for c in clusters)
        self.keys: dict[int, KEM] = {}
        self.pairwise: dict[tuple[int, int], bytes] = {}  # (i,j) i<j -> shared secret
        self.stats_setup = CommStats()
        self._setup()

    def _setup(self) -> None:
        s = self.stats_setup
        t0 = time.perf_counter()
        pk: dict[int, bytes] = {}
        for cluster in self.clusters:
            for i in cluster:
                self.keys[i] = KEM(self.kem_name)
                pk[i] = self.keys[i].keygen()
                s.setup_up += len(pk[i])
        for cluster in self.clusters:
            for a, i in enumerate(cluster):
                for j in cluster[a + 1 :]:
                    ct, ss = KEM.encapsulate(self.kem_name, pk[j])
                    self.pairwise[(min(i, j), max(i, j))] = self.keys[j].decapsulate(ct)
                    assert self.pairwise[(min(i, j), max(i, j))] == ss
                    s.setup_up += len(ct)     # i uploads ct
                    s.setup_down += len(ct)   # j downloads ct
                # every member downloads its c-1 peers' public keys
                s.setup_down += sum(len(pk[j]) for j in cluster if j != i)
        s.t_setup_s = time.perf_counter() - t0
        s.setup_up /= self.n_clients
        s.setup_down /= self.n_clients

    def _pair_mask(self, i: int, j: int, round_idx: int, dim: int) -> np.ndarray:
        ss = self.pairwise[(min(i, j), max(i, j))]
        return masking.prg_mask(masking.pair_seed(ss, round_idx), dim)

    def aggregate_round(
        self,
        updates: dict[int, np.ndarray],
        round_idx: int,
        dropouts: set[int] | None = None,
    ) -> tuple[dict[int, np.ndarray], dict[int, int], CommStats]:
        """updates: client -> float32 delta vector. Clients in `dropouts` fail after
        key setup but before submitting. Returns (cluster_id -> float64 sum over
        alive members, cluster_id -> n_alive, per-round comm stats)."""
        dropouts = dropouts or set()
        dim = len(next(iter(updates.values())))
        stats = CommStats()
        sums: dict[int, np.ndarray] = {}
        alive_counts: dict[int, int] = {}

        for cid, cluster in enumerate(self.clusters):
            c = len(cluster)
            t_thresh = c // 2 + 1
            alive = [i for i in cluster if i not in dropouts]
            if len(alive) < t_thresh:
                # below the Shamir threshold the cluster's masks cannot be removed;
                # the protocol excludes the whole cluster from this round (graceful
                # degradation) rather than stalling the round
                stats.extras.setdefault("failed_clusters", []).append(cid)
                continue

            # --- client side: shares + masked submissions ---
            b_seed: dict[int, bytes] = {}
            b_shares: dict[int, list[tuple[int, int]]] = {}
            submissions: dict[int, np.ndarray] = {}
            t0 = time.perf_counter()
            for i in alive:
                b_seed[i] = shamir.new_secret()
                b_shares[i] = shamir.split(b_seed[i], n=c, t=t_thresh)
                stats.round_up += (c - 1) * shamir.SHARE_BYTES
                stats.round_down += (c - 1) * shamir.SHARE_BYTES
                y = masking.quantize(updates[i]) + masking.prg_mask(b_seed[i] + b"\x00", dim)
                for j in cluster:
                    if j == i:
                        continue
                    m = self._pair_mask(i, j, round_idx, dim)
                    y = y + m if i < j else y - m
                submissions[i] = y
                stats.round_up += y.nbytes
            stats.t_client_mask_s += time.perf_counter() - t0

            # --- server side: sum, then unmask ---
            t0 = time.perf_counter()
            total = np.zeros(dim, dtype=np.uint64)
            for y in submissions.values():
                total = total + y
            # remove survivors' self-masks (reconstruct b_i from t shares held by
            # alive members — shares held by dropped members are unavailable)
            for i in alive:
                avail = [sh for a, sh in enumerate(b_shares[i]) if cluster[a] not in dropouts]
                b = shamir.reconstruct(avail[:t_thresh])
                assert b == b_seed[i]
                total = total - masking.prg_mask(b + b"\x00", dim)
                stats.round_up += t_thresh * shamir.SHARE_BYTES  # survivors upload shares
            # remove orphaned pairwise-mask halves referencing dropped members
            for d in [i for i in cluster if i in dropouts]:
                for j in alive:
                    ss = self.pairwise[(min(d, j), max(d, j))]
                    seed = masking.pair_seed(ss, round_idx)  # revealed by survivor j
                    stats.round_up += len(seed)
                    m = masking.prg_mask(seed, dim)
                    total = total - m if j < d else total + m
            stats.t_server_unmask_s += time.perf_counter() - t0

            sums[cid] = masking.dequantize(total)
            alive_counts[cid] = len(alive)

        stats.round_up /= self.n_clients
        stats.round_down /= self.n_clients
        return sums, alive_counts, stats

    def close(self) -> None:
        for k in self.keys.values():
            k.free()
