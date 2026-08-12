from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import robust

PROBE_MARGIN = 0.25

SCALE_N = 50
PROBE_K_MAD = 1.0
EXCL_K_MAD = 2.5


@dataclass
class ReputationState:
    warmup: int = 5
    tau: float = 0.85
    adaptive: bool = True
    min_gap: float = 0.20
    min_floor: float = 0.35
    n_byzantine: int = 2
    n_clients: int = 0
    flags: dict = field(default_factory=dict)
    rounds: dict = field(default_factory=dict)
    excluded: set = field(default_factory=set)
    last_info: dict = field(default_factory=dict)

    def suspicion(self, i: int) -> float:
        r = self.rounds.get(i, 0)
        return self.flags.get(i, 0) / r if r else 0.0

    def exclude_now(self) -> set:
        cand = [(i, self.suspicion(i)) for i in self.rounds
                if self.rounds[i] >= self.warmup and i not in self.excluded]
        if len(cand) < 3:
            return set()
        if not self.adaptive:
            return {i for i, s in cand if s > self.tau}
        if self.n_clients > SCALE_N:
            arr = np.array([s for _, s in cand])
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med))) + 1e-9
            thr = max(self.min_floor, med + EXCL_K_MAD * mad)
            high = {i for i, s in cand if s > thr}
            if 0 < len(high) <= len(cand) // 2:
                return high
            return set()
        vals = np.sort(np.array([s for _, s in cand]))
        gaps = np.diff(vals)
        if gaps.size == 0 or gaps.max() < self.min_gap:
            return set()
        split = vals[int(np.argmax(gaps))] + gaps.max() / 2.0
        if split < self.min_floor:
            return set()
        high = {i for i, s in cand if s > split}
        if len(high) > len(cand) // 2:
            return set()
        return high


def defend_round(
    state: ReputationState,
    cluster_means: np.ndarray,
    clusters: list[list[int]],
    round_idx: int,
    ref: np.ndarray | None = None,
    probe=None,
    comp: bool = False,
) -> np.ndarray:
    scores: list[float] = []
    if probe is not None:
        for mean_j in cluster_means:
            scores.append(float(probe(mean_j)))
        if state.n_clients > SCALE_N:
            arr = np.array(scores)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med))) + 1e-9
            thr = med + PROBE_K_MAD * mad
            flagged = [j for j, s in enumerate(scores) if s > thr]
        else:
            best = min(scores)
            flagged = [j for j, s in enumerate(scores) if s > best + PROBE_MARGIN]
    else:
        if ref is None:
            med = np.median(cluster_means, axis=0)
            nmed = med / (np.linalg.norm(med) + 1e-12)
            agree_cos = cluster_means @ nmed / (
                np.linalg.norm(cluster_means, axis=1) + 1e-12)
            agree = cluster_means[agree_cos > 0]
            ref = agree.mean(axis=0) if len(agree) else med
        norm_ref = np.linalg.norm(ref) + 1e-12
        for mean_j in cluster_means:
            cos = float(mean_j @ ref) / ((np.linalg.norm(mean_j) + 1e-12) * norm_ref)
            scores.append(cos)
        flagged = [j for j, s in enumerate(scores) if s < 0.0]
    state.last_info = {"flagged": list(flagged), "scores": scores}
    flagset = set(flagged)
    if comp:
        member_groups: dict[int, list[int]] = {}
        for j, cl in enumerate(clusters):
            for i in cl:
                member_groups.setdefault(i, []).append(j)
        for i, gs in member_groups.items():
            state.rounds[i] = state.rounds.get(i, 0) + 1
            if all(g in flagset for g in gs):
                state.flags[i] = state.flags.get(i, 0) + 1
    else:
        for j, cl in enumerate(clusters):
            for i in cl:
                state.rounds[i] = state.rounds.get(i, 0) + 1
                if j in flagset:
                    state.flags[i] = state.flags.get(i, 0) + 1
    if round_idx + 1 >= state.warmup:
        state.excluded |= state.exclude_now()
    keep = [j for j in range(len(clusters)) if j not in flagged]
    if not keep:
        best_j = int(np.argmin(scores)) if scores else 0
        return cluster_means[best_j]
    return cluster_means[keep].mean(axis=0)
