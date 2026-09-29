"""Poisoning attacks for the robustness evaluation.

Three standard attack types at malicious fraction f:
  - labelflip: malicious clients train on labels y -> 9 - y (untargeted data poisoning)
  - signflip:  malicious clients send -gamma * (honest delta) (model poisoning)
  - backdoor:  malicious clients train partly on trigger-stamped images relabeled to a
               target class; success metric is attack success rate (ASR) on a fully
               triggered test set excluding true-target samples.
"""

from __future__ import annotations

import numpy as np
import torch

SIGNFLIP_GAMMA = 5.0
BACKDOOR_TARGET = 0
BACKDOOR_POISON_FRAC = 0.5
TRIGGER_SIZE = 3  # white square, bottom-right corner


def flip_labels(y: torch.Tensor, n_classes: int = 10) -> torch.Tensor:
    return (n_classes - 1) - y


def signflip_delta(delta: np.ndarray, gamma: float = SIGNFLIP_GAMMA) -> np.ndarray:
    return (-gamma * delta).astype(delta.dtype)


def stamp_trigger(x: torch.Tensor) -> torch.Tensor:
    """Stamp a white TRIGGER_SIZE x TRIGGER_SIZE square in the bottom-right corner.
    Expects normalized CHW image batches or single images; stamps max channel value 2.5
    (approximately white after CIFAR-10 normalization)."""
    x = x.clone()
    x[..., -TRIGGER_SIZE:, -TRIGGER_SIZE:] = 2.5
    return x


def poison_batch(x: torch.Tensor, y: torch.Tensor, frac: float = BACKDOOR_POISON_FRAC,
                 target: int = BACKDOOR_TARGET) -> tuple[torch.Tensor, torch.Tensor]:
    """Stamp + relabel the first `frac` of a batch (batches are already shuffled)."""
    n = int(len(y) * frac)
    if n == 0:
        return x, y
    x, y = x.clone(), y.clone()
    x[:n] = stamp_trigger(x[:n])
    y[:n] = target
    return x, y
