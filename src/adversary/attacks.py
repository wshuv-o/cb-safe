from __future__ import annotations

import numpy as np
import torch

SIGNFLIP_GAMMA = 5.0
BACKDOOR_TARGET = 0
BACKDOOR_POISON_FRAC = 0.5
TRIGGER_SIZE = 3


def flip_labels(y: torch.Tensor, n_classes: int = 10) -> torch.Tensor:
    return (n_classes - 1) - y


def signflip_delta(delta: np.ndarray, gamma: float = SIGNFLIP_GAMMA) -> np.ndarray:
    return (-gamma * delta).astype(delta.dtype)


def stamp_trigger(x: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    x[..., -TRIGGER_SIZE:, -TRIGGER_SIZE:] = 2.5
    return x


def poison_batch(x: torch.Tensor, y: torch.Tensor, frac: float = BACKDOOR_POISON_FRAC,
                 target: int = BACKDOOR_TARGET) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(len(y) * frac)
    if n == 0:
        return x, y
    x, y = x.clone(), y.clone()
    x[:n] = stamp_trigger(x[:n])
    y[:n] = target
    return x, y
