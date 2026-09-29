"""Small CNN for CIFAR-10 (~300k parameters) and flat-vector state helpers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int = 10, in_channels: int = 3, img_size: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        feat = 64 * (img_size // 4) ** 2
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(feat, 64), nn.ReLU(), nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class TabularMLP(nn.Module):
    """MLP for tabular intrusion-detection data (Edge-IIoTset)."""

    def __init__(self, n_classes: int, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# dataset name -> {kind, n_classes, and constructor kwargs}. `in_dim` for tabular
# datasets is discovered at load time and injected via set_tabular_dim().
MODEL_SHAPES = {
    "cifar10": {"kind": "cnn", "n_classes": 10, "in_channels": 3, "img_size": 32},
    "fmnist": {"kind": "cnn", "n_classes": 10, "in_channels": 1, "img_size": 28},
    "emnist": {"kind": "cnn", "n_classes": 47, "in_channels": 1, "img_size": 28},
    "edgeiiot": {"kind": "mlp", "n_classes": 15, "in_dim": 63},
}


def set_tabular_dim(dataset: str, n_classes: int, in_dim: int) -> None:
    """Edge-IIoTset's exact feature count / class count depend on preprocessing;
    the loader calls this once so make_model builds a correctly-sized MLP."""
    MODEL_SHAPES[dataset]["n_classes"] = n_classes
    MODEL_SHAPES[dataset]["in_dim"] = in_dim


def n_classes_of(dataset: str) -> int:
    return MODEL_SHAPES[dataset]["n_classes"]


def make_model(dataset: str = "cifar10") -> nn.Module:
    shape = dict(MODEL_SHAPES[dataset])
    kind = shape.pop("kind")
    if kind == "mlp":
        return TabularMLP(**shape)
    return SmallCNN(**shape)


def flat_params(model: nn.Module) -> np.ndarray:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).cpu().numpy().astype(np.float32)


def load_flat_params(model: nn.Module, flat: np.ndarray) -> None:
    t = torch.from_numpy(flat.astype(np.float32))
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(t[offset : offset + n].reshape(p.shape).to(p.device))
            offset += n


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
