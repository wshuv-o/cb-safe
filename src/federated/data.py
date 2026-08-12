from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATA_ROOT = os.environ.get("CBSAFE_DATA_ROOT") or os.path.join(os.path.dirname(__file__), "..", "..", "data")
try:
    os.makedirs(DATA_ROOT, exist_ok=True)
except OSError:
    _base = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
    DATA_ROOT = os.path.join(_base, "data")
    os.makedirs(DATA_ROOT, exist_ok=True)
FMNIST_ROOT = os.path.join(os.path.expanduser("~"), "fl_data")

NORM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
TRAIN_TF = transforms.Compose([transforms.ToTensor(), NORM])
TEST_TF = transforms.Compose([transforms.ToTensor(), NORM])
FMNIST_TF = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])


def load_cifar10() -> tuple[datasets.CIFAR10, datasets.CIFAR10]:
    train = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=TRAIN_TF)
    test = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=TEST_TF)
    return train, test


def load_dataset(name: str):
    if name == "cifar10":
        return load_cifar10()
    if name == "fmnist":
        train = datasets.FashionMNIST(FMNIST_ROOT, train=True, download=True,
                                      transform=FMNIST_TF)
        test = datasets.FashionMNIST(FMNIST_ROOT, train=False, download=True,
                                     transform=FMNIST_TF)
        return train, test
    if name == "emnist":
        tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.1751,), (0.3332,))])
        train = datasets.EMNIST(DATA_ROOT, split="balanced", train=True,
                                download=True, transform=tf)
        test = datasets.EMNIST(DATA_ROOT, split="balanced", train=False,
                               download=True, transform=tf)
        return train, test
    raise ValueError(f"unknown dataset {name!r}")


def dirichlet_partition(labels: np.ndarray, n_clients: int, alpha: float, seed: int,
                        min_size: int = 20) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    n_classes = int(labels.max()) + 1
    floor = min(min_size, max(1, len(labels) // n_clients // 4))
    best: tuple[int, list[list[int]]] | None = None
    for _ in range(200):
        idx_per_client: list[list[int]] = [[] for _ in range(n_clients)]
        for k in range(n_classes):
            idx_k = np.where(labels == k)[0]
            rng.shuffle(idx_k)
            props = rng.dirichlet(np.repeat(alpha, n_clients))
            cuts = (np.cumsum(props) * len(idx_k)).astype(int)[:-1]
            for cid, part in enumerate(np.split(idx_k, cuts)):
                idx_per_client[cid].extend(part.tolist())
        worst = min(len(ix) for ix in idx_per_client)
        if best is None or worst > best[0]:
            best = (worst, idx_per_client)
        if worst >= floor:
            return [np.array(sorted(ix)) for ix in idx_per_client]
    return [np.array(sorted(ix)) for ix in best[1]]


def client_loaders(train_set, partitions: list[np.ndarray], batch_size: int = 64) -> list[DataLoader]:
    return [
        DataLoader(Subset(train_set, ix.tolist()), batch_size=batch_size, shuffle=True,
                   num_workers=0, drop_last=False)
        for ix in partitions
    ]
