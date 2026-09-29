"""CIFAR-10 loading and Dirichlet non-IID partitioning across clients."""

from __future__ import annotations

import os

# Must be set before numpy/torch import anywhere in the process (Windows OpenMP clash).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATA_ROOT = os.environ.get("CBSAFE_DATA_ROOT") or os.path.join(os.path.dirname(__file__), "..", "..", "data")
# On a read-only repo mount (e.g. Kaggle /kaggle/input) fall back to a writable dir
# so torchvision downloads (EMNIST etc.) do not crash with OSError [Errno 30].
try:
    os.makedirs(DATA_ROOT, exist_ok=True)
except OSError:
    _base = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
    DATA_ROOT = os.path.join(_base, "data")
    os.makedirs(DATA_ROOT, exist_ok=True)
# FashionMNIST lives on C: because D: is out of space (see README status note)
FMNIST_ROOT = os.path.join(os.path.expanduser("~"), "fl_data")

NORM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
TRAIN_TF = transforms.Compose([transforms.ToTensor(), NORM])
TEST_TF = transforms.Compose([transforms.ToTensor(), NORM])
FMNIST_TF = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])


class _FastDS(torch.utils.data.Dataset):
    """Dataset whose samples are already transformed and held as plain tensors, so
    __getitem__ is a tensor index rather than a per-sample PIL decode+transform.
    Exposes `.targets` (in original order) for the Dirichlet partitioner."""

    def __init__(self, X: torch.Tensor, targets: torch.Tensor):
        self.X = X
        self.targets = targets

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.targets[i]


def _materialize(ds) -> "_FastDS":
    """Run `ds`'s (slow, PIL-based) transform ONCE and cache the result as tensors.
    Produces bit-identical samples to `ds` in the same index order, so downstream
    partitions/shuffles/batches are unchanged -- only far faster to iterate."""
    loader = DataLoader(ds, batch_size=4096, shuffle=False, num_workers=0)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    return _FastDS(torch.cat(xs), torch.cat(ys))


def load_cifar10() -> tuple[datasets.CIFAR10, datasets.CIFAR10]:
    train = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=TRAIN_TF)
    test = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=TEST_TF)
    return train, test


def load_dataset(name: str):
    """Return (train, test) for a supported dataset name, pre-materialized to tensors
    (transform applied once) so per-round iteration is a tensor index, not a per-sample
    PIL decode. Samples are bit-identical to the raw torchvision datasets."""
    if name == "cifar10":
        train, test = load_cifar10()
    elif name == "fmnist":
        train = datasets.FashionMNIST(FMNIST_ROOT, train=True, download=True,
                                      transform=FMNIST_TF)
        test = datasets.FashionMNIST(FMNIST_ROOT, train=False, download=True,
                                     transform=FMNIST_TF)
    elif name == "emnist":
        # EMNIST-balanced (47 classes). Data root on D: (C: is out of space).
        tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.1751,), (0.3332,))])
        train = datasets.EMNIST(DATA_ROOT, split="balanced", train=True,
                                download=True, transform=tf)
        test = datasets.EMNIST(DATA_ROOT, split="balanced", train=False,
                               download=True, transform=tf)
    else:
        raise ValueError(f"unknown dataset {name!r}")
    return _materialize(train), _materialize(test)


def dirichlet_partition(labels: np.ndarray, n_clients: int, alpha: float, seed: int,
                        min_size: int = 20) -> list[np.ndarray]:
    """Standard Dirichlet(alpha) label-skew partition; resamples until every client
    has at least `floor` samples.

    The floor relaxes for large populations: with N clients over |labels| samples,
    each client averages |labels|/N samples, and requiring all of them to clear a
    fixed 20 under a skewed Dirichlet is unsatisfiable once N is large (e.g. ~60
    samples/client at N=1000), which made the original `while True` loop spin
    forever. We cap the floor at avg/4, bound the retries, and fall back to the
    best partition seen. For N<=500 (avg/4 >= 30) the floor stays 20 and the first
    satisfying draw is returned exactly as before, so existing results are unchanged."""
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
    return [np.array(sorted(ix)) for ix in best[1]]  # best effort after retry cap


def client_loaders(train_set, partitions: list[np.ndarray], batch_size: int = 64) -> list[DataLoader]:
    return [
        DataLoader(Subset(train_set, ix.tolist()), batch_size=batch_size, shuffle=True,
                   num_workers=0, drop_last=False)
        for ix in partitions
    ]
