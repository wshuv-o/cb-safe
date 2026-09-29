"""Dataset loading, partitioning, and the server root set.

Clients receive a Dirichlet(alpha) label-skew partition of the training set. Methods
that use a server root dataset (CB-SAFE+, FLTrust, the FedGT harness) get a small
root set held out from every client. Three root policies reproduce how the reported
runs were prepared:

  per_method       root set drawn from the whole training set and removed from the
                   clients only when the method uses it (CIFAR-10, FashionMNIST,
                   EMNIST grids and sweeps)
  always_exclude   root set drawn from the whole training set and always removed from
                   the clients; the server loader is passed only to methods that use it
                   (client-population scaling, FedGT)
  partition_pool   root set drawn from the partitioned pool, capped at a quarter of it,
                   and always removed from the clients (Edge-IIoTset grid)
  none             no root set (oracle upper bound)
"""

from __future__ import annotations

import os

import numpy as np
from torch.utils.data import DataLoader, Subset

from ..federated import data as fl_data


def load(ds_cfg: dict, seed: int):
    """Return (train, test, labels) for a dataset config."""
    name = ds_cfg["name"]
    if ds_cfg.get("loader") == "edgeiiot_csv":
        from ..federated import kaggle_datasets as kd
        root = os.environ.get("EDGEIIOT_ROOT", ds_cfg.get("search_root"))
        path = kd.find_edgeiiot_csv(root)
        if not path:
            raise FileNotFoundError(
                "Edge-IIoTset CSV (DNN-EdgeIIoT-dataset.csv) not found; set EDGEIIOT_ROOT "
                "to the directory that contains it")
        train, test, labels = kd.load_edgeiiot(path, seed=seed)
        return train, test, np.asarray(labels)
    train, test = fl_data.load_dataset(name)
    return train, test, np.array(train.targets)


def prepare(ds_cfg: dict, seed: int, n_clients: int, policy: str, uses_root: bool,
            root_size: int = 200, batch_size: int | None = None):
    """Return (client_loaders, test_loader, server_loader_or_None).

    batch_size overrides the dataset's client batch size (the FedGT runs used 64 on
    every dataset)."""
    train, test, labels = load(ds_cfg, seed)
    batch = batch_size or ds_cfg.get("batch_size", 64)
    alpha = ds_cfg.get("alpha", 0.5)
    parts = fl_data.dirichlet_partition(labels, n_clients, alpha, seed)
    rng = np.random.default_rng(seed + 99)

    root = None
    if policy == "per_method":
        if uses_root:
            root = set(rng.choice(len(train), size=root_size, replace=False).tolist())
    elif policy == "always_exclude":
        root = set(rng.choice(len(train), size=root_size, replace=False).tolist())
    elif policy == "partition_pool":
        pool = np.concatenate(parts)
        root = set(rng.choice(pool, size=min(root_size, len(pool) // 4), replace=False).tolist())
    elif policy != "none":
        raise ValueError(f"unknown root policy '{policy}'")

    server_dl = None
    if root is not None:
        parts = [np.array([i for i in p if i not in root]) for p in parts]
        if uses_root:
            server_dl = DataLoader(Subset(train, sorted(root)), batch_size=64, shuffle=True)

    if policy == "partition_pool" or policy == "none":
        client_dls = [DataLoader(Subset(train, ix.tolist()), batch_size=batch, shuffle=True)
                      for ix in parts]
    else:
        client_dls = fl_data.client_loaders(train, parts, batch)
    test_dl = DataLoader(test, batch_size=512)
    return client_dls, test_dl, server_dl
