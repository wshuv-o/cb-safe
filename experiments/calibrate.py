"""Download CIFAR-10 and time two full-size FedAvg rounds (sweep calibration)."""

import _bootstrap  # noqa: F401

import numpy as np
from torch.utils.data import DataLoader

from src.federated import data
from src.federated.simulation import Config, run

cfg = Config(n_clients=30, rounds=2, aggregation="fedavg", seed=0)
train, test = data.load_cifar10()
parts = data.dirichlet_partition(np.array(train.targets), cfg.n_clients, cfg.alpha, cfg.seed)
sizes = sorted(len(p) for p in parts)
print(f"partition sizes: min={sizes[0]} median={sizes[len(sizes)//2]} max={sizes[-1]}")
client_dls = data.client_loaders(train, parts, cfg.batch_size)
test_dl = DataLoader(test, batch_size=512, num_workers=0)
hist = run(cfg, client_dls, test_dl)
print("round times:", [round(h["t_round_s"], 1) for h in hist])
