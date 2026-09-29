"""Loaders for the two Kaggle-hosted generalization datasets.

  emnist    - EMNIST-balanced (47 classes, handwritten), auto-downloaded by
              torchvision; a larger, harder label space than CIFAR/FashionMNIST.
  edgeiiot  - Edge-IIoTset IoT/IIoT intrusion detection (tabular). A robust CSV
              loader that adapts to the standard "DNN-EdgeIIoT-dataset.csv":
              drops known leakage/identifier columns, label-encodes the multiclass
              attack type, coerces + standardizes features, and returns tensors.

Both return (train_ds, test_ds, train_labels) so the existing Dirichlet
partitioner and client-loader code apply unchanged.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms

from . import models

# ---------------------------------------------------------------- EMNIST -----

EMNIST_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1751,), (0.3332,)),
])


def load_emnist(root: str):
    """EMNIST-balanced (47 classes). Tries torchvision download; if that fails
    (its NIST URL is historically flaky), falls back to a Kaggle-hosted CSV in the
    common `emnist-balanced-{train,test}.csv` format (label, then 784 pixels)."""
    try:
        train = datasets.EMNIST(root, split="balanced", train=True, download=True,
                                transform=EMNIST_TF)
        test = datasets.EMNIST(root, split="balanced", train=False, download=True,
                               transform=EMNIST_TF)
        return train, test, np.array(train.targets)
    except Exception as exc:  # noqa: BLE001
        tr = _find_emnist_csv("train")
        te = _find_emnist_csv("test")
        if not (tr and te):
            raise RuntimeError(
                "EMNIST torchvision download failed and no EMNIST CSV found under "
                "/kaggle/input. Add a Kaggle dataset such as 'crawford/emnist'."
            ) from exc
        return _load_emnist_csv(tr, te)


def _find_emnist_csv(split: str, search_root: str = "/kaggle/input") -> str | None:
    if not os.path.isdir(search_root):
        return None
    for dirpath, _dirs, files in os.walk(search_root):
        for fn in files:
            low = fn.lower()
            if "emnist" in low and "balanced" in low and split in low and low.endswith(".csv"):
                return os.path.join(dirpath, fn)
    return None


def _load_emnist_csv(train_csv: str, test_csv: str):
    import pandas as pd

    def to_ds(path):
        arr = pd.read_csv(path, header=None).to_numpy()
        y = arr[:, 0].astype(np.int64)
        x = arr[:, 1:].astype(np.float32).reshape(-1, 1, 28, 28) / 255.0
        x = (x - 0.1751) / 0.3332
        return TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), y

    train, ytr = to_ds(train_csv)
    test, _ = to_ds(test_csv)
    return train, test, ytr


# ------------------------------------------------------------- Edge-IIoTset ---

# columns that leak the label or identify the flow; dropped if present
_EDGEIIOT_DROP = [
    "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4", "http.file_data", "http.request.full_uri",
    "icmp.transmit_timestamp", "http.request.uri.query", "tcp.options",
    "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg",
    "Attack_label",  # binary target; we use the multiclass Attack_type
]
_LABEL_COL = "Attack_type"


def load_edgeiiot(csv_path: str, seed: int = 0, test_frac: float = 0.2,
                  max_rows: int = 120_000):
    """Load and preprocess Edge-IIoTset from the standard DNN CSV.
    Returns (train_ds, test_ds, train_labels) and registers the MLP input dim."""
    import pandas as pd

    df = pd.read_csv(csv_path, low_memory=False)
    if len(df) > max_rows:  # subsample for tractable FL rounds, class-stratified
        # Built by concatenating per-group samples rather than groupby().apply():
        # pandas 3 excludes the grouping column from an apply result, which silently
        # dropped Attack_type and made the next line raise KeyError. Older pandas only
        # warned. This form keeps every column on any version.
        per = max(1, max_rows // df[_LABEL_COL].nunique())
        df = pd.concat([g.sample(min(len(g), per), random_state=seed)
                        for _, g in df.groupby(_LABEL_COL, sort=False)])
    df = df.dropna(axis=0, how="any").reset_index(drop=True)

    y_raw = df[_LABEL_COL].astype(str)
    classes = sorted(y_raw.unique())
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y = y_raw.map(cls_to_idx).to_numpy().astype(np.int64)

    drop = [c for c in _EDGEIIOT_DROP + [_LABEL_COL] if c in df.columns]
    X = df.drop(columns=drop)
    # Label-encode any remaining non-numeric column, then coerce the rest.
    # Testing `dtype == object` missed them on pandas 3, where text columns carry
    # StringDtype: they reached the float cast unencoded and raised on values like
    # 'GET'. A numeric-dtype test is correct on every version.
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype("category").cat.codes
    X = X.apply(lambda s: s.astype(np.float32)).to_numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    X = ((X - mu) / sd).astype(np.float32)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    n_test = int(len(X) * test_frac)
    Xtr, ytr = X[n_test:], y[n_test:]
    Xte, yte = X[:n_test], y[:n_test]

    models.set_tabular_dim("edgeiiot", n_classes=len(classes), in_dim=X.shape[1])
    train = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    test = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
    return train, test, ytr


def find_edgeiiot_csv(search_root: str | None = None) -> str | None:
    """Locate the Edge-IIoTset DNN CSV.

    Searches CBSAFE_EDGEIIOT_DIR if set, then the repository's data/ directory, then
    the Kaggle input mount. The local paths are what let the Edge-IIoT experiments run
    off Kaggle: the mount does not exist on a workstation, so the presence probe
    returned None and the dataset was skipped without an error.
    """
    prefer = "DNN-EdgeIIoT-dataset.csv"
    if search_root is not None:
        roots = [search_root]
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        repo = os.path.dirname(os.path.dirname(here))
        roots = [os.environ.get("CBSAFE_EDGEIIOT_DIR"),
                 os.path.join(repo, "data"),
                 "/kaggle/input"]
    fallback = None
    for root in [r for r in roots if r and os.path.isdir(r)]:
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                if fn == prefer:
                    return os.path.join(dirpath, fn)
                if fn.lower().endswith(".csv") and "edge" in fn.lower() \
                        and fallback is None:
                    fallback = os.path.join(dirpath, fn)
    return fallback
