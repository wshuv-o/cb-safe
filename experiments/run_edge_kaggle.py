"""Edge-IIoTset-only Kaggle runner (all 12 methods, sign-flip 3 seeds + label-flip).

This is the ONLY dataset that must run on Kaggle: its source CSV
(DNN-EdgeIIoT-dataset.csv, ~2 GB) is Kaggle-hosted. The three image datasets
(CIFAR-10, FashionMNIST, EMNIST) are handled locally, so this bundle scopes the
grid to Edge-IIoTset only -> ~144 runs, tabular and fast (~2-3 min each) -> one
T4 session covers it.

Upload the bundle as a Kaggle dataset, ADD the Edge-IIoTset dataset (the one with
DNN-EdgeIIoT-dataset.csv), enable a T4 GPU + Internet, and run this. Writes one
CSV per (attack, method, f, seed) under /kaggle/working/results/kaggle/edgeiiot/.
Resumable: re-run to continue; finished CSVs are skipped.
"""

import os

# Scope run_all_kaggle's grid to Edge-IIoTset before importing it.
os.environ["CBSAFE_DATASETS"] = "edgeiiot"

import run_all_kaggle  # noqa: E402


if __name__ == "__main__":
    run_all_kaggle.main()
