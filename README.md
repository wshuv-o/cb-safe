# CB-SAFE

Code-based post-quantum secure aggregation for Byzantine-robust federated learning.

CB-SAFE combines within-cluster masked secure aggregation (HQC / ML-KEM) with
**CB-SAFE+**, a re-randomized temporal group-testing detector that identifies
laundered Byzantine clients which the secure-aggregation boundary hides from
coordinate-wise robust rules. Secure aggregation reveals only per-cluster sums; a
sign-flipping adversary can launder its update so a contaminated cluster mean has
honest magnitude but inverted direction, defeating median / trimmed-mean / Krum /
Bulyan / RFA while they operate correctly. CB-SAFE+ recovers identification from the
aggregate level at no extra leakage.

## Repository layout

```
src/
  federated/     FL simulation, Dirichlet non-IID partitioning, models, data loaders
  aggregation/   robust rules (median, Multi-Krum, Bulyan, RFA, FLTrust), the
                 CB-SAFE+ reputation/hybrid detector, and a faithful reimplementation
                 of FedGT with its real BCJR decoder (BCJR_4_python.dll + fedgt_H30.npy)
  crypto/        HQC / ML-KEM KEM wrappers, ChaCha20 masking, Shamir dropout recovery
  adversary/     sign-flip (laundering), label-flip, backdoor attacks
experiments/
  run_*.py       reproduction runners (write per-round CSVs into results/)
  figures.py     every paper figure as a callable function + CLI dispatcher
  tables.py      every paper LaTeX table as a callable function + CLI dispatcher
results/         per-run CSV logs consumed by figures.py / tables.py
```

## Install

```
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`liboqs-python` (import name `oqs`) is needed only for the KEM overhead/utility
benchmarks and is imported lazily; everything else runs without it. See
`requirements.txt` for the install pointer.

## Reproduce

Run the experiments to (re)generate the per-round CSVs under `results/`:

```
python experiments/run_robustness.py       # sign/label/backdoor sweeps (CIFAR-10, FashionMNIST)
python experiments/run_scale.py            # N=100 / N=500 scaling
python experiments/run_fedgt_faithful.py   # FedGT (real BCJR decoder) baseline at N=30
python experiments/run_overhead.py         # KEM / masking cost (needs liboqs)
python experiments/run_utility.py          # secure-aggregate == plaintext utility check
```

CIFAR-10 and FashionMNIST download automatically (torchvision). EMNIST-balanced
(47-class) and Edge-IIoTset were run on Kaggle with the same simulation core
(`src/federated/kaggle_datasets.py`); their result CSVs are included under
`results/` so the figures and tables reproduce without re-running them.

Then rebuild every figure and table from the CSVs:

```
python experiments/figures.py all    # or a subset: dynamics convergence backdoor cdial suspicion architecture
python experiments/tables.py all     # or a subset: signflip ablation labelflip scale
```

Figures are written to `results/figs/`, LaTeX tables to `results/tables/`.

## Datasets

CIFAR-10, FashionMNIST, EMNIST-balanced (47 classes), and Edge-IIoTset (tabular IoT
intrusion detection), each partitioned non-IID with Dirichlet(alpha = 0.5).
