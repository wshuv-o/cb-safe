# CB-SAFE

Code, configurations, result files, and figures for **"Recovering Byzantine Robustness
from Code-Based Post-Quantum Secure Aggregation by Temporal Group Testing"** by Md
Wahiduzzaman Suva and Esm-e Moula Chowdhury Abha.

CB-SAFE is a secure-aggregation protocol for federated learning (FL) in which the
key-establishment step is a replaceable key-encapsulation mechanism (KEM). It runs
Bonawitz-style double masking within small client clusters and is instantiated with the
code-based KEM HQC or the lattice KEM ML-KEM. CB-SAFE+ is a defense against untargeted
poisoning built on it: it re-randomizes the clusters every round and accumulates
per-client suspicion from root-anchored flags on the revealed cluster sums (temporal
group testing), so the server can identify and exclude malicious clients without
observing any individual update.

## Repository layout

```
run.py                 run an experiment from its config file
analyze.py             regenerate tables, statistics, and figures from the results
configs/
  datasets/            one file per dataset (partitioning, batch size, result folder)
  methods.yaml         every aggregation method and CB-SAFE+ variant
  experiments/         one file per experiment in the paper
src/
  crypto/              KEM wrapper (HQC, ML-KEM via liboqs), masking, Shamir sharing
  aggregation/         cluster secure aggregation, robust rules, CB-SAFE+ detector, FedGT
  adversary/           label-flip, sign-flip, and backdoor attacks
  federated/           datasets, models, and the FL simulation loop
  pipeline/            config loading, job expansion, data preparation, experiment tasks
analysis/              table, statistics, and figure scripts (driven by analyze.py)
scripts/check_env.py   KEM round-trip and cost benchmark
results/               one CSV per run; see results/README.md
```

## Experiments

Each experiment in the paper is one config file. `run.py` expands it into jobs, one per
result file, and skips jobs whose result already exists, so a run can be interrupted and
resumed, and split across machines.

| Config | Paper result |
|---|---|
| `main_grid.yaml` | Sign-flip, label-flip, backdoor, and ablation results on CIFAR-10, FashionMNIST, EMNIST |
| `edgeiiot_grid.yaml` | The same on Edge-IIoTset, including its no-attack rows |
| `no_attack.yaml` | No-attack reference (Table III) |
| `cluster_size.yaml` | Privacy-robustness trade-off over cluster size c = 1, 3, 5 |
| `gamma_sweep.yaml` | Sign-flip amplification sweep (Fig. 4, Table VII) |
| `duty_sweep.yaml` | Duty-cycled attacker (Table VIII) |
| `threshold_sensitivity.yaml` | Detector-threshold sensitivity (Table X) |
| `scale_n100.yaml`, `scale_n500.yaml` | Client-population scaling (Table II, Fig. 3) |
| `fedgt.yaml` | FedGT baseline with its published decoder |
| `oracle.yaml` | Oracle upper bound (Fig. 6) |
| `overhead.yaml` | Communication and computation cost per KEM (Table I) |
| `utility.yaml` | Secure versus plain aggregation, with dropouts |

```bash
python run.py configs/experiments/main_grid.yaml --dry-run          # list the result files it would write
python run.py configs/experiments/main_grid.yaml                    # run every pending job
python run.py configs/experiments/main_grid.yaml --where dataset=fmnist attack=signflip
python run.py configs/experiments/main_grid.yaml --workers 4 --shard 0   # one of four parallel shards
```

A config lists the parameters shared by all its jobs and one or more grid blocks; every
list-valued key is a grid axis. For example, `duty_sweep.yaml`:

```yaml
task: robustness
dataset: cifar10
attack: signflip
f: 0.2
method: hybrid_ov4
rounds: 50
duty: [0.5, 0.7, 0.8, 0.84, 0.9]
seed: [0, 1, 2]
output: {dir: duty_sweep}
```

## Tables, statistics, and figures

```bash
python analyze.py            # everything
python analyze.py tables     # LaTeX tables -> results/tables/
python analyze.py stats      # paired t-tests with Holm correction -> results/significance*.csv
python analyze.py figures    # figures -> results/figs/
```

These read only the CSVs in `results/`; no training is involved.

## Installation

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

`liboqs-python` needs the native liboqs library built with HQC enabled. On Linux, build
liboqs 0.15.0 with `-DOQS_ENABLE_KEM_HQC=ON` and point `OQS_INSTALL_PATH` at the install
prefix (the default is `~/_oqs`):

```bash
export OQS_INSTALL_PATH=$HOME/_oqs
python scripts/check_env.py   # KEM round-trip for HQC and ML-KEM, writes results/crypto_kem_bench.csv
```

liboqs is needed only for `overhead.yaml` and `utility.yaml`; the robustness experiments
train and aggregate without it.

Datasets: CIFAR-10, FashionMNIST, and EMNIST are downloaded by torchvision into `data/`
(override with `CBSAFE_DATA_ROOT`). Edge-IIoTset uses the `DNN-EdgeIIoT-dataset.csv`
release, available on Kaggle; set `EDGEIIOT_ROOT` to the directory that contains it.
Results are written to `results/` (override with `CBSAFE_OUT`).

## FedGT baseline

`configs/experiments/fedgt.yaml` runs the FedGT detector (Xhemrishi et al., IEEE TIFS
2025) with its published BCJR decoder. The compiled decoder (`BCJR_4_python`) and the
n=30 parity-check matrix (`fedgt_H30.npy`) come from the FedGT authors' code and are not
redistributed here; place them in `src/aggregation/` to rerun FedGT. The FedGT result
files are included in `results/fedgt_faithful/`.

## License

MIT; see [LICENSE](LICENSE).
