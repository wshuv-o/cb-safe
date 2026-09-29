# CB-SAFE

Code, result files, and figures for **"Recovering Byzantine Robustness from Code-Based
Post-Quantum Secure Aggregation by Temporal Group Testing"** by Md Wahiduzzaman Suva and
Esm-e Moula Chowdhury Abha.

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
src/
  crypto/        KEM wrapper (HQC, ML-KEM via liboqs), pairwise masking, Shamir sharing
  aggregation/   cluster secure aggregation, robust rules, CB-SAFE+ detector, FedGT baseline
  adversary/     label-flip, sign-flip, and backdoor attacks
  federated/     datasets, models, and the FL simulation loop
experiments/     run scripts, table builders, and plotting scripts
scripts/         environment check (KEM round-trip and cost benchmark)
results/         per-run CSVs behind every table and figure
  tables/        LaTeX tables generated from the CSVs
  figs/          figures generated from the CSVs
```

Result folders: CIFAR-10 runs are in `results/`, FashionMNIST in `results/fmnist/`,
EMNIST and Edge-IIoTset in `results/kaggle/`, the client-population runs in
`results/scale/` (N=100) and `results/scale50/` (N=500), and the sweeps in
`results/gamma_sweep/`, `results/duty_sweep/`, and `results/delta_sweep/`. Each CSV
holds one run (one attack, malicious fraction, aggregation rule, and seed), with
per-round accuracy and exclusion counts.

## Installation

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

`liboqs-python` needs the native liboqs library built with HQC enabled. On Linux,
build liboqs 0.15.0 with `-DOQS_ENABLE_KEM_HQC=ON` and point `OQS_INSTALL_PATH` at the
install prefix (the default is `~/_oqs`):

```bash
export OQS_INSTALL_PATH=$HOME/_oqs
python scripts/check_env.py   # KEM round-trip for HQC and ML-KEM, writes results/crypto_kem_bench.csv
```

liboqs is needed only for the cryptographic runs (overhead and secure/plain
equivalence). The robustness runs train and aggregate without it.

Datasets: CIFAR-10, FashionMNIST, and EMNIST are downloaded by torchvision into
`data/` (override with `CBSAFE_DATA_ROOT`). Edge-IIoTset uses the
`DNN-EdgeIIoT-dataset.csv` release, available on Kaggle.

## Reproducing the results

All scripts are run from the repository root. Results are written to `results/`
(override with `CBSAFE_OUT`), and existing CSVs are skipped, so runs are resumable.

```bash
# Quick correctness check of the crypto and aggregation stack (no training)
python experiments/smoke_test.py

# Communication and computation overhead per KEM (Table I)
python experiments/run_overhead.py

# Secure-versus-plain aggregation equivalence and dropout recovery
python experiments/run_utility.py

# A single robustness run, e.g. CB-SAFE+ under sign-flip at f=0.2
python experiments/run_robustness.py --attack signflip --f 0.2 --aggregator hybrid \
    --overlap 4 --temporal --rounds 50 --seed 0 --dataset cifar10

# The full 50-round grid on CIFAR-10, FashionMNIST, and EMNIST (sharded)
python experiments/run_r50.py --workers 4 --shard 0   # repeat for shards 1..3

# Edge-IIoTset (tabular; the source CSV is hosted on Kaggle)
python experiments/run_edge_kaggle.py

# Sweeps: amplification gamma and duty cycle, detector thresholds
python experiments/run_sweeps.py
python experiments/run_delta_sweep.py

# Client-population scaling on FashionMNIST: N=100 (30 rounds) and N=500 (50 rounds)
python experiments/run_scale.py --ns 100
python experiments/run_scale.py --ns 500 --rounds 50 --outdir scale50
```

Tables and figures are regenerated from the CSVs without retraining:

```bash
python experiments/build_bigtables.py      # sign-flip, label-flip, ablation, no-attack tables
python experiments/build_scale_table.py    # N=100 / N=500 scaling table
python experiments/stats.py                # paired t-tests with Holm correction
python experiments/plot_dynamics.py        # and the other plot_*.py scripts
```

## FedGT baseline

`src/aggregation/fedgt_faithful.py` runs the FedGT detector (Xhemrishi et al., IEEE
TIFS 2025) with its published BCJR decoder. The compiled decoder
(`BCJR_4_python`) and the n=30 parity-check matrix (`fedgt_H30.npy`) come from the
FedGT authors' code and are not redistributed here; place them in
`src/aggregation/` to rerun FedGT. The FedGT result CSVs are included in
`results/fedgt_faithful/`.

## License

MIT; see [LICENSE](LICENSE).
