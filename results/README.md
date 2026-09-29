# Results

One CSV per run, with one row per training round (accuracy, exclusions, and detector
statistics). Every file outside `supplementary/` is produced by a config in
`configs/experiments/`; `python run.py <config> --dry-run` lists the files a config
writes.

| Folder | Contents | Config |
|---|---|---|
| `results/` (top level) | CIFAR-10 runs, cluster-size runs, oracle | `main_grid`, `no_attack`, `cluster_size`, `oracle` |
| `fmnist/` | FashionMNIST runs | `main_grid`, `no_attack`, `oracle` |
| `kaggle/emnist/` | EMNIST runs | `main_grid`, `no_attack`, `oracle` |
| `kaggle/edgeiiot/` | Edge-IIoTset runs | `edgeiiot_grid`, `oracle` |
| `gamma_sweep/` | sign-flip amplification sweep | `gamma_sweep` |
| `duty_sweep/` | duty-cycled attacker | `duty_sweep` |
| `delta_sweep/` | detector-threshold sensitivity | `threshold_sensitivity` |
| `scale/` | N = 100, 30 rounds | `scale_n100` |
| `scale50/` | N = 500, 50 rounds | `scale_n500` |
| `fedgt_faithful/` | FedGT with its published decoder | `fedgt` |
| `overhead.csv` | cost per KEM | `overhead` |
| `utility_acc.csv`, `secure_equivalence.csv` | secure versus plain aggregation | `utility` |
| `crypto_kem_bench.csv` | KEM primitive sizes and timings | `scripts/check_env.py` |
| `significance*.csv`, `summary_multiseed.csv` | statistics | `analyze.py stats` |
| `tables/` | LaTeX tables | `analyze.py tables` |
| `figs/` | figures | `analyze.py figures` |

File names encode the run: `robust_<attack>_<method>[_N<n>]_f<ff>_c<c>[_g<ggg>][_d<ddd>]_s<seed>.csv`,
where `<method>` is a key of `configs/methods.yaml`, `f<ff>` is the malicious fraction
in percent, `c<c>` the cluster size, `_N` the number of clients when it is not 30, `_g`
the sign-flip amplification times ten when it is not 5, and `_d` the attack duty cycle
in percent when it is not 1.

## Notes on how the runs were produced

- **Overlap mode of CB-SAFE+.** CB-SAFE+ spreads its four overlapping tests across
  rounds (temporal overlap, one partition per round). The main grids, the duty-cycle
  sweep, Edge-IIoTset, and N = 500 use temporal overlap. The no-attack rows, the
  CB-SAFE+ points of the amplification sweep, the threshold-sensitivity sweep, and the
  N = 100 runs used four simultaneous partitions per round. The configs record this with
  `temporal: false`, so rerunning them reproduces the reported files. The mode is
  visible in the per-round `n_dirty` and `n_flagged` columns: with temporal overlap they
  cannot exceed the number of clusters in one partition.
- **Edge-IIoTset.** The root set is drawn from the partitioned pool
  (`root_policy: partition_pool`), as in the Kaggle pipeline used for this dataset.
- **Duty-cycle table.** `tables/table_duty.tex` is written by hand from `duty_sweep/`.
- **`supplementary/`** holds runs that no reported result uses; see its README.
