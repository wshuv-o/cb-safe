# Supplementary runs

Runs that are not used by any table or figure in the paper. They are kept for
completeness; no config in `configs/experiments/` produces them.

- `robust_*_fedgt_*` (CIFAR-10, `fmnist/`, `kaggle/emnist/`, `kaggle/edgeiiot/`):
  FedGT-style group testing inside the shared training harness. The reported FedGT
  results use the published decoder instead (`results/fedgt_faithful/`); the harness
  version is used only for the no-attack rows.
- `robust_signflip_reputation_tf_ov4_*`: a trust-free CB-SAFE+ variant not reported.
- `scale/`: N = 500 runs at 30 rounds and FedGT-harness runs; the reported N = 500
  results are the 50-round runs in `results/scale50/`.
- `scale50/`: N = 100 runs at 50 rounds (the reported N = 100 column uses 30 rounds),
  seed 1 at N = 500 for FLTrust (the reported seeds are 0, 2, 3), and FedGT-harness runs.
- `temporal_check/`: a three-run check of temporal overlap on CIFAR-10.
- `laundering_gamma_geom.csv`, `laundering_gamma_sweep.csv`: an early probe of the
  laundering geometry.
- `summary_robustness.csv`, `tables_master_robustness.csv`, `tables_overhead_wide.csv`:
  summaries from earlier analysis scripts.
