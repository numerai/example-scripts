# TabPFN Baseline

## Goal
Establish a TabPFN baseline on Numerai (OOF CV) to compare against LGBM baselines.

## What we're testing
- Replace LGBM with TabPFNRegressor using the same CV + embargo setup.
- Feature set: medium (baseline-aligned).
- Primary metric: bmc_last_200_eras vs ender20.
- Secondary: small_bmc_last200 vs small_lgbm_ender20_baseline, corr stats.

## Plan
1) Run OOF CV with TabPFNRegressor on the medium feature set.
2) Save predictions + results under this experiment folder.
3) Compare against small_lgbm_ender20_baseline using show_experiment.py
   (`--baselines-dir agents/baselines`).
4) Iterate on TabPFN params (ensemble size, device) if results are promising.

## Notes
- Requires tabpfn installed in the venv: `.venv/bin/pip install tabpfn`.
- If runtime is heavy, reduce n_splits or the feature set, or add TabPFN params.
- Training folds are downsampled to 50k rows to respect TabPFN's supported limit.
- Roadblocks encountered: gated Hugging Face model access (terms + auth), and
  CPU-only safety limit (>1000 samples) unless overrides/GPU are used.
- Config: `agents/experiments/tabpfn_small_baseline/configs/tabpfn_small_baseline.py`.
- Status: parked until gated model access + CPU override/GPU are available;
  medium-feature rerun pending once unblocked.
  - Latest run failed with: `TABPFN_ALLOW_CPU_LARGE_DATASET` not set.
