# MLP Baseline

## Goal
Establish a small MLP baseline on Numerai using the standard OOF CV pipeline.

## What we're testing
- MLPRegressor (scikit-learn) with two hidden layers (128, 64) and Adam.
- Feature set: medium (baseline-aligned).
- CV: expanding era splits with embargo.
- Training folds downsampled to 100k rows for feasibility.
- Primary metric: bmc_last_200_eras vs ender20.
- Secondary: small_bmc_last200 vs small_lgbm_ender20_baseline, corr stats.

## Plan
1) Run OOF CV with the MLP config below.
2) Save predictions + results under this experiment folder.
3) Compare vs small_lgbm_ender20_baseline using show_experiment.py.
4) If promising, scale by increasing hidden sizes/epochs or sample size.

## Notes
- Requires scikit-learn in the venv: `.venv/bin/pip install scikit-learn`.
- Config: `agents/experiments/mlp_small_baseline/configs/mlp_small_baseline.py`.
- Run with `--output-dir agents/experiments/mlp_small_baseline`.
