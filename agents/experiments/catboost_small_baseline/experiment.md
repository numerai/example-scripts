# CatBoost Small Baseline

## Goal
Establish a CatBoost baseline on Numerai using OOF CV with embargo.

## What we're testing
- CatBoostRegressor with the medium feature set to match the baseline.
- Primary metric: bmc_last_200_eras vs ender20.
- Secondary: small_bmc_last200 vs small_lgbm_ender20_baseline, corr stats.

## Plan
1) Run OOF CV with CatBoost on the medium feature set (baseline-aligned).
2) Save predictions + results under this experiment folder.
3) Compare against `small_lgbm_ender20_baseline` from `agents/baselines/`.
4) If promising, scale iterations while keeping the medium feature set.

## Status
- Rerun complete with baseline-aligned medium features.

## Results (medium feature set)
| model | bmc_mean | bmc_sharpe | small_bmc_mean | corr_mean |
|---|---|---|---|---|
| small_lgbm_ender20_baseline | 0.000856 | 0.085887 | 0.000000 | 0.025209 |
| small_catboost_baseline | 0.000159 | 0.015339 | -0.000483 | 0.022633 |

## Notes
- CatBoost is installed in the venv: `.venv/bin/pip install catboost`.
- Params align to the small LGBM baseline (depth 5, 2k iters, lr 0.01, rsm 0.1).
- See the catboost-specific sweep for parameter variants on the medium baseline.
- Config: `agents/experiments/catboost_small_baseline/configs/small_catboost_baseline.py`.
