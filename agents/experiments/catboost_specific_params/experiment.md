# CatBoost Specific Params Sweep

## Goal
Test CatBoost-specific knobs while keeping LGBM-comparable hyperparameters fixed.

## What we're testing
All runs keep: depth=5, iterations=2000, learning_rate=0.01, rsm=0.1,
min_data_in_leaf=10000, RMSE loss, 5-fold OOF CV with embargo, and the medium
feature set to match the baseline.

CatBoost-specific variants:
- boosting_type=Ordered
- bootstrap_type=Bayesian (bagging_temperature=1.0)
- leaf_estimation_method=Gradient
- random_strength=0
- Configs live in `agents/experiments/catboost_specific_params/configs/`.

## Plan
1) Run each variant and save results in this folder.
2) Compare small_bmc_last200 vs the small LGBM baseline in `agents/baselines/`.
3) Keep any variants that improve small_bmc_last200 for deeper scaling.

## Results (medium feature set)
| model | bmc_mean | bmc_sharpe | small_bmc_mean | small_bmc_sharpe | corr_mean |
|---|---|---|---|---|---|
| small_catboost_random_strength_zero | 0.000937 | 0.090984 | 0.000292 | 0.059867 | 0.023146 |
| small_catboost_boosting_ordered | 0.000449 | 0.043546 | -0.000245 | -0.048729 | 0.022955 |
| small_catboost_leaf_gradient | 0.000159 | 0.015339 | -0.000483 | -0.100614 | 0.022633 |
| small_catboost_bootstrap_bayesian | -0.000112 | -0.010730 | -0.000714 | -0.138812 | 0.022628 |

Baseline reference: `small_lgbm_ender20_baseline` bmc_mean=0.000856, small_bmc_mean=0.000000.

## Decision
`small_catboost_random_strength_zero` edges the LGBM baseline on bmc_mean and
improves small_bmc_mean; it is the only variant worth considering for a deep
scale-up or multi-seed confirmation.
