# Feature Set Experiments

## Status
This folder is archived under the new baseline-aligned standard
(`feature_set=medium`). The original feature-set sweep results below are legacy
and not comparable to the current baseline; a future feature-set sweep should
define its own baseline per feature set (or keep a dedicated control model).

## Goal
Legacy goal: determine which Numerai feature set yields the best BMC over the
last 200 eras when using identical LGBM parameters. Going forward, experiments
in this repo keep `feature_set` aligned to the baseline for comparability.

## What we're testing
- Legacy sweep of `feature_set` variants from `v5.2/features.json` using the
  same model configuration (`LGBMRegressor` with small params, 2k trees).
- Primary metric: `bmc_last_200_eras` (mean/sharpe/consistency/drawdown plus
  avg correlation with benchmark).
- Baseline for comparison at the time: `small_lgbm_ender20_baseline`.
- Configs live in `agents/experiments/feature_set_experiments/configs/`.
- Baseline artifacts live in `agents/baselines/`.

## Plan
1) Legacy sweep: train one model per feature set using the same LGBM parameters.
2) Store experiment predictions and results under this folder.
3) Rank feature sets by `bmc_last_200_eras.mean`.
4) Summarize top performers and note any stability or benchmark-correlation
   tradeoffs.
5) Scale promising feature sets to deep LGBM and compare against a deep baseline.

## Current status
- Archived under baseline-aligned standardization (feature_set=medium).
- Legacy sweep results below are kept for reference only.

## Metric details
- `bmc_mean`: mean per-era BMC over the last 200 eras (primary score).
- `bmc_sharpe`: mean/std of BMC over the last 200 eras (risk-adjusted).
- `bmc_consistency`: fraction of last-200 eras with positive BMC.
- `bmc_drawdown`: max drawdown of cumulative BMC over the last 200 eras.
- `bmc_avg_corr_bench`: mean per-era correlation between model predictions and
  the benchmark (ender20) over the last 200 eras; higher implies closer to the
  benchmark, lower implies more orthogonal signal.
- `small_bmc_last200`: BMC last‑200 where the benchmark is the
  `small_lgbm_ender20_baseline` predictions (measures uniqueness vs the small baseline).
- `corr_mean`: mean per-era Numerai correlation with target (all validation eras).
- `corr_sharpe`: mean/std of per-era correlation (all validation eras).

## Full results table
| model | feature_set | bmc_mean | bmc_sharpe | bmc_consistency | bmc_drawdown | bmc_avg_corr_bench | small_bmc_mean | small_bmc_sharpe | small_bmc_consistency | small_bmc_drawdown | small_bmc_avg_corr_bench | corr_mean | corr_sharpe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| small_lgbm_featureset_serenity | serenity | 0.001722 | 0.129849 | 0.535000 | 0.317112 | 0.208159 | 0.003171 | 0.243454 | 0.575000 | 0.210431 | 0.215162 | 0.008170 | 0.535969 |
| small_lgbm_featureset_midnight | midnight | 0.001195 | 0.098073 | 0.530000 | 0.210751 | 0.348045 | 0.003982 | 0.329973 | 0.585000 | 0.094578 | 0.331309 | 0.012740 | 0.771096 |
| small_lgbm_featureset_medium | medium | 0.000856 | 0.085887 | 0.510000 | 0.174905 | 0.648293 | 0.004439 | 0.571319 | 0.705000 | 0.050008 | 0.734161 | 0.025209 | 1.565396 |
| small_lgbm_featureset_agility | agility | 0.000777 | 0.060112 | 0.530000 | 0.220589 | 0.191741 | 0.002064 | 0.164109 | 0.595000 | 0.122367 | 0.194926 | 0.009835 | 0.661409 |
| small_lgbm_featureset_charisma | charisma | 0.000548 | 0.049389 | 0.510000 | 0.360046 | 0.198349 | 0.001642 | 0.150996 | 0.555000 | 0.210130 | 0.230900 | 0.008002 | 0.624347 |
| small_lgbm_featureset_faith | faith | 0.000376 | 0.031106 | 0.570000 | 0.275462 | 0.233495 | 0.000647 | 0.055345 | 0.565000 | 0.243866 | 0.324397 | 0.006008 | 0.431424 |
| small_lgbm_featureset_constitution | constitution | -0.000022 | -0.001988 | 0.525000 | 0.182219 | 0.176887 | 0.001614 | 0.142784 | 0.575000 | 0.147194 | 0.151246 | 0.004506 | 0.344262 |
| small_lgbm_featureset_wisdom | wisdom | -0.000046 | -0.003970 | 0.485000 | 0.273420 | 0.192622 | 0.001033 | 0.090351 | 0.515000 | 0.206751 | 0.203494 | 0.007521 | 0.556298 |
| small_lgbm_featureset_strength | strength | -0.000278 | -0.025664 | 0.540000 | 0.390932 | 0.150274 | 0.000587 | 0.055104 | 0.570000 | 0.282010 | 0.164613 | 0.003755 | 0.300605 |
| small_lgbm_featureset_sunshine | sunshine | -0.000649 | -0.055592 | 0.480000 | 0.305757 | 0.283381 | 0.001861 | 0.159327 | 0.580000 | 0.225335 | 0.258212 | 0.006364 | 0.448367 |
| small_lgbm_featureset_v2_equivalent_features | v2_equivalent_features | -0.000779 | -0.070318 | 0.515000 | 0.344304 | 0.431978 | 0.001776 | 0.164517 | 0.575000 | 0.139749 | 0.473423 | 0.018159 | 1.260510 |
| small_lgbm_featureset_intelligence | intelligence | -0.000930 | -0.092377 | 0.455000 | 0.211235 | 0.114430 | -0.000243 | -0.023949 | 0.485000 | 0.165979 | 0.119758 | 0.003774 | 0.302355 |
| small_lgbm_featureset_fncv3_features | fncv3_features | -0.001197 | -0.119043 | 0.475000 | 0.418125 | 0.480077 | 0.001907 | 0.194861 | 0.560000 | 0.145822 | 0.511704 | 0.019116 | 1.334860 |
| small_lgbm_featureset_small | small | -0.001305 | -0.123554 | 0.440000 | 0.325882 | 0.407312 | -0.000033 | -0.003306 | 0.510000 | 0.230071 | 0.510098 | 0.017242 | 1.149331 |
| small_lgbm_ender20_baseline | all | -0.001770 | -0.222607 | 0.425000 | 0.399957 | 0.741181 | 0.000000 |  | 0.000000 | 0.000000 | 0.998583 | 0.025294 | 1.559760 |
| small_lgbm_featureset_v3_equivalent_features | v3_equivalent_features | -0.002055 | -0.199669 | 0.415000 | 0.576717 | 0.509674 | 0.001232 | 0.126792 | 0.540000 | 0.182578 | 0.540563 | 0.019435 | 1.322594 |
| small_lgbm_featureset_dexterity | dexterity | -0.002382 | -0.220059 | 0.410000 | 0.487364 | 0.205318 | -0.000876 | -0.081196 | 0.460000 | 0.257549 | 0.205510 | 0.008523 | 0.593073 |
| small_lgbm_featureset_rain | rain | -0.004016 | -0.415196 | 0.365000 | 0.852571 | 0.498473 | -0.001492 | -0.146418 | 0.450000 | 0.638494 | 0.550800 | 0.013061 | 0.873001 |

## Conclusions
- Top `bmc_last_200_eras.mean` feature sets: `serenity` (0.001722),
  `midnight` (0.001195), `medium` (0.000856), `agility` (0.000777),
  `charisma` (0.000548).
- Best `small_bmc_last200.mean` (uniqueness vs baseline): `medium` (0.004439),
  `midnight` (0.003982), `serenity` (0.003171), `agility` (0.002064),
  `fncv3_features` (0.001907).
- Baseline (`all`) is negative on BMC last‑200 and highly correlated with the
  benchmark (avg corr ~0.74), consistent with “benchmark-like” behavior.
- OOF correlations now sit in a realistic range (corr mean <= ~0.025, sharpe <= ~1.6),
  indicating leakage is removed.
- Deep follow‑up: `midnight` scales positively on BMC; `serenity` does not.

## Deep LGBM follow‑up
| model | feature_set | bmc_mean | bmc_sharpe | bmc_avg_corr_bench | small_bmc_mean | small_bmc_sharpe | small_bmc_avg_corr_bench | corr_mean | corr_sharpe |
|---|---|---|---|---|---|---|---|---|---|
| deep_lgbm_ender20_baseline | all | -0.000489 | -0.085437 | 0.851074 | 0.006134 | 0.913139 | 0.846568 | 0.029024 | 1.724859 |
| deep_lgbm_featureset_serenity | serenity | -0.000397 | -0.033410 | 0.188017 | 0.001314 | 0.109532 | 0.166893 | 0.006738 | 0.475238 |
| deep_lgbm_featureset_midnight | midnight | 0.001951 | 0.166045 | 0.308033 | 0.005097 | 0.427794 | 0.249790 | 0.012036 | 0.764475 |
