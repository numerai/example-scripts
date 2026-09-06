# Ender-60 architecture

Date: 2026-09-06

## Abstract

Treat `target_ender_60` as a low-SNR residual-return problem and score BMC against `v53_lgbm_ender60`. A 400-tree medium-feature LGBM on the explicit payout target is the CORR floor (`corr_mean` 0.0253, `bmc_last_200` −0.00013). Training on the residual of that official LGBM (`proportion=1.0`) is the BMC winner (`bmc_last_200` 0.00558, `corr_mean` 0.0153). XGBoost vector-leaf on `{ender_60, teager2b_60}` keeps BMC that a 50/50 rank blend of the same targets destroys. A 40/40/20 rank-gauss blend of backbone + residual + vector-leaf is the best CORR-preserving ensemble (`corr_mean` 0.0252, `bmc_last_200` 0.00269).

## Hypothesis / Motivation

The official deep LGBM already owns standalone CORR on v5.3. A second model on the same features and same label mostly clones it (high `corr`, low BMC). Uniqueness has to come from the *label*: residualize to `v53_lgbm_ender60` before fitting, mix related 60D targets *inside* a vector-leaf tree, then blend with per-era rank-gauss rather than a raw average.

## Method

- **Data:** v5.3 downsampled every 4th era (`v5.3/downsampled_full.parquet`, 1,698,677 rows × 780 medium features). Benchmarks inner-joined from `v5.3/downsampled_full_benchmark_models.parquet` (1,542,547 overlapping ids). Rows with null `target_ender_60` dropped.
- **Target / benchmark:** explicit `target_ender_60` / `v53_lgbm_ender60`. Sibling models train on `teager2b` / `victor` / `tyler` 60 and score against `target_ender_60`.
- **CV:** expanding 5-fold, embargo 16 eras (60D purge). Fold 0 skipped (empty train after embargo). OOF: 1,430,422 rows / 245 eras (245–1221).
- **Scout hparams:** LGBM `n_estimators=400`, `lr=0.02`, `max_depth=5`, `num_leaves=31`, `min_data_in_leaf=10000`, `colsample_bytree=0.1`, CPU. XGB same tree budget, `multi_strategy=multi_output_tree` for vector-leaf.
- **Decision metric:** `bmc_last_200_eras.mean`. Tie-break `bmc_mean`. Sanity: `corr_mean` in ~0.005–0.04; reject high BMC with collapsed / negative CORR.

## Experiments run

### Layer 1 — Ender-60 LGBM backbone

- `scout_lgbm_ender60` — CORR floor. Artifacts: `results/scout_lgbm_ender60.json`, `predictions/scout_lgbm_ender60.parquet`.

### Track A — residual labels vs official LGBM

- `residual_prop_050` / `075` / `100` — `residual_to_benchmark` against `v53_lgbm_ender60`.
- `subtract_scale_007` — `subtract_benchmark_zscore` with scale 0.07.

### Track B — auxiliary 60D siblings + rank-gauss tilts

- Standalone: `sibling_teager2b60`, `sibling_victor60`, `sibling_tyler60`.
- Backbone tilts: 90/10, 80/20, 70/30 vs each sibling; also backbone × residual 80/20 and 70/30.

### Track C — XGBoost vector-leaf

- `xgb_scalar_ender60` (control).
- Vector-leaf: `xgb_vector_ender_teager`, `xgb_vector_ender_tyler`, `xgb_vector_ender_victor`.
- Prediction-level 50/50 rank-gauss of scalar XGB + the matching sibling.

### Layer 3 — architecture ensemble + scale

- `blend_arch_50_30_20` and `blend_arch_40_40_20`: scout + residual_prop_100 + xgb_vector_ender_teager.
- `scale_lgbm_ender60`: residual `proportion=1.0`, 2000 trees, depth 6 / 63 leaves, still medium features (the downsampled parquet has no `feature_set=all` columns).
- Upload pickle: `model.pkl` from `residual_prop_100` fit on all 1,698,677 labeled rows.

## Results

Primary ranking by `bmc_last_200`. `subtract_scale_007` is excluded from “winner” status: BMC is high because CORR went negative.

| model | corr_mean | bmc_mean | bmc_last_200 | avg_corr_bench |
|---|---:|---:|---:|---:|
| residual_prop_100 | 0.01531 | 0.00367 | **0.00558** | 0.215 |
| scale_lgbm_ender60 | 0.01951 | 0.00227 | 0.00372 | 0.327 |
| blend_arch_40_40_20 | 0.02522 | 0.00253 | 0.00269 | 0.437 |
| residual_prop_075 | 0.01160 | 0.00259 | 0.00248 | 0.167 |
| blend_arch_50_30_20 | 0.02606 | 0.00220 | 0.00201 | 0.460 |
| blend_b70_res100_30 | 0.02500 | 0.00197 | 0.00174 | 0.443 |
| residual_prop_050 | 0.00932 | 0.00134 | 0.00091 | 0.152 |
| xgb_vector_ender_teager | 0.02638 | 0.00164 | 0.00083 | 0.472 |
| sibling_victor60 | 0.02383 | 0.00183 | 0.00032 | 0.415 |
| xgb_vector_ender_victor | 0.02547 | 0.00099 | 0.00014 | 0.473 |
| sibling_teager2b60 | 0.02526 | 0.00093 | 0.00001 | 0.469 |
| scout_lgbm_ender60 | 0.02528 | 0.00091 | −0.00013 | 0.469 |
| xgb_scalar_ender60 | 0.02437 | 0.00011 | −0.00009 | 0.468 |
| blend_50_xgb_teager | 0.02691 | 0.00057 | −0.00006 | 0.508 |
| xgb_vector_ender_tyler | 0.02545 | 0.00091 | −0.00002 | 0.471 |
| sibling_tyler60 | 0.02006 | 0.00026 | −0.00080 | 0.388 |
| subtract_scale_007 | −0.00234 | 0.00771 | 0.00876 | −0.189 |

Track C contrast (same targets, mix inside the tree vs mix after):

| model | corr_mean | bmc_last_200 | avg_corr_bench |
|---|---:|---:|---:|
| xgb_vector_ender_teager | 0.02638 | **0.00083** | 0.472 |
| xgb_scalar_ender60 | 0.02437 | −0.00009 | 0.468 |
| blend_50_xgb_teager | 0.02691 | −0.00006 | 0.508 |

## Standard plot

Official LGBM vs residual winner, confirmatory scale, backbone, architecture blend, and vector-leaf (eras ≥ 575):

![benchmark vs residual and ensemble](plots/v53_lgbm_ender60_vs_residual_prop_100_plus_4_dark.png)

Vector-leaf vs scalar XGB vs 50/50 rank blend:

![vector-leaf vs rank blend](plots/v53_lgbm_ender60_vs_xgb_vector_ender_teager_plus_2_dark.png)

```bash
python -m agents.code.analysis.show_experiment benchmark residual_prop_100 scale_lgbm_ender60 scout_lgbm_ender60 blend_arch_40_40_20 xgb_vector_ender_teager \
  --base-benchmark-model v53_lgbm_ender60 \
  --benchmark-data-path v5.3/downsampled_full_benchmark_models.parquet \
  --target-col target_ender_60 --start-era 575 --dark \
  --output-dir agents/experiments/ender60_architecture
```

## Decisions made

- Eval contract is explicit `target_ender_60` + `v53_lgbm_ender60` + embargo 16. Do not score this work against `v53_lgbm_ender20` or the `target` alias.
- Residual `proportion=1.0` is the BMC champion; 0.50 / 0.75 are strictly weaker. `subtract_benchmark_zscore` is discarded (CORR collapsed).
- Sibling rank-tilts (10–30%) do not beat the residual on BMC. Victor is the only useful standalone sibling; tyler hurts last-200 BMC. Keep siblings as 10–30% tilts only inside an ensemble that already has residual signal.
- Vector-leaf is worth keeping as a high-CORR BMC slot, not as the primary model. Mix targets inside the tree; do not 50/50-blend them after the fact.
- Confirmatory scale stayed on medium features: `downsampled_full.parquet` has no all-feature columns, and a 30k-tree official deep LGBM is out of scope for this scout machine. Scale used 2000 trees / depth 6 on the residual-1.0 recipe.

## Stopping rationale

Residual BMC improved monotonically across the 0.50 → 0.75 → 1.00 sweep. Scale with 5× trees and a deeper tree did **not** beat the 400-tree residual on last-200 BMC (0.00372 vs 0.00558) and moved closer to the official model (`avg_corr_bench` 0.33 vs 0.22). Architecture blends plateau between 40/40/20 and 50/30/20. Two consecutive non-improving capacity / blend steps — stop.

## Findings

- The CORR backbone is a clone-ish scout (`avg_corr_bench` 0.47) with near-zero last-200 BMC. That is the expected floor, not a payout model.
- Residual training is the highest BMC per hour. `proportion=1.0` still keeps CORR in the sane band (0.015). More trees buy CORR and *spend* uniqueness.
- Naive prediction-level blends of related 60D targets keep CORR and erase BMC. Vector-leaf on `{ender_60, teager2b_60}` is the clean counterexample.
- For a submission that must keep CORR near the backbone, use `blend_arch_40_40_20`. For a BMC-max scout upload, use `residual_prop_100` (`model.pkl`).

## Status (2026-09-06)

Medium residual-1.0 pickle is **live on Numerai** (rank-to-`(0, 1]` fix in `export_pkl`). Watching live rounds. Next train is remote — see `next_session.md` and the `numerai-remote-compute` skill. Hardware: HF Jobs `cpu-performance`, not the MacBook.

## Next experiments

1. Build an all-features downsampled table and re-run residual-1.0 + the 40/40/20 blend **on HF `cpu-performance`** (256 GB). Current `downsampled_full.parquet` is medium-only.
2. Confirm residual-1.0 on *full* (not every-4th) v5.3 with the official 30k deep-LGBM budget as the new CORR floor.
3. Vector-leaf on `{ender_60, teager2b_60, victor_60}` with the same tree budget.
4. Partial feature neutralization on residual predictions (BMC up / CORR down) as a proportion sweep.
5. CatBoost or TabM as a later ensemble slot only after GBM-family BMC plateaus on full data.

## Repro commands

From `example-scripts/numerai`:

```bash
# Scout backbone
python -m agents.code.modeling \
  --config agents/experiments/ender60_architecture/configs/scout_lgbm_ender60.py \
  --output-dir agents/experiments/ender60_architecture

# Residual / sibling / vector-leaf scouts (same pattern, other configs/*.py)

# Rank-gauss blend
python -m agents.code.analysis.blend_predictions \
  --predictions \
    agents/experiments/ender60_architecture/predictions/scout_lgbm_ender60.parquet \
    agents/experiments/ender60_architecture/predictions/residual_prop_100.parquet \
    agents/experiments/ender60_architecture/predictions/xgb_vector_ender_teager.parquet \
  --weights 0.4 0.4 0.2 \
  --results-name blend_arch_40_40_20 \
  --output-dir agents/experiments/ender60_architecture \
  --target-col target_ender_60 \
  --benchmark-model v53_lgbm_ender60 \
  --benchmark-data-path v5.3/downsampled_full_benchmark_models.parquet

# Confirmatory residual scale
python -m agents.code.modeling \
  --config agents/experiments/ender60_architecture/configs/scale_lgbm_ender60.py \
  --output-dir agents/experiments/ender60_architecture

# Standard plot (command in Standard plot section)

# Upload pickle (residual-1.0, all labeled rows)
python -m agents.code.modeling.export_pkl \
  --config agents/experiments/ender60_architecture/configs/residual_prop_100.py \
  --output agents/experiments/ender60_architecture/model.pkl
```
