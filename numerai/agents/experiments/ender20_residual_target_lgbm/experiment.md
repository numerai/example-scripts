# Ender20 residual-to-benchmark target (LGBM)

Date: **January 28, 2026**

## Abstract

We tested a target-engineering idea for Numerai Classic: train LightGBM on a per-era residualized version of the ender target (`target`), where the residual is computed against the official benchmark model predictions (`v52_lgbm_ender20`). The motivation is to directly optimize for signal that is linearly orthogonal to the benchmark within each era, which should improve Benchmark Model Contribution (BMC) by construction. On a downsampled OOF run, this produced **materially higher BMC** with **lower correlation to the benchmark**, at the cost of lower raw correlation vs `target` compared to the benchmark.

## Hypothesis

If we train on:

`y_resid = y - (α·bench + β)` (fit per-era),

then a model that predicts `y_resid` should:
- have **lower corr with the benchmark** (more unique), and
- achieve **higher BMC**, since BMC rewards contribution orthogonal to the benchmark.

## Method

### Target transform

We implemented a `model.target_transform` hook that transforms the training label at fit-time only (evaluation is still on the original `target` in the OOF predictions file).

Transform used in this experiment:
- **Type**: `residual_to_benchmark`
- **Benchmark column**: `v52_lgbm_ender20`
- **Per-era**: `True`
- **Fit intercept**: `True` (demean within era before fitting slope)
- **Proportion**: `1.0` (pure residual; no blending with original target)

### Data + evaluation

- **Dataset**: `v5.2/downsampled_full.parquet`
- **Benchmarks**: `v5.2/downsampled_full_benchmark_models.parquet` (for training-time transform + benchmark feature columns)
- **Scoring**: corr + BMC computed vs the official `v52_lgbm_ender20` benchmark predictions
- **CV**: expanding window, 5 splits, 13-era embargo

## Experiments run

### 1) Scout run (downsampled, medium feature set)

Config: `configs/lgbm_ender20_residual_benchmark_scout_downsampled.py`

Outputs:
- Predictions: `predictions/lgbm_ender20_residual_benchmark_scout_downsampled.parquet`
- Results JSON: `results/lgbm_ender20_residual_benchmark_scout_downsampled.json`

#### Metrics (OOF)

The table below matches the `show_experiment` output (baseline = benchmark predictions).

| Model | Feature set | corr_mean | bmc_mean | bmc_last200_mean | avg_corr_with_benchmark |
|---|---:|---:|---:|---:|---:|
| `v52_lgbm_ender20` | - | 0.031550 | 0.000000 | 0.000000 | 0.946794 |
| `lgbm_ender20_residual_benchmark_scout_downsampled` | medium | 0.014935 | 0.004488 | 0.004985 | 0.316216 |

## Standard plot

Generated with `agents.code.analysis.show_experiment` (baseline = `v52_lgbm_ender20`):

![v52 benchmark vs residual-target model](plots/v52_lgbm_ender20_vs_lgbm_ender20_residual_benchmark_scout_downsampled_dark.png)

## Decisions made

- **Residualize per-era (not globally)**: BMC is era-scored; we want orthogonality within each era.
- **Include intercept**: avoids bias from per-era mean shifts and yields a cleaner “orthogonal component”.
- **Run scout on downsampled + medium feature set first**: fast iteration to validate the core idea before scaling.
- **Kept evaluation target as original `target`**: ensures corr/BMC numbers are directly comparable to other runs.

## Findings

- The residualized-label model achieved **strong BMC** on the downsampled OOF run (`bmc_mean≈0.00449`, `bmc_last200_mean≈0.00499`) while keeping corr to the benchmark **much lower** (`avg_corr_with_benchmark≈0.316`) than typical baseline models.
- As expected, raw `corr_mean` vs `target` is lower than the benchmark’s `corr_mean` (uniqueness trade-off).

## Recommended next experiments

1. **Run the deep config** (all features, downsampled first): `configs/lgbm_ender20_residual_benchmark_deep_downsampled.py`
2. **Sweep `proportion`** (e.g., 0.5, 0.8, 1.0) to trade off corr vs BMC.
3. **Try residual-to-benchmark with/without benchmark columns in X** (confirm benchmark features aren’t hurting by encouraging leakage back into the benchmark direction).
4. **Scale to full data** only if the downsampled deep run maintains BMC.

## Repro commands

From repo root:

```bash
# Train (scout)
PYTHONPATH=numerai python3 -m agents.code.modeling --config numerai/agents/experiments/ender20_residual_target_lgbm/configs/lgbm_ender20_residual_benchmark_scout_downsampled.py

# Plot (benchmark baseline)
PYTHONPATH=numerai python3 -m agents.code.analysis.show_experiment benchmark lgbm_ender20_residual_benchmark_scout_downsampled \\
  --base-benchmark-model v52_lgbm_ender20 \\
  --benchmark-data-path numerai/v5.2/full_benchmark_models.parquet \\
  --start-era 575 --dark \\
  --output-dir numerai/agents/experiments/ender20_residual_target_lgbm \\
  --baselines-dir numerai/agents/baselines
```

