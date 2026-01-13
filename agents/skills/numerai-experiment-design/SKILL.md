---
name: numerai-experiment-design
description: Design and manage Numerai experiments in this repo for any model idea, with targeted sweeps aligned to the research question (not always hyperparameters), baseline progression (small -> deep), dataset selection (downsampled_full vs full), and experiment organization/reporting for positive BMC models.
---

# Numerai Experiment Design

Use this workflow to plan, run, and report Numerai experiments for any model idea.

## Planning checklist (answer before running)
- State the model idea and novelty.
- Choose the initial baseline and feature set (`small`/`medium`/`all`).
- Decide the primary metric for the phase (`small_bmc_mean` vs small baseline, `bmc_mean` vs deep baseline).
- Set the gate to move to the next phase (default: `small_bmc_mean >= 0.01` on downsampled).
- Decide which parameter dimensions to explore based on the core idea (targets, model hyperparameters, ensemble weights, data settings).
- Decide whether to skip sweeps and run a single baseline config; document why.
- Decide whether to use baseline residual boosting.
- Confirm the baseline predictions file exists (build it if missing).

## Workflow (scout → scale)
1) **Start downsampled**: Use `v5.2/downsampled_full.parquet` + `v5.2/downsampled_full_benchmark_models.parquet`.
2) **Pick the sweep dimension that matches the core idea**: Run a focused sweep only when it serves the research question; otherwise run a single baseline config and evaluate.
3) **Iterate until improvements stop**: Keep sweeping on that dimension while a round produces a new best metric. If a round does not improve, reassess or pivot.
4) **Focus when a parameter dominates**: If one parameter clearly drives results, dedicate a full round to mapping its range (including extremes) while holding others fixed.
5) **Scale only winners**: Once the gate is met, move to the next phase (feature_set=all + deep baseline), then re-sweep only if the phase change impacts the core research dimension.
6) **Full data final**: Run the top config on full data once the downsampled deep-baseline gate is met.

## Sweep selection by research type
- **New target/label/feature engineering**: Sweep target variants or preprocessing settings; skip hyperparameter sweeps unless performance is unstable.
- **New model architecture**: Run a hyperparameter sweep (depth/width, learning rate, regularization, epochs).
- **Ensemble/blend/stacking**: Sweep combination weights, blend rules, or stacker settings.
- **Training-procedure change**: Sweep procedure-specific params (loss weights, neutralization strength, sampling).
- **Data change**: Sweep universe, era sampling, or feature-set choices.

## Sweep design guidance
- Use one-variable-at-a-time changes for each run in the chosen sweep dimension.
- Build a base config per round, then create variants that change a single parameter or variant.
- Take time to design each round based on last-round results, model type, and known sensitivities.
- If scaling depth/width, lower learning rate and increase epochs.
- Track and compare per-round results; keep the best model and document why it won.

## Baseline residual boosting (optional)
- Use when the idea is a residual or stacked model.
- Configure:
  - `training.target_transform: {"type": "baseline_residual"}`
  - `training.prediction_transform: {"type": "add_baseline"}`
  - `data.small_bmc_baseline` and `data.small_bmc_baseline_path`
- Ensure the baseline predictions file exists before the sweep.

## Baseline alignment
- Declare which baseline the model is aiming to improve on. 
- Keep `feature_set` aligned with the baseline for comparisons.
- If small baseline, focus on small_bmc for assessment.  If deep baseline, focus on bmc for assessment. 

## Experiment organization
- Keep related runs under a single, well-named folder in `agents/experiments/`.
- One experiment folder = one line of inquiry.
  - `configs/` for configs
  - `logs/` for run logs
  - `predictions/` + `results/` from OOF CV
  - `experiment.md` for summary and decisions. Declare the baseline in the experiment.md. Update the experiment.md as you progress.
- Include a **baseline row** in result tables for comparisons.
- Name configs to reflect the single variable change.

## Reporting expectations
- Write a loop to continuously wait for your experiments to finish, so that you don't break your session and report prematurely.
- Once you complete your research and stop finding improvements, write a report for the user. It should describe learnings (what worked and what did not), include the final stats table, and run `python -m agents.analysis.plot_benchmark_corrs` on the final model (share the output path).
- Always report:
  - `bmc` (full) and `bmc_last_200_eras`
  - `small_bmc` (full) and `small_bmc_last200`
  - `corr_mean` and `corr_w_baseline_avg` (use `avg_corr_with_benchmark` as the baseline-corr proxy)
- Use consistent, markdown tables and update `experiment.md` after each run.

## Dataset handling
- Build datasets with `python -m agents.data.build_full_datasets`.
  - Full: `v5.2/full.parquet`, `v5.2/full_benchmark_models.parquet`
  - Downsampled (every 4 eras): `v5.2/downsampled_full.parquet`, `v5.2/downsampled_full_benchmark_models.parquet`
- Prefer downsampled for quick iteration; only scale after a clear signal.

## Useful entry points
- `python -m agents.modeling` (training + metrics)
- `agents/metrics/numerai_metrics.py` (BMC/corr summaries)
- `python -m agents.analysis.show_experiment` (compare runs)
- `python -m agents.data.build_full_datasets` (full + downsampled datasets)
