---
name: numerai-experiment-design
description: Design and manage Numerai experiments in this repo for any model idea.
---

# Numerai Experiment Design
Use this workflow to plan, run, and report Numerai experiments for any model idea.

## Planning checklist (answer before running)
- State the model idea and novelty.
- Choose the initial baseline and feature set. Default to `deep_lgbm_ender20_baseline` (feature_set=all) unless the user explicitly requests the small baseline; keep experiments' feature_set aligned with the chosen baseline.
- Decide the primary metric for the phase (`small_bmc_mean` for small baseline, `bmc_mean` for deep baseline).
- Decide which parameter dimensions to explore based on the core idea (targets, model hyperparameters, ensemble weights, data settings).
- Or decide that no parameter sweeps are necessary because the idea is a small enough change that only one test is needed.

## Workflow 
Core loop (repeat for each experiment round):
1) If the model type is new, implement it with the numerai-model-implementation skill.
2) Create/update experiment configs for the current sweep round.
3) Run training via `python -m agents.code.modeling --config <config> --output-dir <experiment_dir>`, which calls `pipeline.py` for CV/OOF + results.
4) Update `experiment.md` with decisions + metrics, then proceed to the next step in this skill.

## Scout -> Scale
1) **Use downsampled**: Use `v5.2/downsampled_full.parquet` + `v5.2/downsampled_full_benchmark_models.parquet` to save memory and time when experimenting.
2) **Pick the sweep dimension that matches the core idea**: Run a focused sweep only when it serves the research question; otherwise run a single experiment config and evaluate.
3) **Iterate until improvements stop**: Keep sweeping on that dimension while a round produces a new best metric. If a round does not improve, reassess or pivot.
4) **Focus when a parameter dominates**: If one parameter clearly drives results, dedicate a full round to mapping its range (including extremes) while holding others fixed.
5) **Scale only winners**: Once a best option is determined in the small baseline phase, move to phase 2 where you use the deep baseline and all feature_set, and scale the more expensive parameters like n_estimators and network size, if applicable.  
6) **Full data final**: Run the top config on full data and record the final metrics and final bmc when you stop finding improvements.

## Sweep selection by research type
Note that these are examples only. Each idea will call for different sweeps, or no sweeps. These are some guidelines but use your judgement to determine the best experiments to run to answer the core question of "does/can this core idea produce a model that has high bmc_mean?
- **New target/label/feature engineering**: Sweep target variants or preprocessing settings; skip hyperparameter sweeps unless performance is unstable.
- **New model architecture**: Run a hyperparameter sweep (depth/width, learning rate, regularization, epochs).
- **Ensemble/blend/stacking**: Sweep combination weights, blend rules, or stacker settings.
- **Training-procedure change**: Sweep procedure-specific params (loss weights, neutralization strength, sampling).
- **Data change**: Sweep universe, era sampling, or feature-set choices.

## Sweep design guidance
- Use one-variable-at-a-time changes for each run in the chosen sweep dimension.
- Build a base config per round, then create variants that change a single parameter or variant.
- Take time to design each round based on last-round results, model type, and known sensitivities.
- If scaling depth/width/n_estimators or related parameter, consider lower learning rate and/or increase epoch in conjunction.
- Track and compare per-round results; keep the best model and document why it won.

## Baseline alignment
- Declare which baseline the model is aiming to improve on.
- Keep `feature_set` aligned with the baseline for comparisons.
- Default to ender20 (v52_lgbm_ender20) as the benchmark reference and plot baseline, even when sweeping; only use the small baseline when explicitly requested.
- If small baseline, focus on small_bmc for assessment. If deep baseline, focus on bmc for assessment.

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
- Once you complete your research and stop finding improvements, write a report for the user. It should describe learnings (what worked and what did not), include the final stats table, and run `python -m agents.code.analysis.show_experiment benchmark <best_model> --base-benchmark-model v52_lgbm_ender20 --benchmark-data-path v5.2/full_benchmark_models.parquet --start-era 575 --dark --output-dir <experiment_dir> --baselines-dir baselines` to generate the cumulative corr + BMC plot (share the output path).
- Use `python -m agents.code.analysis.plot_benchmark_corrs` only when comparing official benchmark model columns, not for experiment BMC curves.
- Always report:
  - `bmc` (full) and `bmc_last_200_eras`
  - `small_bmc` (full) and `small_bmc_last200`
  - `corr_mean` and `corr_w_baseline_avg` (use `avg_corr_with_benchmark` as the baseline-corr proxy)
- Use consistent, markdown tables and update `experiment.md` after each run.
- Include a cohesive plan and story, finishing with a final result that combines learnings from all experiments. Think of yourself as a scientist writing a paper that walks the reader through your discoveries and thought process so that they understand why you finished with the result you did.

## Dataset handling
- Build datasets with `python -m agents.code.data.build_full_datasets`.
  - Full: `v5.2/full.parquet`, `v5.2/full_benchmark_models.parquet`
  - Downsampled (every 4 eras): `v5.2/downsampled_full.parquet`, `v5.2/downsampled_full_benchmark_models.parquet`
- Prefer downsampled for quick iteration; only scale after a clear signal for the final model.

## Useful entry points
- `python -m agents.code.modeling` (training + metrics)
- `agents/code/metrics/numerai_metrics.py` (BMC/corr summaries)
- `python -m agents.code.analysis.show_experiment` (compare runs)
- `python -m agents.code.data.build_full_datasets` (full + downsampled datasets)
