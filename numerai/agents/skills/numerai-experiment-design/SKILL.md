---
name: numerai-experiment-design
description: Design and manage Numerai experiments in this repo for any model idea.
---

# Numerai Experiment Design
Use this workflow to plan, run, and report Numerai experiments for any model idea.

Note: run commands from `numerai/` (so `agents` is importable), or from repo root with `PYTHONPATH=numerai`.

## Persistence expectation (required)

This skill is *not* complete after a single promising run. You must run experiments in **rounds** (typically **4–5 configs per round**), synthesize results, and decide what to try next. Only finalize when you reach a plateau and additional rounds stop improving the primary metric.

## Planning checklist (answer before running)
- State the model idea and novelty.
- Choose the initial baseline and feature set. Default to `deep_lgbm_ender20_baseline` (feature_set=all) unless the user explicitly requests the small baseline; keep experiments' feature_set aligned with the chosen baseline.
- Decide the primary metric (`bmc_mean` and `bmc_last_200_eras`) where BMC = Benchmark Model Contribution vs official `v52_lgbm_ender20`.
- Decide which parameter dimensions to explore based on the core idea (targets, model hyperparameters, ensemble weights, data settings).
- Or decide that only a minimal round is needed because the change is tiny — but still run multiple variants unless the user explicitly requested exactly one run.

## Handling ambiguity (fast disambiguation)
If the user's request is unclear or underspecified:
1) List 2–4 plausible interpretations (keep them meaningfully different).
2) Implement **quick scout runs** for each interpretation (downsampled data, conservative compute).
3) Compare `bmc_mean` and `bmc_last_200_eras`.
4) Use the best-BMC interpretation going forward, and document the choice + rationale in `experiment.md`.

## Workflow 
Core loop (repeat for each experiment round):
1) If the model type is new, implement it with the numerai-model-implementation skill.
2) Create/update **4–5 configs** for the current round (one base + single-variable variants).
3) Run training for each config via `PYTHONPATH=numerai python3 -m agents.code.modeling --config <config> --output-dir <experiment_dir>`, which calls `pipeline.py` for CV/OOF + results.
4) Wait for the whole round to finish, then **synthesize** results:
   - pick the current best by `bmc_last_200_eras.mean` (primary), with `bmc_mean` as a tie-breaker
   - sanity-check `corr_mean` and `avg_corr_with_benchmark` (avoid “high corr, low BMC” traps)
   - check stability (drawdown/sharpe) and whether the improvement is consistent across eras
5) Update `experiment.md` with: what changed this round, the metrics table, and the next-round decision.
6) Repeat rounds until a plateau is reached (see “When to stop” below), then scale the winner.

## Scout -> Scale
1) **Use downsampled**: Use `v5.2/downsampled_full.parquet` + `v5.2/downsampled_full_benchmark_models.parquet` to save memory and time when experimenting.
2) **Pick the sweep dimension that matches the core idea**: Run a focused sweep only when it serves the research question; otherwise run a single experiment config and evaluate.
3) **Iterate until improvements stop**: Keep sweeping on that dimension while a round produces a new best metric. If a round does not improve, reassess or pivot.
4) **Focus when a parameter dominates**: If one parameter clearly drives results, dedicate a full round to mapping its range (including extremes) while holding others fixed.
5) **Scale only winners**: Once a best option is determined in the small baseline phase, move to phase 2 where you use the deep baseline and all feature_set, and scale the more expensive parameters like n_estimators and network size, if applicable.  
6) **Full data final**: Run the top config on full data and record the final metrics and final bmc when you stop finding improvements.

## When to stop (plateau criteria)

Stop iterating only when **at least two consecutive rounds** fail to beat the current best `bmc_last_200_eras.mean` by a meaningful margin (rule of thumb: ~`1e-4`–`3e-4`), *and* the remaining untried knobs are either redundant with what you already swept or likely to increase overfit/benchmark-correlation.

If you plateau on downsampled data, do *one* confirmatory scale step (bigger feature set and/or more data) before concluding the idea is maxed out.

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
- Default to ender20 (`v52_lgbm_ender20`) as the benchmark reference and plot baseline, even when sweeping; only use the small baseline when explicitly requested.

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
- Run experiments in **rounds** and continuously wait for the round to finish so you don't report prematurely.
- Once you complete your research and stop finding improvements, write a report for the user. It should describe learnings (what worked and what did not), include the final stats table, and run `PYTHONPATH=numerai python3 -m agents.code.analysis.show_experiment benchmark <best_model> --base-benchmark-model v52_lgbm_ender20 --benchmark-data-path numerai/v5.2/full_benchmark_models.parquet --start-era 575 --dark --output-dir <experiment_dir> --baselines-dir numerai/agents/baselines` to generate the cumulative corr + BMC plot (share the output path).
- Use `python -m agents.code.analysis.plot_benchmark_corrs` only when comparing official benchmark model columns, not for experiment BMC curves.
- Always report:
  - `bmc` (full) and `bmc_last_200_eras`
  - `corr_mean` and `avg_corr_with_benchmark` (corr vs the official benchmark predictions)
- Use consistent, markdown tables and update `experiment.md` after each run.
- Include a cohesive plan and story, finishing with a final result that combines learnings from all experiments. Think of yourself as a scientist writing a paper that walks the reader through your discoveries and thought process so that they understand why you finished with the result you did.

## Dataset handling
- Build datasets with `python -m agents.code.data.build_full_datasets`.
  - Full: `numerai/v5.2/full.parquet`, `numerai/v5.2/full_benchmark_models.parquet`
  - Downsampled (every 4 eras): `numerai/v5.2/downsampled_full.parquet`, `numerai/v5.2/downsampled_full_benchmark_models.parquet`
- Prefer downsampled for quick iteration; only scale after a clear signal for the final model.

## Useful entry points
- `PYTHONPATH=numerai python3 -m agents.code.modeling` (training + metrics)
- `agents/code/metrics/numerai_metrics.py` (BMC/corr summaries)
- `PYTHONPATH=numerai python3 -m agents.code.analysis.show_experiment` (compare runs)
- `PYTHONPATH=numerai python3 -m agents.code.data.build_full_datasets` (full + downsampled datasets)

## Deployment (after experiments complete)
Once you have finalized your best model and created a pkl file using the `numerai-model-upload` skill:

1. **Offer deployment**: Ask the user if they want to deploy the pkl to Numerai for automated submissions.

2. **Deployment options** (via the Numerai MCP server):
   - **Create a new model**: Use `create_model` to create a new model slot, then upload the pkl
   - **Upload to existing model**: List the user's existing models and upload to one they choose

3. **Follow the `numerai-model-upload` skill** for the complete deployment workflow using the MCP server tools (`create_model`, `upload_model`, `graphql_query`).

This allows the full research-to-deployment workflow to happen in a single session.
