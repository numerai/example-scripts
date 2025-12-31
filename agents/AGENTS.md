# Numerai Experiment Workflow

Goal: play the Numerai Tournament by running experiments and implementing new
research ideas to find models with strong MMC (BMC is our proxy). The end goal
is a stable of relatively unique, high-performing models for submission.

## Run experiments
- Create or update a config in the experiment-specific `configs/` folder or in
  `agents/baselines/configs/` (Python dict files).
- Ensure `v5.2/full.parquet` and `v5.2/full_benchmark_models.parquet` exist
  (train + validation with validation `data_type != test` filtered out).
  - Rebuild with `agents/build_full_datasets.py --data-version v5.2`:
    - `full.parquet` = train + validation (drop validation rows where `data_type != validation`).
    - `full_benchmark_models.parquet` = train_benchmark + validation_benchmark
      filtered to validation ids where `data_type == validation`.
- Use `.venv/bin/python` for all runs (dependencies live in the venv).
- Naming: `small_lgbm_{xyz}` for 2k-tree params; `deep_lgbm_{xyz}` for 30k-tree params.
- Set `n_jobs=30` for LGBM configs.
- Keep `feature_set` identical to the baseline for comparability (current standard: `medium` for small configs); if you change it, create a matching baseline.
- Philosophy: run small/fast experiments first, then scale only the promising
  candidates to deep/optimal settings.
- Think critically and systematically about what question you are trying to answer with each experiment, and design it such that you are keeping as many variables constant as possible and only changing a single variable so that you can answer definitively whether that specific change was beneficial or not.
- Run with an experiment-scoped output directory:
  - `--output-dir agents/experiments/<experiment_name>`
  - Redirect logs to `<output-dir>/logs/`.
- Keep only the active experiment’s configs in its `experiments/<name>/configs/`
  folder to avoid mixing baselines.
- For long runs, start in the background and allow much longer timeouts to avoid CLI timeouts.

## Outputs
- Predictions: `<output-dir>/predictions/*.parquet` are OOF CV predictions
  (id, era, target, prediction, cv_fold).
- Results: `<output-dir>/results/*.json` (derived metrics).
- Baselines live in `agents/baselines/` with `predictions/` + `results/`.
- Recompute metrics from predictions to add new metrics without retraining.
- Write a short experiment report in `<output-dir>/experiment.md`.
- Use `agents/show_experiment.py` to compare a baseline vs one or more
  experiment models (table + correlation + BMC cumsum plots).

## Validation
- Run `.venv/bin/python -m unittest` after refactors to sanity-check config layout and paths.

## Metrics (evaluation)
- Small experiments: prioritize `small_bmc_last200.mean`, `small_bmc_last200.sharpe`,
  `corr.mean`, and `small_bmc_last200.avg_corr_with_benchmark` (report as `corr_w_baseline_avg`).
- Deep/scale experiments: use `bmc_last_200_eras` vs `ender20` benchmark as the primary score.
- Secondary: stability (sharpe, drawdown, consistency).

## Success criteria
- Small experiments: improves baseline on `small_bmc_last200.mean` without degrading `small_bmc_last200.sharpe`.
- Deep/scale experiments: improves baseline on `bmc_last_200_eras.mean`.
- Does not materially degrade stability metrics.
- (Optional) Re-run with multiple seeds to confirm stability.
