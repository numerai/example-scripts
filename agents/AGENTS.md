# Numerai Agent

Goal: play the Numerai Tournament by running experiments and implementing new
research ideas to find models with strong BMC. The end goal
is a stable of relatively unique, high-performing models for submission.

Always check the agents/skills/ folder for skills that match the user request.

## Organization
- `agents/modeling/` contains the training package: `models/` for model-specific code, `utils/` for pipeline/config/data helpers, and `python -m agents.modeling` as the CLI entrypoint.
- `agents/baselines/` stores baseline configs plus their predictions/results.
- `agents/experiments/<name>/` contains experiment-specific configs, logs, predictions, results, and `experiment.md`.
- `agents/metrics/` holds shared scoring utilities (BMC/corr summaries).
- `agents/analysis/` holds reporting/plotting helpers (`show_experiment.py`, `plot_benchmark_corrs.py`, `summarize_results.py`).
- `agents/data/` holds dataset builders (full + downsampled parquet).
- `agents/skills/` contains Codex skills for this repo (each skill has a `SKILL.md`).
