# Numerai Agent

Goal: play the Numerai Tournament by running experiments and implementing new
research ideas to find models with strong BMC. The end goal
is a stable of relatively unique, high-performing models for submission.

Always check the agents/skills/ folder for skills that match the user request.
When running code, look for a local virtual environment (for example `./venv` or `./python311_venv`) and activate it if available.
We run commands from the `agents/` directory, but data is expected to live under `numerai/data_version`.

## Modeling philosophy
- All model-specific logic belongs in model configs and/or `agents/code/modeling/models/` wrappers.
- `pipeline.py`, `numerai_cv.py`, and metrics/analysis stay model-agnostic: they load requested data, call `fit`/`predict`, and record outputs.
- Keep the pipeline bare-bones; complexity is encapsulated inside the model itself.

## Organization
- `agents/code/` contains shared packages for training, data, analysis, and metrics.
- `agents/code/modeling/` contains the training package: `models/` for model-specific code, `utils/` for model-agnostic pipeline/config/data helpers, and `python -m agents.code.modeling` as the CLI entrypoint.
- `agents/code/metrics/` holds shared scoring utilities (BMC/corr summaries).
- `agents/code/analysis/` holds reporting/plotting helpers (`show_experiment.py`, `plot_benchmark_corrs.py`, `summarize_results.py`).
- `agents/code/data/` holds dataset builders (full + downsampled parquet).
- `agents/baselines/` stores baseline configs plus their predictions/results.
- `agents/experiments/<name>/` contains experiment-specific configs, logs, predictions, results, and `experiment.md`.
- `agents/skills/` contains Codex skills for this repo (each skill has a `SKILL.md`).
