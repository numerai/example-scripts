---
name: report-research
description: Write a complete Numerai experiment report in experiment.md (abstract, methods, results tables, decisions, next steps) and generate/link the standard show_experiment plot(s). Use after running any Numerai research experiments, or when a user asks for a “full report”, “write up”, “experiment.md update”, or “generate the standard plot”.
---

# Report Research

## Overview

This skill turns an experiment run into a durable write-up: a full `experiment.md` plus the standard `show_experiment` plot(s) linked from the report.

## Workflow (do all steps)

### 1) Locate the experiment folder

Use the folder that contains:
- `configs/` (the configs you ran)
- `results/` (JSON metrics output)
- `predictions/` (OOF parquet output)
- `experiment.md` (the report you will write/update)

### 2) Inventory what was actually run

- List configs that exist.
- Determine which ones were executed by checking for matching `results/*.json` and `predictions/*.parquet`.
- Identify the “best” model(s) using `bmc_mean` and `bmc_last_200_eras.mean` (primary), with `corr_mean` as a sanity check.
- If experiments were run in rounds, summarize **each round’s intent** (what changed) and whether it improved the current best.

### 3) Extract metrics for the report

For each run you report, include at least:
- `corr_mean`
- `bmc_mean`
- `bmc_last_200_eras.mean`
- `avg_corr_with_benchmark` (from the BMC summary)

Prefer a single markdown table with one row per model.

### 4) Write a full report in experiment.md

Update/create `experiment.md` with these sections (keep it crisp but complete):
- **Title + Date**
- **Abstract** (what was tested + headline result)
- **Hypothesis / Motivation** (why this should help BMC)
- **Method** (data split, CV, feature set, model type/hparams, any transforms)
- **Experiments run** (one subsection per config that actually ran; include output artifacts)
- **Results** (the metrics table; mention best run + trade-offs)
- **Standard plot** (embed the PNG and include the generating command)
- **Decisions made** (what you chose and why; e.g., per-era vs global, feature set choice, sweep choices)
- **Stopping rationale** (why you stopped iterating; e.g., plateau after N rounds, confirmatory scale step, diminishing returns)
- **Findings** (what worked / didn’t; interpret the plot)
- **Next experiments** (2–5 concrete follow-ups)
- **Repro commands** (train + plot commands from repo root)

### 5) Generate the standard plot(s) and link them

Default standard plot (baseline = benchmark predictions):

```bash
PYTHONPATH=numerai python3 -m agents.code.analysis.show_experiment benchmark <best_model_results_name> \
  --base-benchmark-model v52_lgbm_ender20 \
  --benchmark-data-path numerai/v5.2/full_benchmark_models.parquet \
  --start-era 575 --dark \
  --output-dir numerai/agents/experiments/<experiment_name> \
  --baselines-dir numerai/agents/baselines
```

Then embed it in `experiment.md` with a relative link:

```md
![benchmark vs best model](plots/<generated_plot_name>.png)
```

If you have multiple candidate models, either:
- generate one plot with multiple experiment models, or
- generate one plot per candidate (and link all of them).

### 6) Final checks

- Plot files exist under `plots/`.
- `experiment.md` links resolve (use relative paths).
- Metrics table matches `results/*.json`.
- Report clearly states what was run vs what is only planned/configured.
