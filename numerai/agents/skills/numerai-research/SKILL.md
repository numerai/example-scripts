---
name: numerai-research
description: "End-to-end Numerai research workflow for trying a new idea: design experiments, implement new model types if needed, run scout→scale experiments, write a full experiment.md report with standard plots, and optionally package/upload a Numerai pickle. Use when a user asks to “try/test a new idea”, “run an experiment”, “sweep configs”, “compare model variants”, or otherwise do new Numerai research."
---

# Numerai Research

## Overview

This skill is a “meta-workflow” that sequences existing Numerai skills so research requests reliably produce: (1) runnable configs, (2) executed experiments, (3) a full written report + plots, and (4) a deployable pickle when requested.

## Workflow (always follow this order)

### 1) Design the experiment (use numerai-experiment-design)

- Follow the `numerai-experiment-design` skill to:
  - clarify the idea (or run quick scout interpretations if ambiguous)
  - choose baseline + feature set alignment (default ender20 baseline)
  - create an experiment folder under `numerai/agents/experiments/<experiment_name>/`
  - write configs in `configs/`
  - run training via `PYTHONPATH=numerai python3 -m agents.code.modeling --config <config> --output-dir <experiment_dir>`
  - track metrics with BMC as primary (`bmc_mean`, `bmc_last_200_eras`)
  - **iterate in rounds** (typically 4–5 configs per round), and keep going until you hit a plateau (per the experiment-design skill)
  - **scale winners** (bigger feature set and/or full data) before finalizing the best model

### 2) Implement new model types if needed (use numerai-model-implementation)

Only if the idea requires new code (new model wrapper, new fit/predict behavior, etc.):
- Follow the `numerai-model-implementation` skill to add the model type and register it.
- Add at least one smoke-test config and verify the pipeline runs.

### 3) Report the research (use report-research)

After you have iterated through multiple rounds **and** stopped finding improvements (plateau), and after any confirmatory scale runs:
- Follow the `report-research` skill to:
  - write a full `experiment.md` (abstract + methods + results + decisions + next steps)
  - generate the standard `show_experiment` plot(s)
  - link plots and artifacts in the report

### 4) Package and upload (use numerai-model-upload)

If (and only if) the user wants deployment:
- Follow the `numerai-model-upload` skill to create a Numerai-compatible pickle and upload it via the Numerai MCP.
- Remember: only Classic (tournament 8) supports pickle uploads.

## Defaults (unless user specifies otherwise)

- Scout first on downsampled data; scale only winners.
- Run experiments in rounds (4–5 configs per round) and stop only after a plateau + confirmatory scale step.
- Benchmark reference: `v52_lgbm_ender20`.
- Always record corr + BMC metrics and include the standard plot in the report.
