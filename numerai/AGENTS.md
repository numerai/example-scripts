# Numerai Tournament
This folder contains examples on how to participate in the Numerai Tournament.

For v5.3, select `target_ender_60` explicitly in maintained training examples. The generic `target` column demonstrates the current default alias only; do not use it when the horizon must remain reproducible.

## Directory Guide:
- `agents/`: agentic research framework + training/analysis pipeline (`python -m agents.code.modeling`)
- `v5.3/` (and other `v*/`): Numerai dataset files (often gitignored locally)
- `*.ipynb`: tutorial notebooks for learning the tournament workflows

## Benchmark model coverage note
The `*_benchmark_models.parquet` files do **not necessarily cover every era** that appears in `full.parquet`.

For example, in this repo's `numerai/v5.3/` data:
- `full.parquet` contains eras `0001`–`1198`
- `full_benchmark_models.parquet` contains eras `0158`–`1198` (early eras are missing)

When joining targets to benchmark model predictions (or using target/prediction transforms that depend on benchmark columns), restrict to the **intersection of eras** available in both files (or expect missing values for early eras).
