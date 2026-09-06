---
name: numerai-remote-compute
description: Rent remote CPU for Numerai LightGBM/XGBoost training (Hugging Face Jobs cpu-performance, Hetzner, AWS). Use when the user mentions HF jobs, remote training, all-features scale, GPU vs CPU, don't fry the MacBook, or picking up the next Ender-60 training run.
---

# Numerai remote compute

## Decision (2026-09-06)

Next scale run is **LightGBM residual on `feature_set=all`**, not a GPU job.

| Choice | Value |
|---|---|
| Provider | **Hugging Face Jobs** |
| Flavor | **`cpu-performance`** — 32 vCPU / **256 GB** / 1024 GB disk / **$1.90/hr** |
| Timeout | `--timeout 8h` |
| Expected cost | **$20–40** if the job is clean |
| `n_jobs` | **16** (physical cores; do not set 32 hyperthreads) |
| GPU | **Do not rent** for LightGBM |

Laptop (14" M2) stays git / configs / live-score watching. Offload because **RAM + hours of heat**, not because the Mac lacks cores.

## Why not GPU / other boxes

- Official competitor and our residual primary are LightGBM. Histogram GBMs want **RAM**, not VRAM. The LGBM GPU path is flaky; this repo already falls back to CPU.
- LightGBM scales to ~**16 physical cores**, then flattens / can slow down on multi-socket. Extra vCPUs on `a100x8` / `h200x8` sit idle.
- Scalar XGBoost is **not** mathematically superior (same additive trees). Keep XGB for **vector-leaf** only. TabM/nets (real GPU) only after GBM BMC plateaus on **full** data.
- HF `cpu-upgrade` is 32 GB — wrong row. Cap of **CPU-only** HF is `cpu-performance` **256 GB**.
- Hetzner **Cloud** caps at **CCX63 = 192 GB** (worse than HF). Hetzner **dedicated** AX162-3 is **512 GB** (up to 1 TB on AX162-R) — use that for full-era / 30k-tree, not this scout.
- Next RAM jump on HF after 256 GB is a GPU SKU (`l40sx4` 382 GB / $8.30, `a100x4` 568 GB / $10). Only if `cpu-performance` OOMs.

## CLI (HF Jobs)

`hf jobs run` requires **IMAGE** and **COMMAND**. Flavor alone fails.

Smoke test (cents, `cpu-basic`):

```bash
hf jobs run --name hello-cpu --flavor cpu-basic --timeout 2m python:3.12 python -c 'print("hello from hf jobs")'
```

Training shape (do not fire until the all-features job script exists):

```bash
hf jobs run --name ender60-all-residual \
  --flavor cpu-performance \
  --timeout 8h \
  --detach \
  -e PYTHONUNBUFFERED=1 \
  python:3.12 \
  bash -c '<install deps; download Numerai data ON the box; train>'
```

- Download `v5.3/train.parquet` + `validation.parquet` **inside the job** via NumerAPI. Do **not** `-v` the local 8 GB data tree up to HF.
- Stock `python:3.12` has no LightGBM / this repo. The command must pip-install and get the code (git clone or a **code-only** volume).
- Default job timeout is **30 minutes** — always set `--timeout`.
- Pickle on the Linux 3.12 job when possible (matches `numerai_predict`). `export_pkl` must rank predictions per-era to `(0, 1]` (`rank(pct=True)`).

## Pickup: Ender-60 next run

Bookmark: `agents/experiments/ender60_architecture/next_session.md`.

**Already done (do not redo):** medium-feature residual-1.0 scout; `model.pkl` uploaded (rank-to-unit-interval fix); live Classic watch.

**Not done (this is the work):**

1. Config + downsample that keeps **`feature_set=all` (3555 cols)**. Current `v5.3/downsampled_full.parquet` is **medium-only** (500 MB, 780 cols).
2. HF Jobs entry script (install, download, build downsample, train residual-1.0, write `results/`).
3. Launch `cpu-performance` 8h job.
4. Compare last-200 BMC to **0.00558** and CORR to the 0.005 reject line. Promote only if BMC wins.

Same eval contract: `target_ender_60` + `v53_lgbm_ender60` + embargo 16. Do not swap to XGBoost as the primary.

## When to leave HF

Full weekly v5.3 + 30k-tree confirmatory floor: 256 GB may be tight. Then a **512 GB CPU** box (Hetzner dedicated AX162-3 or AWS `r7i.16xlarge`), not a GPU cluster.
