# Next session — Ender-60 remote scale

Date parked: 2026-09-06

Follow `agents/skills/numerai-remote-compute/SKILL.md`.

## Status

- Medium residual-1.0 is live on Numerai (`model.pkl`, predictions ranked to `(0, 1]`). **Watch live rounds; do not re-upload unless the new run wins.**
- Local M2 is the control plane only. Next train is **HF Jobs `cpu-performance`** (256 GB, $1.90/hr, `--timeout 8h`, `n_jobs=16`).
- Current `v5.3/downsampled_full.parquet` has **no all-feature columns**.

## First tasks next session (in order)

1. Add `configs/residual_prop_100_all.py` (`feature_set: "all"`, same residual-1.0 hparams, `n_jobs: 16`).
2. Extend downsample so the all-features every-4th-era table exists (do **not** load full `train`+`validation` into pandas on the Mac).
3. Write an HF Jobs entry script: pip install, NumerAPI download **on the box**, build downsample, `python -m agents.code.modeling`.
4. Smoke-test `hf jobs run` on `cpu-basic`, then launch `cpu-performance`.
5. Decision: last-200 BMC must beat **0.00558** with CORR still ≳ 0.005.

## Do not

- Rent a GPU for this LGBM run.
- Remount the 8 GB local parquet tree into the job.
- Swap the primary model to XGBoost.
- Start the 30k-tree / full-era job until this scout wins.
