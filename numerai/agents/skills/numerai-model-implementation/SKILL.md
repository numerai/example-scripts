---
name: numerai-model-implementation
description: Add a new Numerai model type to the agents training pipeline. Use when you need to register a model in `agents/code/modeling/utils/model_factory.py`, handle fit/predict quirks in `agents/code/modeling/utils/numerai_cv.py`, and update configs so the model can run via `python -m agents.code.modeling`.
---

# Numerai Model Implementation

## Overview
Add a new model type so it can be selected in configs and trained/evaluated by the base pipeline.

Note: run commands from `numerai/` (so `agents` is importable), or from repo root with `PYTHONPATH=numerai`.

## Implement a New Model Type

1. Define the model API and output shape.
   - Implement `fit(X, y, sample_weight=...)` and `predict(X)`.
   - Put custom wrappers in `agents/code/modeling/models/` so model-specific code stays isolated.
   - Accept pandas DataFrames or convert to NumPy inside the model wrapper.

2. Register the model constructor in `agents/code/modeling/utils/model_factory.py`.
   - Use lazy imports so optional dependencies do not break other workflows.
   - Raise a clear ImportError when the dependency is missing.

```python
if model_type == "XGBRegressor":
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for XGBRegressor. Install with `.venv/bin/pip install xgboost`."
        ) from exc
    return XGBRegressor(**model_params)
```

3. Add or update a config to use the new model type.

```python
CONFIG = {
    "model": {"type": "XGBRegressor", "params": {"n_estimators": 500}},
    "training": {"cv": {"n_splits": 5}},
    "data": {"data_version": "v5.2", "feature_set": "small", "target_col": "target", "era_col": "era"},
    "output": {},
    "preprocessing": {},
}
```

4. Add extra data columns if the model needs them.
   - Update `load_and_prepare_data` in `agents/code/modeling/utils/pipeline.py` to pass extra columns into `load_full_data`.
   - Add corresponding config entries so experiments stay reproducible.

## Validate
- Run a smoke test: `.venv/bin/python -m agents.code.modeling --config <config_path>`.
- Run metrics on the smoke test and make sure corr_mean is > 0.005 and < 0.04. If it's less then something is probably fundamentally wrong. If it's higher than there is likely leakage and you need to find the problem.
- Double check that any early stopping mechanisms or modifications to the fit/predict loop don't over-estimate accuracy. Accurately estimating performance is of paramount importance on Numerai because we need to be able to decide if we should stake or not.
- Run unit tests after refactors: `.venv/bin/python -m unittest`.

## Next Steps
After validating the model implementation:
1. Use the `numerai-experiment-design` skill to run **multiple rounds** of experiments (4–5 configs per round), then **scale winners** until you hit a plateau.
2. Use the `numerai-model-upload` skill to create a pkl file **only after** you have a stable, scaled “best model” you intend to deploy.
3. Deploy to Numerai using the MCP server (see `numerai-model-upload` skill for deployment workflow).
