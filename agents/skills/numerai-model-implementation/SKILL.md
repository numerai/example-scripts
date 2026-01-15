---
name: numerai-model-implementation
description: Add a new Numerai model type to the agents training pipeline. Use when you need to register a model in `agents/utils/modeling/utils/model_factory.py`, handle fit/predict quirks in `agents/utils/modeling/utils/numerai_cv.py`, and update configs so the model can run via `python -m agents.utils.modeling`.
---

# Numerai Model Implementation

## Overview
Add a new model type so it can be selected in configs and trained/evaluated by the base pipeline.

## Implement a New Model Type

1. Define the model API and output shape.
   - Implement `fit(X, y, sample_weight=...)` and `predict(X)`.
   - Put custom wrappers in `agents/utils/modeling/models/` so model-specific code stays isolated.
   - Accept pandas DataFrames or convert to NumPy inside the model wrapper.

2. Register the model constructor in `agents/utils/modeling/utils/model_factory.py`.
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
   - Update `load_and_prepare_data` in `agents/utils/modeling/utils/pipeline.py` to pass extra columns into `load_full_data`.
   - Add corresponding config entries so experiments stay reproducible.

## Validate
- Run a smoke test: `.venv/bin/python -m agents.utils.modeling --config <config_path>`.
- Run metrics on the smoke test and make sure corr_mean is > 0.005 and < 0.04. If it's less then something is probably fundamentally wrong. If it's higher than there is likely leakage and you need to find the problem.
- Run unit tests after refactors: `.venv/bin/python -m unittest`.
