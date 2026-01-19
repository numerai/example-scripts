---
name: numerai-model-upload
description: Create Numerai Tournament model upload pickles (.pkl) with a self-contained predict() function. Use when preparing upload artifacts, debugging numerai_predict import errors, or documenting model-upload requirements and testing steps.
---

# Numerai Model Upload

## Overview
Create a portable `predict(live_features, live_benchmark_models)` pickle that runs inside Numerai's `numerai_predict` container without repo dependencies.

## Requirements
- Implement `predict(live_features, live_benchmark_models)` and return a DataFrame with a `prediction` column aligned to the input index.
- Preserve training-time preprocessing (feature order, imputation values, scaling params) inside the pickle.
- Avoid imports from local repo modules (no `agents.*`), because Numerai's container will not have them.
- Prefer numpy/pandas/scipy-only inference; do not rely on torch/xgboost unless you verify the container has those packages.
- Move any trained model to CPU before exporting and store plain numpy weights.
- Validate required columns (`era` for per-era ranking, benchmark column if used).

## Workflow
1. Train on the desired full dataset (train + validation) with the same preprocessing and early-stopping scheme as the best model.
2. Export an inference bundle from the trained model:
   - Feature list and ordering
   - Imputation values and scaling stats
   - Model weights/biases (numpy arrays)
   - Activation name and any constants
   - Benchmark column name if needed as a feature
3. Build a `predict` function that:
   - Reads only from the bundle and standard libraries
   - Applies preprocessing and a numpy forward pass
   - Ranks predictions per era to [0, 1] when required
4. `cloudpickle.dump(predict, "model.pkl")` and keep the output in the repo root for upload.
5. Test the pickle with the Numerai container before uploading.

## Testing
Run the Numerai debug container locally:

```bash
docker run -i --rm -v "$PWD:$PWD" ghcr.io/numerai/numerai_predict_py_3_11:a78dedd --debug --model $PWD/[PICKLE_FILE]
```

## Common Pitfalls
- `ImportError: No module named 'agents'`: occurs when the pickle references repo classes. Fix by exporting a pure-numpy inference bundle and rebuilding `predict` without repo imports.
- Missing `era` column: per-era ranking requires `live_features["era"]`.
- Benchmark misalignment: ensure `live_benchmark_models` is reindexed to `live_features` (by id) before use.
- Feature drift: ensure feature order in inference matches training order exactly.

## Reference
- Use `numerai/example_model.ipynb` for the expected `predict` signature and output format.
