---
name: numerai-model-upload
description: Create Numerai Tournament model upload pickles (.pkl) with a self-contained predict() function. Use when preparing upload artifacts, debugging numerai_predict import errors, or documenting model-upload requirements and testing steps.
---

# Numerai Model Upload

## Overview
Create a portable `predict(live_features, live_benchmark_models)` pickle that runs inside Numerai's `numerai_predict` container without repo dependencies.

## CRITICAL: Python Version Compatibility

**Before creating any pkl file**, you must ensure your Python environment matches Numerai's compute environment. Mismatched versions cause segfaults and validation failures due to binary incompatibility (especially with numpy).

### Step 1: Query the Default Docker Image (MCP Required)

If the `numerai` MCP server is available, **always query the default Python version first**:

```graphql
query { computePickleDockerImages { id name image tag default } }
```

Look for the entry with `default: true`. The image name indicates the Python version:
- `numerai_predict_py_3_12:a78dedd` → Python 3.12 (current default as of 2026)
- `numerai_predict_py_3_11:a78dedd` → Python 3.11
- `numerai_predict_py_3_10:a78dedd` → Python 3.10

### Step 2: Create Matching Virtual Environment with pyenv

Use `pyenv` to create a virtual environment with the exact Python version:

```bash
# 1. List available pyenv Python versions
ls ~/.pyenv/versions/

# 2. Find the matching minor version (e.g., for Python 3.12)
PYENV_PY=$(ls -d ~/.pyenv/versions/3.12.* 2>/dev/null | head -1)

# 3. Create the virtual environment
$PYENV_PY/bin/python -m venv ./venv

# 4. Activate and install pkl dependencies
source ./venv/bin/activate
pip install --upgrade pip
pip install numpy pandas cloudpickle scipy
# Add lightgbm, torch, etc. only if your model needs them
```

### Step 3: Create pkl in the Correct Environment

**Always create pkl files using the matching venv**:

```bash
./venv/bin/python create_model_pkl.py
```

## Requirements
- Implement `predict(live_features, live_benchmark_models)` and return a DataFrame with a `prediction` column aligned to the input index.
- Preserve training-time preprocessing (feature order, imputation values, scaling params) inside the pickle.
- Avoid imports from local repo modules (no `agents.*`), because Numerai's container will not have them.
- Prefer numpy/pandas/scipy-only inference; do not rely on torch/xgboost unless you verify the container has those packages.
- Move any trained model to CPU before exporting and store plain numpy weights.
- Validate required columns (`era` for per-era ranking, benchmark column if used).

## Workflow
1. **Query the default Docker image** from the MCP to determine the required Python version.
2. **Create/activate a matching venv** using pyenv (see above).
3. Train on the desired full dataset (train + validation) with the same preprocessing and early-stopping scheme as the best model.
4. Export an inference bundle from the trained model:
   - Feature list and ordering
   - Imputation values and scaling stats
   - Model weights/biases (numpy arrays)
   - Activation name and any constants
   - Benchmark column name if needed as a feature
5. Build a `predict` function that:
   - Reads only from the bundle and standard libraries
   - Applies preprocessing and a numpy forward pass
   - Ranks predictions per era to [0, 1] when required
6. `cloudpickle.dump(predict, "model.pkl")` using the matching venv's Python.
7. Test the pickle with the Numerai container before uploading.

## Testing
Run the Numerai debug container locally (use the same image tag as the default):

```bash
# Get the default image tag from MCP query, then test:
docker run -i --rm -v "$PWD:$PWD" ghcr.io/numerai/numerai_predict_py_3_12:a78dedd --debug --model $PWD/[PICKLE_FILE]
```

## Common Pitfalls
- **Segmentation fault / numpy binary incompatibility**: The pkl was created with a different Python version than Numerai's container. **Always query the default docker image first** and create pkl files using a matching pyenv-based venv.
- `ImportError: No module named 'agents'`: occurs when the pickle references repo classes. Fix by exporting a pure-numpy inference bundle and rebuilding `predict` without repo imports.
- Missing `era` column: per-era ranking requires `live_features["era"]`.
- Benchmark misalignment: ensure `live_benchmark_models` is reindexed to `live_features` (by id) before use.
- Feature drift: ensure feature order in inference matches training order exactly.

## Debugging Validation Failures

If your pickle fails validation, query the trigger status and logs:

```graphql
query {
  account {
    models {
      username
      computePickleUpload {
        filename
        validationStatus
        triggerStatus
        triggers {
          id
          status
          statuses {
            status
            description
            insertedAt
          }
        }
      }
    }
  }
}
```

Common error descriptions:
- `"Segmentation fault! Ensure python and library versions match our environment."` → Python/numpy version mismatch
- `"No currently open rounds!"` → Model validated successfully but no round is open for submission

## Reference
- Use `numerai/example_model.ipynb` for the expected `predict` signature and output format.

---

## Deploying to Numerai via MCP Server

After creating and testing your pkl file, you can deploy it to Numerai using the **Numerai MCP server**. The MCP server provides tools for creating models and uploading pkl files programmatically.

### Available MCP Tools

The `numerai` MCP server provides these key tools:

1. **`check_api_credentials`** - Verify your API token and see granted scopes
2. **`create_model`** - Create a new model in a tournament
3. **`upload_model`** - Upload pkl files (multi-step workflow)
4. **`graphql_query`** - List existing models and perform custom queries

### Authentication

All authenticated operations require a Numerai API token with `upload_submission` scope:
- Format: `PUBLIC_ID$SECRET_KEY`
- Get your API key from https://numer.ai/account

### Option 1: Upload to an Existing Model

If you already have a model slot you want to use:

1. **List your models** using `graphql_query`:
   ```graphql
   query {
     account { models { id name } }
   }
   ```

2. **Get upload authorization** for your pkl file:
   - Call `upload_model` with `operation: "get_upload_auth"`, `modelId: "<model_uuid>"`, `filename: "model.pkl"`
   - This returns a presigned URL for uploading

3. **Upload the pkl file**
   - Call a PUT file upload on the pre-signed URL with the path to the pkl file.

4. **Register the upload** with Numerai:
   - Call `upload_model` with `operation: "create"`, `modelId: "<model_uuid>"`, `filename: "model.pkl"`
   - This triggers validation of your pickle

5. **Check validation status**:
   - Call `upload_model` with `operation: "list"` to see all pickles and their status
   - Wait for validation to complete successfully

6. **Assign the pickle to the model slot**:
   - Call `upload_model` with `operation: "assign"`, `modelId: "<model_uuid>"`, `pickleId: "<pickle_uuid>"`
   - This makes the pickle active for automated submissions

### Option 2: Create a New Model and Upload

If you want to create a new model slot:

1. **Create the model**:
   - Call `create_model` with `name: "<unique_model_name>"`, `tournament: 8` (for Classic)
   - Note: Model names must be unique within the tournament

2. **Get the model ID** from the response

3. **Follow steps 2-6 from Option 1** to upload and assign the pkl file

### Upload Workflow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    PKL DEPLOYMENT WORKFLOW                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Create pkl file (this skill's main workflow)                │
│  2. Test pkl locally with numerai_predict container             │
│  3. Choose: create new model OR use existing model              │
│                                                                  │
│  For new model:                                                  │
│    └─> create_model(name, tournament=8)                         │
│                                                                  │
│  For existing model:                                             │
│    └─> graphql_query to list models and get model ID            │
│                                                                  │
│  4. upload_model(operation="get_upload_auth", modelId, filename)│
│  5. upload_model(operation="put_file", presignedUrl, localPath) │
│  6. upload_model(operation="create", modelId, filename)         │
│  7. upload_model(operation="list") - wait for validation        │
│  8. upload_model(operation="assign", modelId, pickleId)         │
│                                                                  │
│  Optional:                                                       │
│  - upload_model(operation="trigger", pickleId) to test          │
│  - upload_model(operation="get_logs", pickleId, triggerId)      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Important Notes

- Only the **Classic tournament (tournament=8)** supports pickle uploads
- The model must have its submission webhook disabled before uploading
- **CRITICAL**: Before creating pkl files, query the default docker image to ensure Python version compatibility
- Use this GraphQL query to check available runtimes and the default:
  ```graphql
  query { computePickleDockerImages { id name image tag default } }
  ```
- Use `upload_model(operation="list_data_versions")` to see available dataset versions
- After assignment, Numerai will automatically run your pickle each round

### Pre-Upload Checklist

Before uploading a pkl file, verify:
1. ✅ Queried `computePickleDockerImages` to get the default Python version
2. ✅ Created venv using pyenv with matching Python version
3. ✅ Created pkl file using the matching venv's Python interpreter
4. ✅ Tested pkl locally with the matching docker container (optional but recommended)

### Triggering and Debugging

After assigning a pickle, you can manually trigger it for testing:

1. **Trigger the pickle**:
   - Call `upload_model` with `operation: "trigger"`, `pickleId: "<pickle_uuid>"`, `triggerValidation: true`
   
2. **View execution logs**:
   - Call `upload_model` with `operation: "get_logs"`, `pickleId: "<pickle_uuid>"`, `triggerId: "<trigger_uuid>"`

### Asking the User

Before deploying, confirm with the user:
1. Do they want to deploy the pkl to Numerai?
2. Should we create a new model or upload to an existing one?
3. If new: what name should the model have?
4. If existing: which model should receive the upload?
5. Do they have their API token ready (or is it already configured)?
