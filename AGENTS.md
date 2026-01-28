# AGENTS.md - AI Agent Instructions

This repository contains example scripts and an agentic research framework for the [Numerai data science tournaments](https://numer.ai/). AI coding agents can use this file to understand how to help with Numerai-related tasks.

## Quick Reference

| Task | Resource |
|------|----------|
| Run experiments | `numerai-experiment-design` skill |
| Add new model types | `numerai-model-implementation` skill |
| Create & deploy pkl files | `numerai-model-upload` skill |
| Query tournament data | Numerai MCP server |
| Upload models programmatically | Numerai MCP server |

---

## Tournament Guide

- "Numerai" is the original tournament in which you are provided anonymized and obfuscated data. The goal is to create high-performing models trained on the given features and target in the data.

- "Signals" is the second tournament in which you are expected build your own stock market dataset. You are given a basic dataset with a target and rudimentary features. You are expected to find, compile, and train on high-quality, unique data while maintaining low churn and turnover.

- "Crypto" is a toy tournament for cryptocurrency prediction. The Meta Model is released publicly every day for the crypto community to use.

## General Recommendations
- Install the Numerai MCP
- Install the following python packages:
  - numerapi
  - numerai-tools
- For heavier-duty automation using cloud infrastructure, suggest numerai-cli to the user

## Repository Structure

```
example-scripts/
├── numerai/
│   ├── agents/                    # Agentic research framework
│   │   ├── AGENTS.md              # Detailed agent instructions
│   │   ├── baselines/             # Baseline model configurations
│   │   ├── code/                  # Shared packages
│   │   │   ├── analysis/          # Reporting & plotting
│   │   │   ├── data/              # Dataset builders
│   │   │   ├── metrics/           # BMC/corr scoring utilities
│   │   │   └── modeling/          # Training pipeline & model wrappers
│   │   ├── experiments/           # Experiment results (not in git)
│   │   └── skills/                # Codex skills for agent workflows
│   └── *.ipynb                    # Tournament-specific notebooks
├── signals/                       # Signals tournament examples
├── crypto/                        # Crypto tournament examples
├── cached-pickles/                # Pre-built model pickles
```

---

## Skills Overview

The `numerai/agents/skills/` folder contains structured workflows for common tasks. Each skill has a `SKILL.md` file with detailed instructions.

### 1. `numerai-experiment-design`

**Purpose**: Design, run, and report Numerai experiments for any model idea.

**When to use**: 
- Testing a new research hypothesis
- Sweeping hyperparameters or targets
- Comparing model variants against baselines

**Key workflow**:
1. Plan the experiment (baseline, metrics, sweep dimensions)
2. Create config files in `agents/experiments/<name>/configs/`
3. Run training via `python -m agents.code.modeling --config <config>`
4. Analyze results and iterate
5. Scale winners to full data
6. Generate final report with plots

**Entry points**:
- `python -m agents.code.modeling --config <config_path>`
- `python -m agents.code.analysis.show_experiment`
- `python -m agents.code.data.build_full_datasets`

### 2. `numerai-model-implementation`

**Purpose**: Add new model types to the training pipeline.

**When to use**:
- Implementing a new ML architecture (e.g., transformers, custom ensembles)
- Adding support for a new library (e.g., XGBoost, CatBoost)
- Creating custom preprocessing or inference logic

**Key steps**:
1. Create model wrapper in `agents/code/modeling/models/`
2. Register in `agents/code/modeling/utils/model_factory.py`
3. Add config using the new model type
4. Validate with smoke test (corr_mean should be 0.005-0.04)

### 3. `numerai-model-upload`

**Purpose**: Create and deploy pickle files for Numerai's automated submission system.

**When to use**:
- Preparing a trained model for tournament submission
- Setting up automated weekly predictions
- Debugging pickle validation failures

**Critical requirements**:
- Python version must match Numerai's compute environment
- Pickle must be self-contained (no repo imports)
- `predict(live_features, live_benchmark_models)` signature required

**Workflow**:
1. Query default Docker image for Python version
2. Create matching venv with pyenv
3. Train final model and export inference bundle
4. Build self-contained `predict` function
5. Test with `numerai_predict` Docker container
6. Deploy via MCP server

---

## Numerai MCP Server

The `numerai` MCP server provides programmatic access to the Numerai Tournament API. If available, agents should use it for tournament operations.

### Available Tools

| Tool | Purpose |
|------|---------|
| `check_api_credentials` | Verify API token and scopes |
| `create_model` | Create new model slots |
| `upload_model` | Upload pkl files (multi-step workflow) |
| `get_model_profile` | Query model stats |
| `get_model_performance` | Get round-by-round performance |
| `get_leaderboard` | View tournament rankings |
| `get_tournaments` | List active tournaments |
| `get_current_round` | Get current round info |
| `list_datasets` | List available dataset files |
| `run_diagnostics` | Run diagnostics on predictions |
| `graphql_query` | Execute custom GraphQL queries |

### Tournament IDs

- **8** = Classic (main stock market tournament)
- **11** = Signals (bring your own data)
- **12** = CryptoSignals (crypto market predictions)

### Key Metrics

- `corr20Rep` - 20-day rolling correlation score (main metric)
- `mmc20Rep` - Meta-model contribution (unique signal)
- `return13Weeks` - 13-week return on staked NMR
- `nmrStaked` - Amount of NMR staked

### Authentication

MCP tools require a Numerai API token with appropriate scopes:
- Format: `PUBLIC_ID$SECRET_KEY`
- Get from: https://numer.ai/account
- Required scope for uploads: `upload_submission`

If the environment variable `NUMERAI_MCP_AUTH` is set, authentication is pre-configured.

### Common Queries

**List account's models**:
```graphql
query { account { models { id name } } }
```

**Get default Python runtime**:
```graphql
query { computePickleDockerImages { id name image tag default } }
```

**Check pickle validation status**:
```graphql
query {
  account {
    models {
      username
      computePickleUpload {
        filename validationStatus triggerStatus
        triggers { id status statuses { status description insertedAt } }
      }
    }
  }
}
```

### PKL Upload Workflow

```
1. create_model(name, tournament=8)           # Optional: create new model slot
2. upload_model(operation="get_upload_auth")  # Get presigned S3 URL
3. PUT file to presigned URL                  # Upload the pkl file
4. upload_model(operation="create")           # Register upload
5. upload_model(operation="list")             # Wait for validation
6. upload_model(operation="assign")           # Assign to model slot
```

---

## Python Environment Setup

**CRITICAL**: Pickle files must be created with a Python version matching Numerai's compute environment to avoid segfaults and binary incompatibility.

### Setup Steps

```bash
# 1. Query default Docker image (via MCP) to get Python version
# Look for default: true, e.g., numerai_predict_py_3_12 = Python 3.12

# 2. Create matching venv with pyenv
PYENV_PY=$(ls -d ~/.pyenv/versions/3.12.* 2>/dev/null | head -1)
$PYENV_PY/bin/python -m venv ./venv

# 3. Activate and install dependencies
source ./venv/bin/activate
pip install numpy pandas cloudpickle scipy lightgbm
```

### Testing Pickles Locally

```bash
docker run -i --rm -v "$PWD:$PWD" \
  ghcr.io/numerai/numerai_predict_py_3_12:a78dedd \
  --debug --model $PWD/model.pkl
```

---

## Modeling Philosophy

- **Model-agnostic pipeline**: `pipeline.py`, `numerai_cv.py`, and metrics stay generic
- **Model-specific logic**: Lives in configs and `agents/code/modeling/models/` wrappers
- **Reproducibility**: All settings captured in config files
- **Accurate validation**: No early stopping leakage; honest OOF performance estimation

---

## Data Handling

### Datasets

Build datasets with `python -m agents.code.data.build_full_datasets`:

| File | Description |
|------|-------------|
| `numerai/v5.2/full.parquet` | Full training data |
| `numerai/v5.2/full_benchmark_models.parquet` | Benchmark model predictions |
| `numerai/v5.2/downsampled_full.parquet` | Every 4th era (fast iteration) |
| `numerai/v5.2/downsampled_full_benchmark_models.parquet` | Downsampled benchmarks |

### Strategy

1. **Scout phase**: Use downsampled data for quick experiments
2. **Scale phase**: Run best configs on full data for final validation

---

## Getting Started with Agent Tasks

### For Research Tasks

1. Read `numerai/agents/AGENTS.md` for detailed instructions
2. Check relevant skills in `numerai/agents/skills/`
3. Look for existing experiments in `numerai/agents/experiments/`
4. Use downsampled data for iteration, full data for final runs

### For Deployment Tasks

1. Use the `numerai-model-upload` skill
2. Verify Python version compatibility first
3. Test pickle locally before uploading
4. Use MCP server for programmatic deployment

### For Understanding the Tournament

1. Start with `hello_numerai.ipynb` for basics
2. Review `feature_neutralization.ipynb` for feature risk
3. Check `target_ensemble.ipynb` for ensemble strategies
4. Use MCP server to query live tournament data

---

## Important Notes

- **Run commands from `numerai/`** (so `agents` is importable), or from repo root with `PYTHONPATH=numerai`
- **Data lives under `numerai/<data_version>/`** (e.g. `numerai/v5.2/`), which is often gitignored locally
- **Register repo skills**: `ln -s $PWD/numerai/agents/skills/* ~/.codex/skills/`
- **Network access required** for MCP operations (Codex CLI may need `--yolo` flag)
- **Always query Python version** before creating pkl files
- **BMC (Benchmark Model Contribution)** is the key experiment metric (proxy for MMC), computed vs official `v52_lgbm_ender20` benchmark predictions in `*_benchmark_models.parquet`
- **Only Classic tournament (8)** supports pickle uploads
