# Numerai Agent

Goal: play the Numerai Tournament by running experiments and implementing new
research ideas to find models with strong BMC. The end goal
is a stable of relatively unique, high-performing models for submission.

Always check the agents/skills/ folder for skills that match the user request.
We run commands from the `agents/` directory, but data is expected to live under `numerai/data_version`.

## Python Environment Setup

**IMPORTANT**: Before creating pkl files for upload, ensure your Python version matches Numerai's compute environment to avoid binary incompatibility errors (e.g., segfaults from numpy version mismatches).

```graphql
query { computePickleDockerImages { id name image tag default } }
```

Look for the image with `default: true` to determine the required Python version (e.g., `numerai_predict_py_3_12:a78dedd` means Python 3.12).

### Creating the Correct Virtual Environment

Use `pyenv` to create a virtual environment matching Numerai's Python version:

```bash
# 1. Check available pyenv versions
ls ~/.pyenv/versions/

# 2. Create venv with matching Python version (e.g., for Python 3.12)
~/.pyenv/versions/3.12.*/bin/python -m venv ./venv

# 3. Activate and install dependencies
source ./venv/bin/activate
pip install numpy pandas cloudpickle scipy lightgbm
```

When running code, look for a local virtual environment and activate it. Prefer the venv that matches the Numerai default Python version.

## Modeling philosophy
- All model-specific logic belongs in model configs and/or `agents/code/modeling/models/` wrappers.
- `pipeline.py`, `numerai_cv.py`, and metrics/analysis stay model-agnostic: they load requested data, call `fit`/`predict`, and record outputs.
- Keep the pipeline bare-bones; complexity is encapsulated inside the model itself.

## Numerai MCP Server
You may have access to the **Numerai MCP server** (`numerai`) which provides tools for interacting with the Numerai Tournament API:
- `check_api_credentials` - Verify API tokens and scopes
- `create_model` - Create new model slots in tournaments
- `upload_model` - Upload pkl files for automated submissions
- `get_model_profile`, `get_model_performance` - Query model stats
- `get_leaderboard`, `get_tournaments`, `get_current_round` - Tournament info
- `graphql_query` - Custom queries (e.g., list account's models)

If the user has the MCP installed, you can assume that they have set their authentication correctly to be used by an agent. You can confirm this by checking the existence of an environment variable called `NUMERAI_MCP_AUTH`. The MCP configuration should be already set to use this.

After creating pkl files, you can deploy directly to Numerai using these tools.

See the `numerai-model-upload` skill for the complete deployment workflow. If the user has not provided an existing model name or ID to upload this to, and this is a new model, run the `create_model` MCP tool using the model name the user has provided, and if they have not provided one, create one that makes sense based on the prompt.

To upload a pkl file, your agent will need network access. If the PUT call to the presigned S3 url fails due to DNS resolution, it's likely that the agent does not have networking access. To fix this in Codex CLI, you need to run it with the `--yolo` flag.

## Organization
- `agents/code/` contains shared packages for training, data, analysis, and metrics.
- `agents/code/modeling/` contains the training package: `models/` for model-specific code, `utils/` for model-agnostic pipeline/config/data helpers, and `python -m agents.code.modeling` as the CLI entrypoint.
- `agents/code/metrics/` holds shared scoring utilities (BMC/corr summaries).
- `agents/code/analysis/` holds reporting/plotting helpers (`show_experiment.py`, `plot_benchmark_corrs.py`, `summarize_results.py`).
- `agents/code/data/` holds dataset builders (full + downsampled parquet).
- `agents/baselines/` stores baseline configs plus their predictions/results.
- `agents/experiments/<name>/` contains experiment-specific configs, logs, predictions, results, and `experiment.md`.
- `agents/skills/` contains Codex skills for this repo (each skill has a `SKILL.md`).
