from __future__ import annotations

import json
import runpy
from pathlib import Path


def load_config(config_path: Path) -> dict:
    if config_path.suffix == ".py":
        module_vars = runpy.run_path(str(config_path))
        if "CONFIG" in module_vars:
            return module_vars["CONFIG"]
        if "config" in module_vars:
            return module_vars["config"]
        raise ValueError(
            f"Config file {config_path} must define CONFIG (or config) dict."
        )
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_results_path(config: dict, config_path: Path, results_dir: Path) -> Path:
    output_config = config.get("output", {})
    results_name = output_config.get("results_name") or config_path.stem
    return results_dir / f"{results_name}.json"


def resolve_predictions_path(
    config: dict, config_path: Path, predictions_dir: Path
) -> Path:
    output_config = config.get("output", {})
    predictions_name = output_config.get("predictions_name")
    if not predictions_name:
        predictions_name = output_config.get("results_name") or config_path.stem
    return predictions_dir / f"{predictions_name}.parquet"
