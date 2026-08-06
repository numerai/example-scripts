"""Generate target-versioned example predictions and provenance for Data publication."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
from numerapi import NumerAPI


DATA_VERSION = "v5.3"
TARGET_COL = "target_ender_60"
MODEL_ARTIFACT = "example_model_v53_ender60.pkl"
SOURCE_NOTEBOOK = "numerai/example_model.ipynb"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predict(model: object, features: pd.DataFrame) -> pd.DataFrame:
    """Run either supported Model Upload callable signature and validate output."""
    if len(inspect.signature(model).parameters) == 1:
        predictions = model(features)
    else:
        predictions = model(features, pd.DataFrame(index=features.index))

    if not isinstance(predictions, pd.DataFrame) or list(predictions) != ["prediction"]:
        raise ValueError("model must return a DataFrame with one 'prediction' column")
    if not predictions.index.equals(features.index):
        raise ValueError("prediction index does not match the input feature index")
    if not np.isfinite(predictions["prediction"]).all():
        raise ValueError("predictions contain non-finite values")
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.model.open("rb") as file:
        model = cloudpickle.load(file)

    napi = NumerAPI()
    features_path = Path(DATA_VERSION) / "features.json"
    napi.download_dataset(f"{DATA_VERSION}/features.json")
    feature_metadata = json.loads(features_path.read_text())
    feature_cols = feature_metadata["feature_sets"]["small"]

    outputs: dict[str, dict[str, object]] = {}
    for split in ("validation", "live"):
        dataset_path = Path(DATA_VERSION) / f"{split}.parquet"
        napi.download_dataset(f"{DATA_VERSION}/{split}.parquet")
        split_features = pd.read_parquet(dataset_path, columns=feature_cols)
        predictions = predict(model, split_features)
        output_path = args.output_dir / f"{split}_example_preds_v53_ender60.parquet"
        predictions.to_parquet(output_path)
        outputs[split] = {
            "file": output_path.name,
            "rows": len(predictions),
            "sha256": sha256(output_path),
        }

    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": os.environ.get("GITHUB_SHA", "local"),
        "source_notebook": SOURCE_NOTEBOOK,
        "dataset_version": DATA_VERSION,
        "target_col": TARGET_COL,
        "model_artifact": args.model.name,
        "model_sha256": sha256(args.model),
        "predictions": outputs,
    }
    provenance_path = args.output_dir / "v53_ender60_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
