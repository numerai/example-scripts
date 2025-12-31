"""Build full dataset and benchmark parquet files for Numerai."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from numerapi import NumerAPI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build full.parquet and full_benchmark_models.parquet."
    )
    parser.add_argument(
        "--data-version",
        type=str,
        default="v5.2",
        help="Numerai data version (default: v5.2).",
    )
    return parser.parse_args()


def build_full_dataset(napi: NumerAPI, data_version: str) -> Path:
    full_path = Path(f"{data_version}/full.parquet")
    train_path = f"{data_version}/train.parquet"
    validation_path = f"{data_version}/validation.parquet"

    napi.download_dataset(train_path)
    napi.download_dataset(validation_path)

    train = pd.read_parquet(train_path)
    validation = pd.read_parquet(validation_path)
    validation = validation[validation["data_type"] == "validation"].copy()

    full = pd.concat([train, validation], ignore_index=False)
    full = full.drop(columns=["data_type"], errors="ignore")
    if full.index.name and full.index.name not in full.columns:
        full = full.reset_index()
    full.to_parquet(full_path, index=False)
    return full_path


def build_full_benchmark(napi: NumerAPI, data_version: str) -> Path:
    full_path = Path(f"{data_version}/full_benchmark_models.parquet")
    train_path = f"{data_version}/train_benchmark_models.parquet"
    validation_path = f"{data_version}/validation_benchmark_models.parquet"
    validation_data_path = f"{data_version}/validation.parquet"

    napi.download_dataset(train_path)
    napi.download_dataset(validation_path)
    napi.download_dataset(validation_data_path)

    validation_meta = pd.read_parquet(validation_data_path, columns=["data_type"])
    validation_meta = validation_meta[validation_meta["data_type"] == "validation"]
    validation_ids = validation_meta.index

    train = pd.read_parquet(train_path)
    validation = pd.read_parquet(validation_path)
    if "id" in train.columns:
        train = train.set_index("id")
    if "id" in validation.columns:
        validation = validation.set_index("id")
    validation = validation.loc[validation.index.intersection(validation_ids)]

    full = pd.concat([train, validation], axis=0)
    full.to_parquet(full_path)
    return full_path


def main() -> None:
    args = parse_args()
    data_version = args.data_version
    napi = NumerAPI()

    full_data = build_full_dataset(napi, data_version)
    full_benchmark = build_full_benchmark(napi, data_version)

    print(f"Built {full_data}")
    print(f"Built {full_benchmark}")


if __name__ == "__main__":
    main()
