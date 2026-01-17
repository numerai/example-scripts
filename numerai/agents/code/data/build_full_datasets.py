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
    parser.add_argument(
        "--downsample-eras-step",
        type=int,
        default=4,
        help="Keep every Nth era when building downsampled_full (default: 4).",
    )
    parser.add_argument(
        "--downsample-eras-offset",
        type=int,
        default=0,
        help="Offset when selecting every Nth era (default: 0).",
    )
    parser.add_argument(
        "--skip-downsample",
        action="store_true",
        help="Skip building downsampled_full datasets.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild full datasets even if they already exist.",
    )
    return parser.parse_args()


def build_full_dataset(
    napi: NumerAPI, data_version: str, reuse_existing: bool = True
) -> Path:
    full_path = Path(f"{data_version}/full.parquet")
    if reuse_existing and full_path.exists():
        return full_path
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


def build_full_benchmark(
    napi: NumerAPI, data_version: str, reuse_existing: bool = True
) -> Path:
    full_path = Path(f"{data_version}/full_benchmark_models.parquet")
    if reuse_existing and full_path.exists():
        return full_path
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


def build_downsampled_full_dataset(
    full_path: Path,
    data_version: str,
    era_step: int,
    era_offset: int,
) -> Path:
    if era_step < 2:
        raise ValueError("downsample-eras-step must be >= 2.")
    if era_offset < 0 or era_offset >= era_step:
        raise ValueError("downsample-eras-offset must be in [0, downsample-eras-step).")
    downsampled_path = Path(f"{data_version}/downsampled_full.parquet")
    full = pd.read_parquet(full_path)
    era_col = "era"
    if era_col not in full.columns:
        raise ValueError(f"{full_path} missing '{era_col}' column.")
    unique_eras = sorted(full[era_col].unique(), key=lambda x: int(x))
    keep_eras = {
        era for idx, era in enumerate(unique_eras) if idx % era_step == era_offset
    }
    downsampled = full[full[era_col].isin(keep_eras)].copy()
    downsampled.to_parquet(downsampled_path, index=False)
    return downsampled_path


def build_downsampled_full_benchmark(
    full_benchmark_path: Path,
    downsampled_full_path: Path,
    data_version: str,
) -> Path:
    downsampled_path = Path(f"{data_version}/downsampled_full_benchmark_models.parquet")
    ids = pd.read_parquet(downsampled_full_path, columns=["id"])
    if "id" not in ids.columns:
        raise ValueError(f"{downsampled_full_path} missing 'id' column.")
    id_values = ids["id"].dropna().unique()
    benchmark = pd.read_parquet(full_benchmark_path)
    if "id" in benchmark.columns:
        benchmark = benchmark.set_index("id")
    benchmark = benchmark.loc[benchmark.index.intersection(id_values)]
    benchmark.to_parquet(downsampled_path)
    return downsampled_path


def main() -> None:
    args = parse_args()
    data_version = args.data_version
    napi = NumerAPI()
    reuse_existing = not args.rebuild

    full_data = build_full_dataset(napi, data_version, reuse_existing=reuse_existing)
    full_benchmark = build_full_benchmark(
        napi, data_version, reuse_existing=reuse_existing
    )

    print(f"Built {full_data}")
    print(f"Built {full_benchmark}")
    if not args.skip_downsample:
        downsampled_full = build_downsampled_full_dataset(
            full_data,
            data_version,
            args.downsample_eras_step,
            args.downsample_eras_offset,
        )
        downsampled_benchmark = build_downsampled_full_benchmark(
            full_benchmark,
            downsampled_full,
            data_version,
        )
        print(f"Built {downsampled_full}")
        print(f"Built {downsampled_benchmark}")


if __name__ == "__main__":
    main()
