from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numerapi import NumerAPI

from agents.utils.metrics import numerai_metrics


def load_features(napi: NumerAPI, data_version: str, feature_set: str) -> list[str]:
    napi.download_dataset(f"{data_version}/features.json")
    with open(f"{data_version}/features.json", "r", encoding="utf-8") as f:
        feature_metadata = json.load(f)
    return feature_metadata["feature_sets"][feature_set]


def ensure_full_dataset(napi: NumerAPI, data_version: str) -> Path:
    full_path = Path(f"{data_version}/full.parquet")
    if full_path.exists():
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


def load_full_data(
    napi: NumerAPI,
    data_version: str,
    features: list[str],
    era_col: str,
    target_col: str,
    id_col: str | None,
    full_data_path: str | Path | None = None,
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    if full_data_path:
        full_path = Path(full_data_path).expanduser().resolve()
        if not full_path.exists():
            raise FileNotFoundError(f"Full data file not found: {full_path}")
    else:
        full_path = ensure_full_dataset(napi, data_version)
    extra_cols = extra_cols or []
    columns = [era_col, target_col] + features + extra_cols
    seen = set()
    columns = [col for col in columns if not (col in seen or seen.add(col))]
    if id_col:
        columns.append(id_col)
    try:
        df = pd.read_parquet(full_path, columns=columns)
    except Exception:
        df = pd.read_parquet(full_path, columns=[era_col, target_col] + features)
        if id_col and id_col not in df.columns:
            df[id_col] = df.index
    return df


def apply_missing_all_twos_as_nan(
    df: pd.DataFrame,
    feature_cols: list[str],
    era_col: str,
    missing_value: float,
) -> pd.DataFrame:
    """Replace per-era feature values with NaN when the entire era is missing (all 2s)."""
    features = df[feature_cols].astype("float32")
    is_all_twos = features.eq(missing_value).groupby(df[era_col]).transform("all")
    features = features.mask(is_all_twos, np.nan)
    non_features = df.drop(columns=feature_cols)
    updated = pd.concat([non_features, features], axis=1)
    return updated[df.columns]


def attach_benchmark_column(
    full: pd.DataFrame,
    data_version: str,
    benchmark_model: str,
    benchmark_data_path: str | Path | None,
    era_col: str,
    id_col: str,
) -> tuple[pd.DataFrame, str]:
    if benchmark_data_path:
        benchmark, benchmark_col = numerai_metrics.load_benchmark_predictions_from_path(
            benchmark_data_path,
            benchmark_model,
            era_col=era_col,
            id_col=id_col,
        )
    else:
        benchmark, benchmark_col = numerai_metrics.load_benchmark_predictions(
            data_version, benchmark_model=benchmark_model, era_col=era_col
        )
    updated = numerai_metrics.attach_benchmark_predictions(
        full, benchmark, benchmark_col, era_col=era_col, id_col=id_col
    )
    return updated, benchmark_col


def attach_baseline_column(
    full: pd.DataFrame,
    baseline_name: str,
    baseline_predictions_path: str | Path,
    era_col: str,
    id_col: str,
    pred_col: str = "prediction",
) -> tuple[pd.DataFrame, str]:
    baseline_path = Path(baseline_predictions_path).expanduser().resolve()
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline predictions file not found: {baseline_path}")
    baseline, baseline_col = numerai_metrics.load_custom_benchmark_predictions(
        baseline_path,
        baseline_name,
        pred_col=pred_col,
        era_col=era_col,
        id_col=id_col,
    )
    updated = numerai_metrics.attach_benchmark_predictions(
        full, baseline, baseline_col, era_col=era_col, id_col=id_col
    )
    return updated, baseline_col
