"""Reusable Numerai scoring helpers built on numerai_tools."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numerapi import NumerAPI
from numerai_tools.scoring import correlation_contribution, numerai_corr


def _as_list(cols: Iterable) -> List:
    if isinstance(cols, (list, tuple, pd.Index, np.ndarray)):
        return list(cols)
    return [cols]


def _sort_era_index(scores: pd.DataFrame | pd.Series):
    if scores.index.dtype == object:
        try:
            order = sorted(scores.index, key=lambda x: int(x))
            return scores.loc[order]
        except (TypeError, ValueError):
            return scores.sort_index()
    return scores.sort_index()


def _normalize_per_era(per_era, pred_cols: Sequence) -> pd.DataFrame:
    if isinstance(per_era, pd.DataFrame):
        return _sort_era_index(per_era)
    if isinstance(per_era, pd.Series):
        if per_era.index.nlevels > 1:
            per_era = per_era.unstack()
            return _sort_era_index(per_era)
        return _sort_era_index(per_era.to_frame(pred_cols[0]))
    raise TypeError("Unexpected per-era score type")


def _last_n_eras(per_era_scores: pd.DataFrame | pd.Series, n: int):
    if n <= 0:
        raise ValueError("n must be > 0")
    per_era_scores = _sort_era_index(per_era_scores)
    return per_era_scores.tail(n)


def per_era_corr(
    df: pd.DataFrame,
    pred_cols: Sequence,
    target_col: str,
    era_col: str = "era",
) -> pd.DataFrame:
    """Compute per-era Numerai correlation for one or more prediction columns."""
    pred_cols = _as_list(pred_cols)
    df = df.dropna(subset=pred_cols + [target_col])

    def _corr(group):
        return numerai_corr(group[pred_cols], group[target_col])

    per_era = df.groupby(era_col).apply(_corr)
    return _normalize_per_era(per_era, pred_cols)


def per_era_mmc(
    df: pd.DataFrame,
    pred_cols: Sequence,
    meta_col: str,
    target_col: str,
    era_col: str = "era",
) -> pd.DataFrame:
    """Compute per-era MMC using Numerai's meta model predictions."""
    pred_cols = _as_list(pred_cols)
    df = df.dropna(subset=pred_cols + [meta_col, target_col])

    def _mmc(group):
        return correlation_contribution(
            group[pred_cols], group[meta_col], group[target_col]
        )

    per_era = df.groupby(era_col).apply(_mmc)
    return _normalize_per_era(per_era, pred_cols)


def per_era_pred_corr(
    df: pd.DataFrame,
    pred_cols: Sequence,
    benchmark_col: str,
    era_col: str = "era",
) -> pd.DataFrame:
    """Compute per-era correlation between predictions and benchmark predictions."""
    pred_cols = _as_list(pred_cols)
    df = df.dropna(subset=pred_cols + [benchmark_col])

    def _corr(group):
        return numerai_corr(group[pred_cols], group[benchmark_col])

    per_era = df.groupby(era_col).apply(_corr)
    return _normalize_per_era(per_era, pred_cols)


def per_era_bmc(
    df: pd.DataFrame,
    pred_cols: Sequence,
    benchmark_col: str,
    target_col: str,
    era_col: str = "era",
) -> pd.DataFrame:
    """Compute per-era benchmark contribution using the benchmark predictions."""
    pred_cols = _as_list(pred_cols)
    df = df.dropna(subset=pred_cols + [benchmark_col, target_col])

    def _bmc(group):
        return correlation_contribution(
            group[pred_cols], group[benchmark_col], group[target_col]
        )

    per_era = df.groupby(era_col).apply(_bmc)
    return _normalize_per_era(per_era, pred_cols)


def max_drawdown(scores: pd.Series) -> float:
    scores = scores.dropna()
    if scores.empty:
        return np.nan
    cumsum = scores.cumsum()
    running_max = cumsum.expanding(min_periods=1).max()
    return (running_max - cumsum).max()


def score_summary(scores: pd.Series) -> dict:
    scores = scores.dropna()
    if scores.empty:
        return {
            "mean": np.nan,
            "std": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "consistency": np.nan,
        }
    mean = scores.mean()
    std = scores.std(ddof=0)
    sharpe = mean / std if std != 0 else np.nan
    return {
        "mean": mean,
        "std": std,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(scores),
        "consistency": (scores > 0).mean(),
    }


def summarize_scores(per_era_scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-era score DataFrame into mean/std/sharpe/drawdown/consistency."""
    per_era_scores = _sort_era_index(per_era_scores)
    summary = {col: score_summary(per_era_scores[col]) for col in per_era_scores}
    return pd.DataFrame(summary).T


def _parquet_columns(path: str | Path) -> list[str]:
    return pq.ParquetFile(path).schema.names


def _resolve_benchmark_column(columns: Iterable[str], benchmark_model: str) -> str:
    if benchmark_model in columns:
        return benchmark_model
    suffix = f"_{benchmark_model}"
    matches = [col for col in columns if col.endswith(suffix)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"Benchmark model '{benchmark_model}' not found in columns.")
    raise ValueError(
        f"Benchmark model '{benchmark_model}' matched multiple columns: {matches}"
    )


def load_custom_benchmark_predictions(
    predictions_path: str | Path,
    benchmark_name: str,
    pred_col: str = "prediction",
    era_col: str = "era",
    id_col: str = "id",
) -> tuple[pd.DataFrame, str]:
    """Load predictions from a local file to use as a benchmark column."""
    columns = _parquet_columns(predictions_path)
    required = [col for col in [era_col, pred_col, id_col] if col in columns]
    benchmark = pd.read_parquet(predictions_path, columns=required)
    if pred_col not in benchmark.columns:
        raise ValueError(
            f"Expected '{pred_col}' column in benchmark file: {predictions_path}"
        )
    if pred_col != benchmark_name:
        benchmark = benchmark.rename(columns={pred_col: benchmark_name})
    return benchmark, benchmark_name


def ensure_full_benchmark_models(napi: NumerAPI, data_version: str) -> Path:
    full_path = Path(f"{data_version}/full_benchmark_models.parquet")
    if full_path.exists():
        return full_path

    train_path = Path(f"{data_version}/train_benchmark_models.parquet")
    validation_path = Path(f"{data_version}/validation_benchmark_models.parquet")
    validation_data_path = Path(f"{data_version}/validation.parquet")
    if not train_path.exists():
        napi.download_dataset(str(train_path))
    if not validation_path.exists():
        napi.download_dataset(str(validation_path))
    if not validation_data_path.exists():
        napi.download_dataset(str(validation_data_path))

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


def load_benchmark_predictions(
    data_version: str,
    split: str = "full",
    benchmark_model: str = "v52_lgbm_ender20",
    era_col: str = "era",
) -> tuple[pd.DataFrame, str]:
    """Download and load benchmark predictions for the given data split."""
    napi = NumerAPI()
    if split == "full":
        dataset = str(ensure_full_benchmark_models(napi, data_version))
    else:
        dataset = f"{data_version}/{split}_benchmark_models.parquet"
        napi.download_dataset(dataset)
    columns = _parquet_columns(dataset)
    benchmark_col = _resolve_benchmark_column(columns, benchmark_model)
    benchmark = pd.read_parquet(dataset, columns=[era_col, benchmark_col])
    if benchmark.index.name is None:
        benchmark.index.name = "id"
    return benchmark, benchmark_col


def load_benchmark_predictions_from_path(
    benchmark_path: str | Path,
    benchmark_model: str,
    era_col: str = "era",
    id_col: str = "id",
) -> tuple[pd.DataFrame, str]:
    """Load benchmark predictions from a local parquet path."""
    path = Path(benchmark_path).expanduser().resolve()
    columns = _parquet_columns(path)
    benchmark_col = _resolve_benchmark_column(columns, benchmark_model)
    read_cols = [benchmark_col]
    if id_col in columns:
        read_cols.append(id_col)
    if era_col in columns:
        read_cols.append(era_col)
    benchmark = pd.read_parquet(path, columns=read_cols)
    if id_col in benchmark.columns:
        benchmark = benchmark.set_index(id_col)
    if benchmark.index.name is None:
        benchmark.index.name = id_col
    return benchmark, benchmark_col


def attach_benchmark_predictions(
    predictions: pd.DataFrame,
    benchmark: pd.DataFrame,
    benchmark_col: str,
    era_col: str = "era",
    id_col: str = "id",
) -> pd.DataFrame:
    """Align benchmark predictions to the model predictions and attach the column."""
    if id_col in predictions.columns:
        preds = predictions.set_index(id_col)
        bench = benchmark
        if bench.index.name != id_col:
            if id_col in bench.columns:
                bench = bench.set_index(id_col)
            else:
                raise ValueError(f"Benchmark data missing id column '{id_col}'.")
        common_ids = preds.index.intersection(bench.index)
        if common_ids.empty:
            raise ValueError("No overlapping ids between predictions and benchmark.")
        preds = preds.loc[common_ids]
        bench = bench.loc[common_ids]
        if era_col in bench.columns and not np.array_equal(
            bench[era_col].astype(str).to_numpy(),
            preds[era_col].astype(str).to_numpy(),
        ):
            raise ValueError("Benchmark eras do not align with predictions by id.")
        preds[benchmark_col] = bench[benchmark_col].to_numpy()
        return preds.reset_index()

    bench = benchmark[benchmark[era_col].isin(predictions[era_col])]
    if len(bench) != len(predictions):
        raise ValueError(
            "Benchmark rows do not match predictions; include id in predictions."
        )
    if not np.array_equal(bench[era_col].to_numpy(), predictions[era_col].to_numpy()):
        raise ValueError(
            "Benchmark eras do not align with predictions; include id in predictions."
        )
    enriched = predictions.copy()
    enriched[benchmark_col] = bench[benchmark_col].to_numpy()
    return enriched


def summarize_prediction_file(
    predictions_path: str | Path,
    pred_cols: Sequence,
    target_col: str,
    era_col: str = "era",
) -> dict[str, pd.DataFrame]:
    """Load a predictions parquet file and summarize per-era correlation metrics."""
    pred_cols = _as_list(pred_cols)
    required_cols = []
    columns = _parquet_columns(predictions_path)
    for col in [era_col, target_col, *pred_cols]:
        if col not in required_cols:
            required_cols.append(col)
    if "id" in columns:
        required_cols.append("id")
    predictions = pd.read_parquet(predictions_path, columns=required_cols)
    per_era = per_era_corr(predictions, pred_cols, target_col, era_col)
    return {"corr": summarize_scores(per_era)}


def summarize_prediction_file_with_bmc(
    predictions_path: str | Path,
    pred_cols: Sequence,
    target_col: str,
    data_version: str,
    benchmark_model: str = "v52_lgbm_ender20",
    benchmark_data_path: str | Path | None = None,
    era_col: str = "era",
    id_col: str = "id",
    small_bmc_baseline_path: str | Path | None = None,
    small_bmc_baseline_name: str = "small_lgbm_ender20_baseline",
) -> dict[str, pd.DataFrame]:
    """Load predictions, attach benchmark model, and summarize corr and BMC metrics."""
    summaries = summarize_prediction_file(
        predictions_path, pred_cols, target_col, era_col=era_col
    )
    columns = _parquet_columns(predictions_path)
    required_cols = [era_col, target_col, *_as_list(pred_cols)]
    if id_col in columns:
        required_cols.append(id_col)
    predictions = pd.read_parquet(predictions_path, columns=required_cols)
    if benchmark_data_path is not None:
        benchmark_path = Path(benchmark_data_path).expanduser().resolve()
        if not benchmark_path.exists():
            raise FileNotFoundError(f"Benchmark data file not found: {benchmark_path}")
        columns = _parquet_columns(benchmark_path)
        benchmark_col = _resolve_benchmark_column(columns, benchmark_model)
        read_cols = [benchmark_col]
        if id_col in columns:
            read_cols.append(id_col)
        if era_col in columns:
            read_cols.append(era_col)
        benchmark = pd.read_parquet(benchmark_path, columns=read_cols)
        if id_col in benchmark.columns:
            benchmark = benchmark.set_index(id_col)
        if benchmark.index.name is None:
            benchmark.index.name = "id"
    else:
        benchmark, benchmark_col = load_benchmark_predictions(
            data_version, benchmark_model=benchmark_model, era_col=era_col
        )
    predictions = attach_benchmark_predictions(
        predictions,
        benchmark,
        benchmark_col,
        era_col=era_col,
        id_col=id_col,
    )
    per_era = per_era_bmc(predictions, pred_cols, benchmark_col, target_col, era_col)
    benchmark_corr = per_era_pred_corr(
        predictions, pred_cols, benchmark_col, era_col=era_col
    )

    bmc_summary = summarize_scores(per_era)
    benchmark_corr_mean = benchmark_corr.mean()
    for col in bmc_summary.index:
        bmc_summary.loc[col, "avg_corr_with_benchmark"] = float(
            benchmark_corr_mean.get(col, np.nan)
        )
    summaries["bmc"] = bmc_summary

    per_era_recent = _last_n_eras(per_era, 200)
    bmc_recent_summary = summarize_scores(per_era_recent)
    benchmark_corr_recent = _last_n_eras(benchmark_corr, 200)
    benchmark_corr_recent_mean = benchmark_corr_recent.mean()
    for col in bmc_recent_summary.index:
        bmc_recent_summary.loc[col, "avg_corr_with_benchmark"] = float(
            benchmark_corr_recent_mean.get(col, np.nan)
        )
    summaries["bmc_last_200_eras"] = bmc_recent_summary

    if small_bmc_baseline_path is not None:
        baseline_path = Path(small_bmc_baseline_path)
        if baseline_path.exists():
            baseline, baseline_col = load_custom_benchmark_predictions(
                baseline_path,
                small_bmc_baseline_name,
                pred_col="prediction",
                era_col=era_col,
                id_col=id_col,
            )
            baseline_predictions = attach_benchmark_predictions(
                predictions,
                baseline,
                baseline_col,
                era_col=era_col,
                id_col=id_col,
            )
            per_era_small = per_era_bmc(
                baseline_predictions,
                pred_cols,
                baseline_col,
                target_col,
                era_col,
            )
            small_corr = per_era_pred_corr(
                baseline_predictions, pred_cols, baseline_col, era_col=era_col
            )
            small_corr_mean = small_corr.mean()
            small_bmc_summary = summarize_scores(per_era_small)
            for col in small_bmc_summary.index:
                small_bmc_summary.loc[col, "avg_corr_with_benchmark"] = float(
                    small_corr_mean.get(col, np.nan)
                )
            summaries["small_bmc"] = small_bmc_summary

            per_era_small_recent = _last_n_eras(per_era_small, 200)
            small_bmc_recent_summary = summarize_scores(per_era_small_recent)
            small_corr_recent = _last_n_eras(small_corr, 200)
            small_corr_recent_mean = small_corr_recent.mean()
            for col in small_bmc_recent_summary.index:
                small_bmc_recent_summary.loc[col, "avg_corr_with_benchmark"] = float(
                    small_corr_recent_mean.get(col, np.nan)
                )
            summaries["small_bmc_last200"] = small_bmc_recent_summary
    return summaries
