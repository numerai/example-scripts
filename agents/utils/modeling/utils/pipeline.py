from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from numerapi import NumerAPI

from agents.utils.metrics import numerai_metrics
from .config import (
    load_config,
    resolve_predictions_path,
    resolve_results_path,
    resolve_small_bmc_baseline_path,
)
from .constants import (
    DEFAULT_BASELINES_DIR,
    DEFAULT_BENCHMARK_MODEL,
    DEFAULT_LGBM_PARAMS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SMALL_BMC_BASELINE,
)
from .data import (
    apply_missing_all_twos_as_nan,
    attach_baseline_column,
    attach_benchmark_column,
    load_features,
    load_full_data,
)
from .numerai_cv import build_oof_predictions


def resolve_output_locations(
    config: dict, output_dir_override: Path | None
) -> tuple[Path, Path, Path, Path]:
    output_config = config.get("output", {})
    data_config = config.get("data", {})

    output_dir = output_dir_override or output_config.get("output_dir")
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir = output_dir.resolve()

    baselines_dir = (
        output_config.get("baselines_dir")
        or data_config.get("baselines_dir")
        or DEFAULT_BASELINES_DIR
    )
    baselines_dir = Path(baselines_dir).expanduser().resolve()

    results_dir = output_dir / "results"
    predictions_dir = output_dir / "predictions"
    return output_dir, baselines_dir, results_dir, predictions_dir


def resolve_model_config(model_config: dict) -> tuple[str, dict]:
    model_type = model_config.get("type", "LGBMRegressor")
    model_params = model_config.get("params")
    if model_params is None:
        if model_type == "LGBMRegressor":
            model_params = DEFAULT_LGBM_PARAMS
        else:
            model_params = {}
    return model_type, model_params


def load_and_prepare_data(
    napi: NumerAPI,
    data_version: str,
    feature_set: str,
    target_col: str,
    era_col: str,
    id_col: str,
    full_data_path: str | Path | None,
    group_feature: str | None,
    nan_missing_all_twos: bool,
    missing_value: float,
    benchmark_needed: bool,
    benchmark_model: str,
    benchmark_data_path: str | Path | None,
    baseline_needed: bool,
    baseline_name: str,
    baseline_predictions_path: str | Path | None,
) -> tuple[pd.DataFrame, list[str], str | None, str | None]:
    features = load_features(napi, data_version, feature_set)
    extra_cols = [group_feature] if group_feature else []
    full = load_full_data(
        napi,
        data_version,
        features,
        era_col,
        target_col,
        id_col,
        full_data_path=full_data_path,
        extra_cols=extra_cols,
    )

    if nan_missing_all_twos:
        full = apply_missing_all_twos_as_nan(full, features, era_col, missing_value)

    benchmark_col = None
    if benchmark_needed:
        full, benchmark_col = attach_benchmark_column(
            full,
            data_version,
            benchmark_model,
            benchmark_data_path,
            era_col,
            id_col,
        )

    baseline_col = None
    if baseline_needed:
        if baseline_predictions_path is None:
            raise ValueError("baseline_predictions_path is required for baseline transforms.")
        full, baseline_col = attach_baseline_column(
            full,
            baseline_name,
            baseline_predictions_path,
            era_col,
            id_col,
        )
    return full, features, benchmark_col, baseline_col


def select_prediction_columns(
    predictions: pd.DataFrame,
    id_col: str | None,
    era_col: str,
    target_col: str,
) -> pd.DataFrame:
    prediction_cols = [
        col for col in [id_col, era_col, target_col, "prediction", "cv_fold"] if col
    ]
    prediction_cols = [col for col in prediction_cols if col in predictions.columns]
    return predictions[prediction_cols].copy()


def save_predictions(
    predictions: pd.DataFrame,
    config: dict,
    config_path: Path,
    predictions_dir: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = resolve_predictions_path(config, config_path, predictions_dir)
    predictions.to_parquet(predictions_path, index=False)
    print(f"Saved predictions to {predictions_path}")
    predictions_relative = predictions_path.relative_to(output_dir)
    return predictions_path, predictions_relative


def summarize_predictions(
    predictions_path: Path,
    target_col: str,
    data_version: str,
    benchmark_model: str,
    benchmark_data_path: str | None,
    small_bmc_baseline_path: Path,
    small_bmc_baseline_name: str,
    era_col: str,
    id_col: str,
) -> dict:
    return numerai_metrics.summarize_prediction_file_with_bmc(
        predictions_path,
        ["prediction"],
        target_col,
        data_version,
        benchmark_model=benchmark_model,
        benchmark_data_path=benchmark_data_path,
        small_bmc_baseline_path=small_bmc_baseline_path,
        small_bmc_baseline_name=small_bmc_baseline_name,
        era_col=era_col,
        id_col=id_col,
    )


def build_results_payload(
    *,
    model_type: str,
    model_params: dict,
    nan_missing_all_twos: bool,
    missing_value: float,
    data_version: str,
    feature_set: str,
    target_col: str,
    full_data_path: str | Path | None,
    full: pd.DataFrame,
    predictions: pd.DataFrame,
    era_col: str,
    embargo_eras: int,
    benchmark_model: str,
    benchmark_data_path: str | Path | None,
    small_bmc_baseline: str,
    small_bmc_baseline_path: Path,
    output_dir: Path,
    predictions_relative: Path,
    summaries: dict,
    cv_meta: dict,
    cv_enabled: bool,
    era_weighting_enabled: bool,
    era_weighting_scheme: str,
    era_weighting_min: float,
    era_weighting_max: float,
    target_transform: dict,
    prediction_transform: dict,
    ranker_config: dict,
    max_train_samples: int | None,
    sample_seed: int,
) -> dict:
    try:
        small_bmc_baseline_rel = str(small_bmc_baseline_path.relative_to(output_dir))
    except ValueError:
        small_bmc_baseline_rel = str(small_bmc_baseline_path)

    results = {
        "model": {
            "type": model_type,
            "params": model_params,
        },
        "preprocessing": {
            "nan_missing_all_twos": nan_missing_all_twos,
            "missing_value": missing_value,
        },
        "data": {
            "data_version": data_version,
            "feature_set": feature_set,
            "target": target_col,
            "full_data_path": full_data_path,
            "full_rows": int(full.shape[0]),
            "full_eras": int(full[era_col].nunique()),
            "oof_rows": int(predictions.shape[0]),
            "oof_eras": int(predictions[era_col].nunique()),
            "embargo_eras": embargo_eras,
        },
        "benchmark": {
            "model": benchmark_model,
            "file": benchmark_data_path
            or f"{data_version}/full_benchmark_models.parquet",
        },
        "small_bmc_benchmark": {
            "model": small_bmc_baseline,
            "predictions_file": str(small_bmc_baseline_rel),
        },
        "output": {
            "output_dir": str(output_dir),
            "predictions_file": str(predictions_relative),
        },
        "metrics": {
            "corr": summaries["corr"].loc["prediction"].to_dict(),
            "bmc_last_200_eras": summaries["bmc_last_200_eras"]
            .loc["prediction"]
            .to_dict(),
        },
        "cv": cv_meta,
        "training": {
            "era_weighting": {
                "enabled": era_weighting_enabled,
                "scheme": era_weighting_scheme if era_weighting_enabled else None,
                "min_weight": era_weighting_min if era_weighting_enabled else None,
                "max_weight": era_weighting_max if era_weighting_enabled else None,
            },
            "target_transform": target_transform or None,
            "prediction_transform": prediction_transform or None,
            "ranker": {
                "grouping": (
                    ranker_config.get("grouping")
                    if model_type == "LGBMRanker"
                    else None
                ),
                "label_bins": (
                    ranker_config.get("label_bins")
                    if model_type == "LGBMRanker"
                    else None
                ),
            },
            "data_sampling": {
                "max_train_samples": max_train_samples,
                "sample_seed": sample_seed if max_train_samples else None,
            },
            "cv": {
                "enabled": cv_enabled,
                "n_splits": cv_meta["n_splits"],
                "embargo": cv_meta["embargo"],
                "mode": cv_meta["mode"],
                "min_train_size": cv_meta["min_train_size"],
            },
        },
    }
    if "bmc" in summaries:
        results["metrics"]["bmc"] = summaries["bmc"].loc["prediction"].to_dict()
    if "small_bmc" in summaries:
        results["metrics"]["small_bmc"] = (
            summaries["small_bmc"].loc["prediction"].to_dict()
        )
    if "small_bmc_last200" in summaries:
        results["metrics"]["small_bmc_last200"] = (
            summaries["small_bmc_last200"].loc["prediction"].to_dict()
        )
    return results


def save_results(results: dict, results_path: Path) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"Saved results to {results_path}")


def run_training(
    config_path: Path, output_dir_override: Path | None = None
) -> tuple[Path, Path]:
    config = load_config(config_path)

    data_config = config.get("data", {})
    preprocessing_config = config.get("preprocessing", {})
    training_config = config.get("training", {})
    model_config = config.get("model", {})

    data_version = data_config.get("data_version", "v5.2")
    feature_set = data_config.get("feature_set", "small")
    target_col = data_config.get("target_col", "target")
    era_col = data_config.get("era_col", "era")
    id_col = data_config.get("id_col", "id")
    full_data_path = data_config.get("full_data_path")
    benchmark_data_path = data_config.get("benchmark_data_path")
    embargo_eras = data_config.get("embargo_eras", 13)
    benchmark_model = data_config.get("benchmark_model", DEFAULT_BENCHMARK_MODEL)
    small_bmc_baseline = data_config.get(
        "small_bmc_baseline", DEFAULT_SMALL_BMC_BASELINE
    )

    nan_missing_all_twos = preprocessing_config.get("nan_missing_all_twos", False)
    missing_value = preprocessing_config.get("missing_value", 2.0)

    era_weighting = training_config.get("era_weighting", {})
    era_weighting_enabled = era_weighting.get("enabled", False)
    era_weighting_scheme = era_weighting.get("scheme", "linear")
    era_weighting_min = float(era_weighting.get("min_weight", 0.5))
    era_weighting_max = float(era_weighting.get("max_weight", 1.5))
    max_train_samples = training_config.get("max_train_samples")
    if max_train_samples is not None:
        max_train_samples = int(max_train_samples)
    sample_seed = int(training_config.get("sample_seed", 1337))
    ranker_config = training_config.get("ranker", {})
    target_transform = training_config.get("target_transform", {})
    prediction_transform = training_config.get("prediction_transform", {})
    prediction_batch_size = training_config.get("prediction_batch_size")
    target_transform_type = target_transform.get("type")
    prediction_transform_type = prediction_transform.get("type")

    model_type = model_config.get("type", "LGBMRegressor")
    grouping = ranker_config.get("grouping", {}) if model_type == "LGBMRanker" else {}
    group_feature = grouping.get("feature")

    output_dir, baselines_dir, results_dir, predictions_dir = resolve_output_locations(
        config, output_dir_override
    )

    napi = NumerAPI()

    benchmark_needed = (
        target_transform_type == "benchmark_residual"
        or prediction_transform_type == "add_benchmark"
    )
    baseline_needed = (
        target_transform_type == "baseline_residual"
        or prediction_transform_type == "add_baseline"
    )

    small_bmc_baseline_override = (
        data_config.get("small_bmc_baseline_path")
        or config.get("output", {}).get("small_bmc_baseline_path")
    )
    small_bmc_baseline_path = resolve_small_bmc_baseline_path(
        small_bmc_baseline,
        predictions_dir,
        baselines_dir,
        override_path=small_bmc_baseline_override,
    )

    full, features, benchmark_col, baseline_col = load_and_prepare_data(
        napi,
        data_version,
        feature_set,
        target_col,
        era_col,
        id_col,
        full_data_path,
        group_feature,
        nan_missing_all_twos,
        missing_value,
        benchmark_needed,
        benchmark_model,
        benchmark_data_path,
        baseline_needed,
        small_bmc_baseline,
        small_bmc_baseline_path,
    )

    model_type, model_params = resolve_model_config(model_config)

    cv_config = dict(training_config.get("cv", {}))
    cv_config.setdefault("embargo", embargo_eras)
    cv_enabled = cv_config.get("enabled", True)
    if not cv_enabled:
        raise ValueError("CV/OOF pipeline is required for all experiments.")

    predictions, cv_meta = build_oof_predictions(
        full,
        features,
        target_col,
        era_col,
        id_col,
        model_type,
        model_params,
        cv_config,
        era_weighting_enabled,
        era_weighting_scheme,
        era_weighting_min,
        era_weighting_max,
        max_train_samples,
        sample_seed,
        ranker_config,
        target_transform,
        prediction_transform,
        benchmark_col,
        baseline_col,
        prediction_batch_size,
    )

    predictions = select_prediction_columns(predictions, id_col, era_col, target_col)
    predictions_path, predictions_relative = save_predictions(
        predictions, config, config_path, predictions_dir, output_dir
    )

    summaries = summarize_predictions(
        predictions_path,
        target_col,
        data_version,
        benchmark_model,
        benchmark_data_path,
        small_bmc_baseline_path,
        small_bmc_baseline,
        era_col,
        id_col,
    )

    results_path = resolve_results_path(config, config_path, results_dir)
    results = build_results_payload(
        model_type=model_type,
        model_params=model_params,
        nan_missing_all_twos=nan_missing_all_twos,
        missing_value=missing_value,
        data_version=data_version,
        feature_set=feature_set,
        target_col=target_col,
        full_data_path=full_data_path,
        full=full,
        predictions=predictions,
        era_col=era_col,
        embargo_eras=embargo_eras,
        benchmark_model=benchmark_model,
        benchmark_data_path=benchmark_data_path,
        small_bmc_baseline=small_bmc_baseline,
        small_bmc_baseline_path=small_bmc_baseline_path,
        output_dir=output_dir,
        predictions_relative=predictions_relative,
        summaries=summaries,
        cv_meta=cv_meta,
        cv_enabled=cv_enabled,
        era_weighting_enabled=era_weighting_enabled,
        era_weighting_scheme=era_weighting_scheme,
        era_weighting_min=era_weighting_min,
        era_weighting_max=era_weighting_max,
        target_transform=target_transform,
        prediction_transform=prediction_transform,
        ranker_config=ranker_config,
        max_train_samples=max_train_samples,
        sample_seed=sample_seed,
    )
    save_results(results, results_path)

    return predictions_path, results_path
