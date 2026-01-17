from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from numerapi import NumerAPI

from agents.code.metrics import numerai_metrics
from .config import (
    load_config,
    resolve_predictions_path,
    resolve_results_path,
    resolve_small_bmc_baseline_path,
)
from .constants import (
    BASE_DIR,
    DEFAULT_BASELINES_DIR,
    DEFAULT_BENCHMARK_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SMALL_BMC_BASELINE,
)
from .data import (
    apply_missing_all_twos_as_nan,
    attach_baseline_column,
    attach_benchmark_models,
    load_features,
    load_full_data,
)
from .model_data import build_model_data_loader, build_x_cols, normalize_x_groups
from .numerai_cv import build_oof_predictions


def resolve_output_locations(
    config: dict, output_dir_override: Path | None
) -> tuple[Path, Path, Path, Path]:
    output_config = config.get("output", {})
    data_config = config.get("data", {})

    output_dir = _resolve_repo_dir(
        output_dir_override or output_config.get("output_dir"),
        DEFAULT_OUTPUT_DIR,
    )
    baselines_dir = _resolve_repo_dir(
        output_config.get("baselines_dir") or data_config.get("baselines_dir"),
        DEFAULT_BASELINES_DIR,
    )

    results_dir = output_dir / "results"
    predictions_dir = output_dir / "predictions"
    return output_dir, baselines_dir, results_dir, predictions_dir


def _resolve_repo_dir(path: str | Path | None, default: Path) -> Path:
    if not path:
        return default
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.parts and candidate.parts[0] == BASE_DIR.name:
        return (BASE_DIR.parent / candidate).resolve()
    return (BASE_DIR / candidate).resolve()


def resolve_model_config(model_config: dict) -> tuple[str, dict]:
    model_type = model_config.get("type", "LGBMRegressor")
    model_params = model_config.get("params")
    if model_params is None:
        raise ValueError("model.params must be specified; pipeline does not set defaults.")
    return model_type, model_params


def load_and_prepare_data(
    napi: NumerAPI,
    data_version: str,
    feature_set: str,
    target_col: str,
    era_col: str,
    id_col: str,
    full_data_path: str | Path | None,
    nan_missing_all_twos: bool,
    missing_value: float,
) -> tuple[pd.DataFrame, list[str]]:
    features = load_features(napi, data_version, feature_set)
    full = load_full_data(
        napi,
        data_version,
        features,
        era_col,
        target_col,
        id_col,
        full_data_path=full_data_path,
    )

    if nan_missing_all_twos:
        full = apply_missing_all_twos_as_nan(full, features, era_col, missing_value)

    return full, features


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
    model_config: dict,
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
    max_train_samples: int | None,
    sample_seed: int,
) -> dict:
    try:
        small_bmc_baseline_rel = str(small_bmc_baseline_path.relative_to(output_dir))
    except ValueError:
        small_bmc_baseline_rel = str(small_bmc_baseline_path)

    model_meta = {
        "type": model_type,
        "params": model_params,
    }
    for key in (
        "x_groups",
        "data_needed",
        "target_transform",
        "prediction_transform",
        "era_weighting",
        "prediction_batch_size",
        "benchmark",
        "baseline",
    ):
        if key in model_config:
            model_meta[key] = model_config[key]

    results = {
        "model": model_meta,
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

    max_train_samples = training_config.get("max_train_samples")
    if max_train_samples is not None:
        max_train_samples = int(max_train_samples)
    sample_seed = int(training_config.get("sample_seed", 1337))

    output_dir, baselines_dir, results_dir, predictions_dir = resolve_output_locations(
        config, output_dir_override
    )

    napi = NumerAPI()

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

    full, features = load_and_prepare_data(
        napi,
        data_version,
        feature_set,
        target_col,
        era_col,
        id_col,
        full_data_path=full_data_path,
        nan_missing_all_twos=nan_missing_all_twos,
        missing_value=missing_value,
    )

    model_type, model_params = resolve_model_config(model_config)
    raw_x_groups = model_config.get("x_groups") or model_config.get("data_needed")
    x_groups = normalize_x_groups(raw_x_groups)
    benchmark_cols: list[str] = []
    baseline_col: str | None = None

    if "benchmark_models" in x_groups:
        if not id_col:
            raise ValueError("id_col is required to attach benchmark models.")
        full, benchmark_cols = attach_benchmark_models(
            full,
            napi,
            data_version,
            benchmark_data_path,
            era_col,
            id_col,
        )

    if "baseline" in x_groups:
        baseline_spec = model_config.get("baseline", {})
        baseline_name = baseline_spec.get("name")
        baseline_path = baseline_spec.get("predictions_path")
        pred_col = baseline_spec.get("pred_col", "prediction")
        if not baseline_name or not baseline_path:
            raise ValueError(
                "model.baseline.name and model.baseline.predictions_path are required when baseline data is requested."
            )
        if not id_col:
            raise ValueError("id_col is required to attach baseline predictions.")
        full, baseline_col = attach_baseline_column(
            full,
            baseline_name,
            baseline_path,
            era_col,
            id_col,
            pred_col=pred_col,
        )

    x_cols = build_x_cols(
        x_groups=x_groups,
        features=features,
        benchmark_cols=benchmark_cols,
        era_col=era_col,
        id_col=id_col,
        baseline_col=baseline_col,
    )
    data_loader = build_model_data_loader(
        full=full,
        x_cols=x_cols,
        era_col=era_col,
        target_col=target_col,
        id_col=id_col,
    )

    cv_config = dict(training_config.get("cv", {}))
    cv_config.setdefault("embargo", embargo_eras)
    cv_enabled = cv_config.get("enabled", True)
    if not cv_enabled:
        raise ValueError("CV/OOF pipeline is required for all experiments.")

    predictions, cv_meta = build_oof_predictions(
        full[era_col],
        data_loader,
        model_type,
        model_params,
        model_config,
        cv_config,
        max_train_samples,
        sample_seed,
        id_col,
        era_col,
        target_col,
        feature_cols=features,
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
        model_config=model_config,
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
        max_train_samples=max_train_samples,
        sample_seed=sample_seed,
    )
    save_results(results, results_path)

    return predictions_path, results_path
