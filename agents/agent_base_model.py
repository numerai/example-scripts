"""Train and evaluate a basic Numerai model using a JSON config."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from numerapi import NumerAPI

import numerai_cv
import numerai_metrics

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "baselines"
    / "configs"
    / "small_lgbm_ender20_baseline.py"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINES_DIR = Path(__file__).resolve().parent / "baselines"
DEFAULT_BENCHMARK_MODEL = "ender20"
DEFAULT_SMALL_BMC_BASELINE = "small_lgbm_ender20_baseline"
DEFAULT_LGBM_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "max_depth": 5,
    "num_leaves": 2**5 - 1,
    "colsample_bytree": 0.1,
    "min_data_in_leaf": 10000,
    "n_jobs": 30,
    "random_state": 1337,
}


def load_features(napi: NumerAPI, data_version: str, feature_set: str) -> list[str]:
    napi.download_dataset(f"{data_version}/features.json")
    feature_metadata = json.load(open(f"{data_version}/features.json"))
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
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
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


def get_embargo_eras(train_eras: list[str], embargo: int) -> list[str]:
    last_train_era = int(train_eras[-1])
    return [
        str(era).zfill(4)
        for era in range(last_train_era + 1, last_train_era + embargo + 1)
    ]


def build_model(model_type: str, model_params: dict):
    if model_type == "LGBMRegressor":
        return lgb.LGBMRegressor(**model_params)
    if model_type == "LGBMRanker":
        return lgb.LGBMRanker(**model_params)
    if model_type == "CatBoostRegressor":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(**model_params)
    if model_type == "MLPRegressor":
        try:
            from sklearn.neural_network import MLPRegressor
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for MLPRegressor. Install with `.venv/bin/pip install scikit-learn`."
            ) from exc

        return MLPRegressor(**model_params)
    if model_type in {"TabPFNRegressor", "TabPFNClassifier"}:
        try:
            import tabpfn
        except ImportError as exc:
            raise ImportError(
                "tabpfn is required for TabPFN models. Install with `.venv/bin/pip install tabpfn`."
            ) from exc
        model_class = getattr(tabpfn, model_type, None)
        if model_class is None:
            raise ValueError(
                f"tabpfn does not expose {model_type}; upgrade tabpfn or choose a supported model."
            )
        return model_class(**model_params)
    raise ValueError(f"Unsupported model type: {model_type}")


def build_era_weights(
    eras: pd.Series,
    scheme: str,
    min_weight: float,
    max_weight: float,
) -> np.ndarray:
    unique_eras = sorted(eras.unique(), key=lambda x: int(x))
    if len(unique_eras) < 2:
        return np.ones(len(eras), dtype="float32")
    era_index = {era: idx for idx, era in enumerate(unique_eras)}
    ranks = eras.map(era_index).astype("float32")
    if scheme == "linear":
        weights = min_weight + (max_weight - min_weight) * (ranks / (len(unique_eras) - 1))
    elif scheme == "exponential":
        if min_weight <= 0 or max_weight <= 0:
            raise ValueError("min_weight and max_weight must be > 0 for exponential scheme.")
        scale = np.log(max_weight / min_weight)
        weights = min_weight * np.exp(scale * (ranks / (len(unique_eras) - 1)))
    else:
        raise ValueError(f"Unknown era weighting scheme: {scheme}")
    return weights.to_numpy()


def build_ranker_labels(target: pd.Series, n_bins: int) -> pd.Series:
    if n_bins < 2:
        raise ValueError("ranker label_bins must be >= 2.")
    if target.isna().any():
        raise ValueError("Ranker targets contain NaNs; cannot bin labels.")
    ranks = target.rank(method="average", pct=True)
    labels = np.floor(ranks * n_bins).astype(int)
    labels = labels.clip(lower=0, upper=n_bins - 1)
    return labels


def ensure_label_gain(model_params: dict, n_bins: int) -> dict:
    if "label_gain" in model_params:
        return model_params
    updated = dict(model_params)
    updated["label_gain"] = list(range(n_bins))
    return updated


def build_group_key(
    df: pd.DataFrame,
    strategy: str,
    era_col: str,
    feature_col: str | None = None,
) -> pd.Series:
    if strategy == "era":
        return df[era_col].astype(str)
    if strategy == "era_x_feature":
        if not feature_col:
            raise ValueError("feature is required for era_x_feature grouping.")
        if feature_col not in df.columns:
            raise ValueError(f"grouping feature {feature_col} not found in data.")
        feature_key = df[feature_col].where(df[feature_col].notna(), "nan").astype(str)
        return df[era_col].astype(str) + "_" + feature_key
    raise ValueError(f"Unknown grouping strategy: {strategy}")


def prepare_ranker_data(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    group_key: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[int]]:
    grouped = df.assign(__group_key=group_key).sort_values("__group_key")
    group_sizes = grouped.groupby("__group_key", sort=False).size().astype(int).tolist()
    return grouped, grouped[features], grouped[target_col], group_sizes


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


def build_oof_predictions(
    full: pd.DataFrame,
    features: list[str],
    target_col: str,
    era_col: str,
    id_col: str,
    model_type: str,
    model_params: dict,
    cv_config: dict,
    era_weighting_enabled: bool,
    era_weighting_scheme: str,
    era_weighting_min: float,
    era_weighting_max: float,
    max_train_samples: int | None,
    sample_seed: int,
    ranker_config: dict | None,
) -> tuple[pd.DataFrame, dict]:
    cv_n_splits = int(cv_config.get("n_splits", 5))
    cv_embargo = int(cv_config.get("embargo", 13))
    cv_mode = cv_config.get("mode", "expanding")
    cv_min_train_size = int(cv_config.get("min_train_size", 0))

    splits = numerai_cv.era_cv_splits(
        full[era_col],
        n_splits=cv_n_splits,
        embargo=cv_embargo,
        mode=cv_mode,
        min_train_size=cv_min_train_size,
    )

    predictions = []
    fold_info = []
    is_tabpfn = model_type.startswith("TabPFN")
    is_tabpfn_classifier = model_type == "TabPFNClassifier"
    is_ranker = model_type == "LGBMRanker"
    ranker_config = ranker_config or {}
    ranker_grouping = ranker_config.get("grouping", {})
    group_strategy = ranker_grouping.get("strategy", "era")
    group_feature = ranker_grouping.get("feature")
    label_bins = int(ranker_config.get("label_bins", 5))

    for fold_idx, (train_eras, val_eras) in enumerate(splits):
        if not train_eras or not val_eras:
            continue
        train = numerai_cv.filter_eras(full, train_eras, era_col=era_col)
        validation = numerai_cv.filter_eras(full, val_eras, era_col=era_col)

        if train.empty or validation.empty:
            continue

        if max_train_samples and train.shape[0] > max_train_samples:
            train = train.sample(n=max_train_samples, random_state=sample_seed)
            print(
                f"Downsampled training fold {fold_idx} to {max_train_samples} rows for {model_type}."
            )

        params = model_params
        if is_ranker:
            params = ensure_label_gain(model_params, label_bins)
        model = build_model(model_type, params)
        if is_ranker:
            train_key = build_group_key(train, group_strategy, era_col, group_feature)
            train_sorted, X_train, y_train, train_groups = prepare_ranker_data(
                train, features, target_col, train_key
            )
            y_train = build_ranker_labels(train_sorted[target_col], label_bins)
            sample_weight = None
            if era_weighting_enabled:
                sample_weight = build_era_weights(
                    train_sorted[era_col],
                    era_weighting_scheme,
                    era_weighting_min,
                    era_weighting_max,
                )

            fit_kwargs = {"group": train_groups}
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight
            try:
                model.fit(X_train, y_train, **fit_kwargs)
            except TypeError:
                print(
                    "Model does not support sample_weight; training without era weighting."
                )
                fit_kwargs.pop("sample_weight", None)
                model.fit(X_train, y_train, **fit_kwargs)

            val_key = build_group_key(validation, group_strategy, era_col, group_feature)
            val_sorted, X_val, _, _ = prepare_ranker_data(
                validation, features, target_col, val_key
            )
            preds = model.predict(X_val)
            pred_series = pd.Series(np.asarray(preds).ravel(), index=val_sorted.index)
            fold_predictions = validation[[id_col, era_col, target_col]].copy()
            fold_predictions["prediction"] = pred_series.loc[validation.index].to_numpy()
        else:
            sample_weight = None
            if era_weighting_enabled:
                sample_weight = build_era_weights(
                    train[era_col],
                    era_weighting_scheme,
                    era_weighting_min,
                    era_weighting_max,
                )

            X_train = train[features]
            y_train = train[target_col]
            X_val = validation[features]
            if is_tabpfn:
                X_train = X_train.to_numpy()
                y_train = y_train.to_numpy()
                X_val = X_val.to_numpy()

            if sample_weight is None:
                model.fit(X_train, y_train)
            else:
                try:
                    model.fit(X_train, y_train, sample_weight=sample_weight)
                except TypeError:
                    print(
                        "Model does not support sample_weight; training without era weighting."
                    )
                    model.fit(X_train, y_train)

            fold_predictions = validation[[id_col, era_col, target_col]].copy()
            if is_tabpfn_classifier:
                preds = model.predict_proba(X_val)[:, 1]
            else:
                preds = model.predict(X_val)
            fold_predictions["prediction"] = np.asarray(preds).ravel()
        fold_predictions["cv_fold"] = fold_idx
        predictions.append(fold_predictions)
        fold_info.append(
            {
                "fold": fold_idx,
                "train_eras": len(train_eras),
                "val_eras": len(val_eras),
                "train_rows": int(train.shape[0]),
                "val_rows": int(validation.shape[0]),
            }
        )

    if not predictions:
        raise ValueError("No CV folds produced predictions; check CV settings.")

    oof = pd.concat(predictions, ignore_index=True)
    if oof[id_col].duplicated().any():
        raise ValueError("OOF predictions contain duplicate ids.")

    cv_meta = {
        "n_splits": cv_n_splits,
        "embargo": cv_embargo,
        "mode": cv_mode,
        "min_train_size": cv_min_train_size,
        "folds_used": len(fold_info),
        "folds": fold_info,
    }
    return oof, cv_meta


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


def resolve_small_bmc_baseline_path(
    baseline_name: str,
    predictions_dir: Path,
    baselines_dir: Path,
    override_path: str | None = None,
) -> Path:
    if override_path:
        return Path(override_path).expanduser().resolve()
    candidate_paths = [
        baselines_dir / "predictions" / f"{baseline_name}.parquet",
        predictions_dir / f"{baseline_name}.parquet",
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    return candidate_paths[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the base Numerai model."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the config file (.py or .json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base output directory for results and predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config
    config = load_config(config_path)

    data_config = config.get("data", {})
    preprocessing_config = config.get("preprocessing", {})
    training_config = config.get("training", {})
    model_config = config.get("model", {})
    output_config = config.get("output", {})

    data_version = data_config.get("data_version", "v5.2")
    feature_set = data_config.get("feature_set", "small")
    target_col = data_config.get("target_col", "target")
    era_col = data_config.get("era_col", "era")
    id_col = data_config.get("id_col", "id")
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
    grouping = ranker_config.get("grouping", {}) if model_config.get("type") == "LGBMRanker" else {}
    group_feature = grouping.get("feature")

    output_dir = args.output_dir or output_config.get("output_dir")
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

    napi = NumerAPI()

    features = load_features(napi, data_version, feature_set)
    extra_cols = [group_feature] if group_feature else []
    full = load_full_data(
        napi, data_version, features, era_col, target_col, id_col, extra_cols=extra_cols
    )

    if nan_missing_all_twos:
        full = apply_missing_all_twos_as_nan(
            full, features, era_col, missing_value
        )

    model_type = model_config.get("type", "LGBMRegressor")
    model_params = model_config.get("params")
    if model_params is None:
        if model_type == "LGBMRegressor":
            model_params = DEFAULT_LGBM_PARAMS
        else:
            model_params = {}

    cv_config = dict(training_config.get("cv", {}))
    cv_config.setdefault("embargo", embargo_eras)
    if not cv_config.get("enabled", True):
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
    )

    prediction_cols = [col for col in [id_col, era_col, target_col, "prediction", "cv_fold"] if col]
    prediction_cols = [col for col in prediction_cols if col in predictions.columns]
    predictions = predictions[prediction_cols].copy()
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = resolve_predictions_path(
        config, config_path, predictions_dir
    )
    predictions.to_parquet(predictions_path, index=False)
    print(f"Saved predictions to {predictions_path}")
    predictions_relative = predictions_path.relative_to(output_dir)
    small_bmc_baseline_override = (
        data_config.get("small_bmc_baseline_path")
        or output_config.get("small_bmc_baseline_path")
    )
    small_bmc_baseline_path = resolve_small_bmc_baseline_path(
        small_bmc_baseline,
        predictions_dir,
        baselines_dir,
        override_path=small_bmc_baseline_override,
    )

    summaries = numerai_metrics.summarize_prediction_file_with_bmc(
        predictions_path,
        ["prediction"],
        target_col,
        data_version,
        benchmark_model=benchmark_model,
        small_bmc_baseline_path=small_bmc_baseline_path,
        small_bmc_baseline_name=small_bmc_baseline,
        era_col=era_col,
        id_col=id_col,
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = resolve_results_path(config, config_path, results_dir)
    try:
        small_bmc_baseline_rel = str(
            small_bmc_baseline_path.relative_to(output_dir)
        )
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
            "full_rows": int(full.shape[0]),
            "full_eras": int(full[era_col].nunique()),
            "oof_rows": int(predictions.shape[0]),
            "oof_eras": int(predictions[era_col].nunique()),
            "embargo_eras": embargo_eras,
        },
        "benchmark": {
            "model": benchmark_model,
            "file": f"{data_version}/full_benchmark_models.parquet",
        },
        "small_bmc_benchmark": {
            "model": small_bmc_baseline,
            "predictions_file": str(
                small_bmc_baseline_rel
            ),
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
            "ranker": {
                "grouping": (
                    ranker_config.get("grouping") if model_type == "LGBMRanker" else None
                ),
                "label_bins": (
                    ranker_config.get("label_bins") if model_type == "LGBMRanker" else None
                ),
            },
            "data_sampling": {
                "max_train_samples": max_train_samples,
                "sample_seed": sample_seed if max_train_samples else None,
            },
            "cv": {
                "enabled": cv_config.get("enabled", True),
                "n_splits": cv_meta["n_splits"],
                "embargo": cv_meta["embargo"],
                "mode": cv_meta["mode"],
                "min_train_size": cv_meta["min_train_size"],
            },
        },
    }
    if "small_bmc_last200" in summaries:
        results["metrics"]["small_bmc_last200"] = (
            summaries["small_bmc_last200"].loc["prediction"].to_dict()
        )

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
