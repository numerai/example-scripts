"""Era-based CV helpers and OOF prediction builder."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from agents.utils.modeling.models.ranker import (
    build_group_key,
    build_ranker_labels,
    ensure_label_gain,
    prepare_ranker_data,
)
from agents.utils.modeling.utils.era import build_era_weights
from agents.utils.modeling.utils.model_factory import build_model


def _era_sort_key(era):
    try:
        return int(era)
    except (TypeError, ValueError):
        return str(era)


def _sorted_unique_eras(eras: Iterable) -> List:
    return sorted(set(eras), key=_era_sort_key)


def era_cv_splits(
    eras: Sequence,
    n_splits: int = 5,
    embargo: int = 13,
    mode: str = "expanding",
    min_train_size: int = 1,
) -> List[Tuple[List, List]]:
    """Create sequential era splits with optional embargo.

    mode="expanding": train uses eras before validation; embargo removes eras
    immediately before the validation window.

    mode="blocked": train uses all eras except the validation window plus an
    embargo buffer before and after.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if embargo < 0:
        raise ValueError("embargo must be >= 0")

    eras_sorted = _sorted_unique_eras(eras)
    if n_splits > len(eras_sorted):
        raise ValueError("n_splits must be <= number of eras")

    fold_size = len(eras_sorted) // n_splits
    remainder = len(eras_sorted) % n_splits
    splits: List[Tuple[List, List]] = []

    for i in range(n_splits):
        start = i * fold_size
        end = (i + 1) * fold_size
        if i == n_splits - 1:
            end += remainder
        val_eras = eras_sorted[start:end]
        if not val_eras:
            continue

        if mode == "expanding":
            train_end = max(0, start - embargo)
            train_eras = eras_sorted[:train_end]
        elif mode == "blocked":
            left = max(0, start - embargo)
            right = min(len(eras_sorted), end + embargo)
            train_eras = eras_sorted[:left] + eras_sorted[right:]
        else:
            raise ValueError("mode must be 'expanding' or 'blocked'")

        if len(train_eras) < min_train_size:
            raise ValueError(
                "train split too small; reduce n_splits/embargo or set min_train_size"
            )

        splits.append((train_eras, val_eras))

    return splits


def filter_eras(df, eras: Sequence, era_col: str = "era"):
    """Filter a DataFrame to the given era values."""
    return df[df[era_col].isin(eras)]


def split_by_era(df, train_eras: Sequence, val_eras: Sequence, era_col: str = "era"):
    """Split a DataFrame into train/validation using era lists."""
    train_df = filter_eras(df, train_eras, era_col=era_col)
    val_df = filter_eras(df, val_eras, era_col=era_col)
    return train_df, val_df


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
    target_transform: dict | None,
    prediction_transform: dict | None,
    benchmark_col: str | None,
    baseline_col: str | None,
    prediction_batch_size: int | None,
) -> tuple[pd.DataFrame, dict]:
    cv_n_splits = int(cv_config.get("n_splits", 5))
    cv_embargo = int(cv_config.get("embargo", 13))
    cv_mode = cv_config.get("mode", "expanding")
    cv_min_train_size = int(cv_config.get("min_train_size", 0))

    splits = era_cv_splits(
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

    target_transform = target_transform or {}
    prediction_transform = prediction_transform or {}
    target_transform_type = target_transform.get("type")
    target_transform_scale = target_transform.get("scale")
    prediction_transform_type = prediction_transform.get("type")
    uses_benchmark = (
        target_transform_type == "benchmark_residual"
        or prediction_transform_type == "add_benchmark"
    )
    uses_baseline = (
        target_transform_type == "baseline_residual"
        or prediction_transform_type == "add_baseline"
    )
    if uses_benchmark and not benchmark_col:
        raise ValueError("benchmark_col is required for benchmark-based transforms.")
    if uses_baseline and not baseline_col:
        raise ValueError("baseline_col is required for baseline-based transforms.")

    if target_transform_scale is None:
        scale_unit_variance = False
        scale_era_zscore = False
    else:
        scale_key = str(target_transform_scale).lower()
        scale_unit_variance = scale_key == "unit_variance"
        scale_era_zscore = scale_key == "era_zscore"
        if not (scale_unit_variance or scale_era_zscore):
            raise ValueError(
                "Unsupported target_transform scale. Use 'unit_variance' or 'era_zscore'."
            )

    def _safe_std(values):
        std = float(np.nanstd(values))
        return std if std > 0 else 1.0

    def _era_zscore(values, eras):
        means = values.groupby(eras).transform("mean")
        stds = values.groupby(eras).transform("std")
        stds = stds.fillna(0.0)
        stds = stds.mask(stds <= 0, 1.0)
        return (values - means) / stds

    def _batched_predict(model, X, batch_size: int | None, classifier: bool = False):
        if not batch_size or batch_size <= 0:
            if classifier:
                return model.predict_proba(X)[:, 1]
            return model.predict(X)
        total = len(X)
        preds = []
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            if hasattr(X, "iloc"):
                X_batch = X.iloc[start:end]
            else:
                X_batch = X[start:end]
            if classifier:
                batch_preds = model.predict_proba(X_batch)[:, 1]
            else:
                batch_preds = model.predict(X_batch)
            preds.append(np.asarray(batch_preds).ravel())
        if not preds:
            return np.array([], dtype="float32")
        return np.concatenate(preds)

    for fold_idx, (train_eras, val_eras) in enumerate(splits):
        if not train_eras or not val_eras:
            continue
        train = filter_eras(full, train_eras, era_col=era_col)
        validation = filter_eras(full, val_eras, era_col=era_col)

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
            target_scale = None
            ref_scale = None
            train_key = build_group_key(train, group_strategy, era_col, group_feature)
            train_sorted, X_train, y_train, train_groups = prepare_ranker_data(
                train, features, target_col, train_key
            )
            if target_transform_type == "benchmark_residual":
                if scale_unit_variance:
                    target_scale = _safe_std(train_sorted[target_col].to_numpy())
                    ref_scale = _safe_std(train_sorted[benchmark_col].to_numpy())
                    residual = (
                        train_sorted[target_col] / target_scale
                        - train_sorted[benchmark_col] / ref_scale
                    )
                elif scale_era_zscore:
                    residual = _era_zscore(
                        train_sorted[target_col], train_sorted[era_col]
                    ) - _era_zscore(
                        train_sorted[benchmark_col], train_sorted[era_col]
                    )
                else:
                    residual = train_sorted[target_col] - train_sorted[benchmark_col]
                y_train = build_ranker_labels(residual, label_bins)
            elif target_transform_type == "baseline_residual":
                if scale_unit_variance:
                    target_scale = _safe_std(train_sorted[target_col].to_numpy())
                    ref_scale = _safe_std(train_sorted[baseline_col].to_numpy())
                    residual = (
                        train_sorted[target_col] / target_scale
                        - train_sorted[baseline_col] / ref_scale
                    )
                elif scale_era_zscore:
                    residual = _era_zscore(
                        train_sorted[target_col], train_sorted[era_col]
                    ) - _era_zscore(
                        train_sorted[baseline_col], train_sorted[era_col]
                    )
                else:
                    residual = train_sorted[target_col] - train_sorted[baseline_col]
                y_train = build_ranker_labels(residual, label_bins)
            else:
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
            if prediction_transform_type == "add_benchmark":
                if scale_unit_variance and target_transform_type == "benchmark_residual":
                    scale = target_scale if target_scale is not None else 1.0
                    ref = ref_scale if ref_scale is not None else 1.0
                    pred_series = pred_series * scale + val_sorted[benchmark_col] * (
                        scale / ref
                    )
                elif scale_era_zscore and target_transform_type == "benchmark_residual":
                    pred_series = pred_series + _era_zscore(
                        val_sorted[benchmark_col], val_sorted[era_col]
                    )
                else:
                    pred_series = pred_series + val_sorted[benchmark_col]
            elif prediction_transform_type == "add_baseline":
                if scale_unit_variance and target_transform_type == "baseline_residual":
                    scale = target_scale if target_scale is not None else 1.0
                    ref = ref_scale if ref_scale is not None else 1.0
                    pred_series = pred_series * scale + val_sorted[baseline_col] * (
                        scale / ref
                    )
                elif scale_era_zscore and target_transform_type == "baseline_residual":
                    pred_series = pred_series + _era_zscore(
                        val_sorted[baseline_col], val_sorted[era_col]
                    )
                else:
                    pred_series = pred_series + val_sorted[baseline_col]
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
            if target_transform_type == "benchmark_residual":
                if scale_unit_variance:
                    target_scale = _safe_std(train[target_col].to_numpy())
                    ref_scale = _safe_std(train[benchmark_col].to_numpy())
                    y_train = train[target_col] / target_scale - train[
                        benchmark_col
                    ] / ref_scale
                elif scale_era_zscore:
                    y_train = _era_zscore(train[target_col], train[era_col]) - _era_zscore(
                        train[benchmark_col], train[era_col]
                    )
                else:
                    y_train = y_train - train[benchmark_col]
            elif target_transform_type == "baseline_residual":
                if scale_unit_variance:
                    target_scale = _safe_std(train[target_col].to_numpy())
                    ref_scale = _safe_std(train[baseline_col].to_numpy())
                    y_train = train[target_col] / target_scale - train[
                        baseline_col
                    ] / ref_scale
                elif scale_era_zscore:
                    y_train = _era_zscore(train[target_col], train[era_col]) - _era_zscore(
                        train[baseline_col], train[era_col]
                    )
                else:
                    y_train = y_train - train[baseline_col]
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
                preds = _batched_predict(
                    model,
                    X_val,
                    prediction_batch_size,
                    classifier=True,
                )
            else:
                preds = _batched_predict(
                    model,
                    X_val,
                    prediction_batch_size,
                    classifier=False,
                )
            if prediction_transform_type == "add_benchmark":
                preds = np.asarray(preds).ravel()
                if scale_unit_variance and target_transform_type == "benchmark_residual":
                    scale = target_scale if target_scale is not None else 1.0
                    ref = ref_scale if ref_scale is not None else 1.0
                    preds = preds * scale + validation[benchmark_col].to_numpy() * (
                        scale / ref
                    )
                elif scale_era_zscore and target_transform_type == "benchmark_residual":
                    preds = preds + _era_zscore(
                        validation[benchmark_col], validation[era_col]
                    ).to_numpy()
                else:
                    preds = preds + validation[benchmark_col].to_numpy()
            elif prediction_transform_type == "add_baseline":
                preds = np.asarray(preds).ravel()
                if scale_unit_variance and target_transform_type == "baseline_residual":
                    scale = target_scale if target_scale is not None else 1.0
                    ref = ref_scale if ref_scale is not None else 1.0
                    preds = preds * scale + validation[baseline_col].to_numpy() * (
                        scale / ref
                    )
                elif scale_era_zscore and target_transform_type == "baseline_residual":
                    preds = preds + _era_zscore(
                        validation[baseline_col], validation[era_col]
                    ).to_numpy()
                else:
                    preds = preds + validation[baseline_col].to_numpy()
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
