from __future__ import annotations

import numpy as np
import pandas as pd


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
