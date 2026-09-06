from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


def rank_gauss(series: pd.Series) -> pd.Series:
    """Per-vector rank-gauss with unit standard deviation, keeping NaNs."""
    ranked = series.rank(method="average", na_option="keep")
    n = int(ranked.count())
    if n == 0:
        return pd.Series(np.nan, index=series.index, name=series.name)
    if n == 1:
        out = pd.Series(0.0, index=series.index, name=series.name)
        out[ranked.isna()] = np.nan
        return out
    unit = (ranked - 0.5) / n
    gauss = pd.Series(stats.norm.ppf(unit), index=series.index, name=series.name)
    std = float(gauss.std(ddof=0))
    if not np.isfinite(std) or std == 0.0:
        return gauss
    return gauss / std


def rank_gauss_per_era(
    df: pd.DataFrame,
    pred_cols: Sequence[str],
    era_col: str = "era",
) -> pd.DataFrame:
    ranked = df.copy()
    ranked[list(pred_cols)] = (
        df.groupby(era_col, group_keys=False)[list(pred_cols)]
        .transform(lambda col: rank_gauss(col))
    )
    return ranked


def blend_rank_gauss(
    df: pd.DataFrame,
    pred_cols: Sequence[str],
    weights: Sequence[float],
    era_col: str = "era",
    out_col: str = "prediction",
) -> pd.Series:
    if len(pred_cols) != len(weights):
        raise ValueError("pred_cols and weights must have the same length.")
    weight = np.asarray(weights, dtype="float64")
    if not np.isfinite(weight).all():
        raise ValueError("weights must be finite.")
    if np.isclose(weight.sum(), 0.0):
        raise ValueError("weights must sum to a non-zero value.")
    weight = weight / weight.sum()

    ranked = rank_gauss_per_era(df, pred_cols, era_col=era_col)
    blended = ranked[list(pred_cols)].to_numpy(dtype="float64") @ weight
    blended = pd.Series(blended, index=df.index, name=out_col)
    return (
        df.assign(_blend=blended)
        .groupby(era_col, group_keys=False)["_blend"]
        .transform(lambda col: rank_gauss(col))
        .rename(out_col)
    )
