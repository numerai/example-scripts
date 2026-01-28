from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def apply_target_transform(
    y: pd.Series,
    X: pd.DataFrame,
    transform: dict[str, Any] | str | None,
) -> pd.Series:
    if transform is None or transform == {}:
        return y
    if isinstance(transform, str):
        transform = {"type": transform}
    if not isinstance(transform, dict):
        raise TypeError(
            "model.target_transform must be a dict, a string identifier, or None."
        )

    transform_type = transform.get("type")
    if transform_type is None:
        raise ValueError("model.target_transform.type is required.")

    if transform_type in {"residual_to_benchmark", "residualize_to_benchmark"}:
        benchmark_col = transform.get("benchmark_col")
        if not benchmark_col:
            raise ValueError(
                "model.target_transform.benchmark_col is required for residual_to_benchmark."
            )
        era_col = transform.get("era_col", "era")
        per_era = bool(transform.get("per_era", True))
        fit_intercept = bool(transform.get("fit_intercept", True))
        proportion = float(transform.get("proportion", 1.0))
        return residualize_to_column(
            y,
            X,
            benchmark_col=benchmark_col,
            era_col=era_col,
            per_era=per_era,
            fit_intercept=fit_intercept,
            proportion=proportion,
        )

    if transform_type in {"subtract_benchmark", "subtract_benchmark_zscore"}:
        benchmark_col = transform.get("benchmark_col")
        if not benchmark_col:
            raise ValueError(
                "model.target_transform.benchmark_col is required for subtract_benchmark."
            )
        era_col = transform.get("era_col", "era")
        scale = float(transform.get("scale", 0.07))
        return subtract_scaled_zscore_column(
            y,
            X,
            benchmark_col=benchmark_col,
            era_col=era_col,
            scale=scale,
        )

    raise ValueError(f"Unknown target_transform type: {transform_type}")


def residualize_to_column(
    y: pd.Series,
    X: pd.DataFrame,
    *,
    benchmark_col: str,
    era_col: str = "era",
    per_era: bool = True,
    fit_intercept: bool = True,
    proportion: float = 1.0,
) -> pd.Series:
    if benchmark_col not in X.columns:
        raise ValueError(
            f"Benchmark column '{benchmark_col}' not found in X. "
            "Ensure model.x_groups includes 'benchmark_models' (and the benchmark file contains that column)."
        )
    if per_era and era_col not in X.columns:
        raise ValueError(
            f"Era column '{era_col}' not found in X. Ensure model.x_groups includes 'era'."
        )
    if not (0.0 <= proportion <= 1.0):
        raise ValueError("proportion must be in [0, 1].")

    benchmark = X[benchmark_col]
    eras = X[era_col] if per_era else None
    residual = _linear_residual(y, benchmark, groups=eras, fit_intercept=fit_intercept)
    if proportion == 1.0:
        return residual
    return (1.0 - proportion) * y.astype("float64") + proportion * residual


def subtract_scaled_zscore_column(
    y: pd.Series,
    X: pd.DataFrame,
    *,
    benchmark_col: str,
    era_col: str = "era",
    scale: float = 0.07,
) -> pd.Series:
    if benchmark_col not in X.columns:
        raise ValueError(
            f"Benchmark column '{benchmark_col}' not found in X. "
            "Ensure model.x_groups includes 'benchmark_models' (and the benchmark file contains that column)."
        )
    if era_col not in X.columns:
        raise ValueError(
            f"Era column '{era_col}' not found in X. Ensure model.x_groups includes 'era'."
        )
    if not np.isfinite(scale):
        raise ValueError("scale must be finite.")

    benchmark = X[benchmark_col]
    eras = X[era_col]
    z_benchmark = _zscore(benchmark, groups=eras)

    y_values = pd.to_numeric(y, errors="coerce").to_numpy(dtype="float64", copy=False)
    z_values = pd.to_numeric(z_benchmark, errors="coerce").to_numpy(
        dtype="float64", copy=False
    )
    transformed = y_values - float(scale) * z_values
    return pd.Series(transformed, index=y.index, name=y.name)


def _zscore(
    x: pd.Series,
    *,
    groups: pd.Series | None,
) -> pd.Series:
    x_values = pd.to_numeric(x, errors="coerce").to_numpy(dtype="float64", copy=False)

    if groups is None:
        z = _zscore_global(x_values)
        return pd.Series(z, index=x.index, name=x.name)

    group_codes, _ = pd.factorize(groups, sort=False)
    z = _zscore_groupwise(x_values, group_codes)
    return pd.Series(z, index=x.index, name=x.name)


def _zscore_global(x: np.ndarray) -> np.ndarray:
    mask = np.isfinite(x)
    z = np.full_like(x, np.nan, dtype="float64")
    if not mask.any():
        return z

    xx = x[mask]
    mean = float(xx.mean())
    var = float(np.mean((xx - mean) ** 2))
    std = float(np.sqrt(var))
    if std == 0.0:
        z[mask] = 0.0
        return z
    z[mask] = (xx - mean) / std
    return z


def _zscore_groupwise(x: np.ndarray, group_codes: np.ndarray) -> np.ndarray:
    mask = (group_codes >= 0) & np.isfinite(x)
    z = np.full_like(x, np.nan, dtype="float64")
    if not mask.any():
        return z

    g = group_codes[mask]
    xx = x[mask]

    n_groups = int(g.max()) + 1
    counts = np.bincount(g, minlength=n_groups).astype("float64")
    sum_x = np.bincount(g, weights=xx, minlength=n_groups)
    sum_x2 = np.bincount(g, weights=xx * xx, minlength=n_groups)

    mean = np.divide(
        sum_x, counts, out=np.zeros_like(sum_x, dtype="float64"), where=counts != 0.0
    )
    mean_x2 = np.divide(
        sum_x2,
        counts,
        out=np.zeros_like(sum_x2, dtype="float64"),
        where=counts != 0.0,
    )
    var = mean_x2 - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))

    denom = std[g]
    diff = xx - mean[g]
    z_vals = np.divide(diff, denom, out=np.zeros_like(diff), where=denom != 0.0)
    z[mask] = z_vals
    return z


def _linear_residual(
    y: pd.Series,
    x: pd.Series,
    *,
    groups: pd.Series | None,
    fit_intercept: bool,
) -> pd.Series:
    y_values = pd.to_numeric(y, errors="coerce").to_numpy(dtype="float64", copy=False)
    x_values = pd.to_numeric(x, errors="coerce").to_numpy(dtype="float64", copy=False)

    if groups is None:
        resid = _linear_residual_global(y_values, x_values, fit_intercept=fit_intercept)
        return pd.Series(resid, index=y.index, name=y.name)

    group_codes, _ = pd.factorize(groups, sort=False)
    resid = _linear_residual_groupwise(
        y_values, x_values, group_codes, fit_intercept=fit_intercept
    )
    return pd.Series(resid, index=y.index, name=y.name)


def _linear_residual_global(
    y: np.ndarray,
    x: np.ndarray,
    *,
    fit_intercept: bool,
) -> np.ndarray:
    mask = np.isfinite(y) & np.isfinite(x)
    resid = np.full_like(y, np.nan, dtype="float64")
    if not mask.any():
        return resid

    yy = y[mask]
    xx = x[mask]

    if fit_intercept:
        yy = yy - yy.mean()
        xx = xx - xx.mean()

    denom = float(np.dot(xx, xx))
    alpha = float(np.dot(xx, yy) / denom) if denom != 0.0 else 0.0
    resid_vals = yy - alpha * xx
    resid[mask] = resid_vals
    return resid


def _linear_residual_groupwise(
    y: np.ndarray,
    x: np.ndarray,
    group_codes: np.ndarray,
    *,
    fit_intercept: bool,
) -> np.ndarray:
    mask = (group_codes >= 0) & np.isfinite(y) & np.isfinite(x)
    resid = np.full_like(y, np.nan, dtype="float64")
    if not mask.any():
        return resid

    g = group_codes[mask]
    yy = y[mask]
    xx = x[mask]

    n_groups = int(g.max()) + 1
    counts = np.bincount(g, minlength=n_groups).astype("float64")
    sum_y = np.bincount(g, weights=yy, minlength=n_groups)
    sum_x = np.bincount(g, weights=xx, minlength=n_groups)

    mean_y = np.divide(
        sum_y, counts, out=np.zeros_like(sum_y, dtype="float64"), where=counts != 0.0
    )
    mean_x = np.divide(
        sum_x, counts, out=np.zeros_like(sum_x, dtype="float64"), where=counts != 0.0
    )

    if fit_intercept:
        y_centered = yy - mean_y[g]
        x_centered = xx - mean_x[g]
    else:
        y_centered = yy
        x_centered = xx

    sum_xx = np.bincount(g, weights=x_centered * x_centered, minlength=n_groups)
    sum_xy = np.bincount(g, weights=x_centered * y_centered, minlength=n_groups)

    alpha = np.zeros(n_groups, dtype="float64")
    good = sum_xx != 0.0
    alpha[good] = sum_xy[good] / sum_xx[good]

    resid_vals = y_centered - alpha[g] * x_centered
    resid[mask] = resid_vals
    return resid


class TargetTransformWrapper:
    def __init__(self, model, target_transform: dict[str, Any] | str):
        self._model = model
        self._target_transform = target_transform

    def fit(self, X, y, **kwargs):
        if not hasattr(y, "index") or not hasattr(X, "columns"):
            raise TypeError(
                "TargetTransformWrapper requires pandas inputs (X DataFrame, y Series)."
            )
        y_transformed = apply_target_transform(y, X, self._target_transform)
        self._model.fit(X, y_transformed, **kwargs)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def __getattr__(self, name: str):
        return getattr(self._model, name)
