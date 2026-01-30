from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd


_KNOWN_X_GROUPS = {
    "features",
    "era",
    "benchmark_models",
    "baseline",
    "id",
}
_DEFAULT_X_GROUPS = ("features", "era", "benchmark_models")


@dataclass(frozen=True)
class ModelDataBatch:
    X: pd.DataFrame
    y: pd.Series
    era: pd.Series
    id: pd.Series | None


@dataclass(frozen=True)
class ModelDataLoader:
    full: pd.DataFrame
    era_col: str
    target_col: str
    id_col: str | None
    x_cols: tuple[str, ...]

    def load(self, eras: Sequence) -> ModelDataBatch:
        subset = self.full[self.full[self.era_col].isin(eras)]
        X = subset[list(self.x_cols)]
        y = subset[self.target_col]
        era = subset[self.era_col]
        ids = subset[self.id_col] if self.id_col else None
        return ModelDataBatch(X=X, y=y, era=era, id=ids)


def build_model_data_loader(
    *,
    full: pd.DataFrame,
    x_cols: Iterable[str],
    era_col: str,
    target_col: str,
    id_col: str | None,
) -> ModelDataLoader:
    x_cols = tuple(x_cols)
    if not x_cols:
        raise ValueError("x_cols must be a non-empty list.")
    return ModelDataLoader(
        full=full,
        era_col=era_col,
        target_col=target_col,
        id_col=id_col,
        x_cols=x_cols,
    )


def normalize_x_groups(x_groups: Iterable[str] | None) -> list[str]:
    if not x_groups:
        x_groups = list(_DEFAULT_X_GROUPS)
    normalized = []
    for key in x_groups:
        if key in {"target", "y"}:
            continue
        if key in {"benchmark", "benchmarks"}:
            key = "benchmark_models"
        if key not in _KNOWN_X_GROUPS:
            raise ValueError(
                f"Unknown x_group '{key}'. Supported keys: {sorted(_KNOWN_X_GROUPS)}"
            )
        if key not in normalized:
            normalized.append(key)
    for required in _DEFAULT_X_GROUPS:
        if required not in normalized:
            normalized.append(required)
    return normalized


def build_x_cols(
    *,
    x_groups: Sequence[str],
    features: Sequence[str],
    benchmark_cols: Sequence[str],
    era_col: str,
    id_col: str | None,
    baseline_col: str | None = None,
) -> list[str]:
    x_cols: list[str] = []
    for key in x_groups:
        if key == "features":
            x_cols.extend(features)
        elif key == "era":
            x_cols.append(era_col)
        elif key == "benchmark_models":
            if not benchmark_cols:
                raise ValueError("benchmark_models requested but no benchmark columns found.")
            x_cols.extend(benchmark_cols)
        elif key == "baseline":
            if not baseline_col:
                raise ValueError("baseline requested but no baseline column provided.")
            x_cols.append(baseline_col)
        elif key == "id":
            if not id_col:
                raise ValueError("id requested but id_col is not set.")
            x_cols.append(id_col)
        else:
            raise ValueError(f"Unknown x_group '{key}'.")
    seen = set()
    return [col for col in x_cols if not (col in seen or seen.add(col))]
