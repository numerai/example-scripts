from __future__ import annotations

import numpy as np
import pandas as pd


def get_embargo_eras(train_eras: list[str], embargo: int) -> list[str]:
    last_train_era = int(train_eras[-1])
    return [
        str(era).zfill(4)
        for era in range(last_train_era + 1, last_train_era + embargo + 1)
    ]


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
        weights = min_weight + (max_weight - min_weight) * (
            ranks / (len(unique_eras) - 1)
        )
    elif scheme == "exponential":
        if min_weight <= 0 or max_weight <= 0:
            raise ValueError("min_weight and max_weight must be > 0 for exponential scheme.")
        scale = np.log(max_weight / min_weight)
        weights = min_weight * np.exp(scale * (ranks / (len(unique_eras) - 1)))
    else:
        raise ValueError(f"Unknown era weighting scheme: {scheme}")
    return weights.to_numpy()
