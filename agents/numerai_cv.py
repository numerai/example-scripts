"""Helpers for era-based cross-validation with embargo."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


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
