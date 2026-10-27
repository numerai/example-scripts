"""Build ticker-local time-series features from Unity's medium feature set.

    python time_series.py --tickers 'AAPL US' 'MSFT US'
    python time_series.py
"""

import argparse
import json
import re
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["numerai_ticker", "date"]
SUFFIXES = ["rank_52w", "mean_13w_minus_52w", "weeks_since_high_52w", "weeks_since_change"]


def ages(values: np.ndarray, days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Elapsed weeks since the most recent rolling high and observed value change."""
    high_age = np.full(len(values), np.nan, dtype="float32")
    change_age = high_age.copy()
    highs = deque()
    previous, changed = np.nan, None
    for i, value in enumerate(values):
        # The window is (today - 52 weeks, today], including today's observation.
        while highs and days[highs[0]] <= days[i] - 364:
            highs.popleft()
        if np.isnan(value):
            # A missing observation breaks knowledge of the unchanged run.
            previous, changed = np.nan, None
            continue
        while highs and values[highs[-1]] <= value:
            highs.pop()  # use the most recent occurrence when the high is tied
        highs.append(i)
        high_age[i] = (days[i] - days[highs[0]]) / 7
        if np.isnan(previous):
            changed = None
        elif value != previous:
            changed = days[i]
        if changed is not None:
            change_age[i] = (days[i] - changed) / 7
        previous = value
    return high_age, change_age


def build_features(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return four features per source feature, keyed by ticker and date.

    Input may contain multiple data splits. Sort and group before computing so no
    other ticker or future row enters a calculation. Targets are never used.
    """
    data = data[KEYS + features].copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    if data[KEYS].isna().any().any():
        raise ValueError("Ticker and date must not be missing")
    if data.duplicated(KEYS).any():
        raise ValueError("Expected one observation per ticker and date")
    nonnumeric = [f for f in features if not pd.api.types.is_numeric_dtype(data[f])]
    if nonnumeric:
        raise ValueError(f"Time-series transforms require numeric features: {nonnumeric}")
    if not features:
        raise ValueError("No numeric features selected")
    data = data.sort_values(KEYS).reset_index(drop=True)
    output = {key: data[key] for key in KEYS}
    groups = data.groupby("numerai_ticker", sort=False, observed=True).indices.values()
    dates = pd.DatetimeIndex(data["date"])
    days = dates.to_numpy(dtype="datetime64[D]").astype("int64")
    for feature in features:
        result = np.full((len(data), 4), np.nan, dtype="float32")
        for indices in groups:
            values = pd.Series(data[feature].iloc[indices].to_numpy(dtype="float64"), index=dates[indices])
            long = values.rolling("364D", min_periods=1)
            short = values.rolling("91D", min_periods=1)
            high, changed = ages(values.to_numpy(), days[indices])
            transformed = np.column_stack([
                long.rank(method="average", pct=True).to_numpy(),
                (short.mean() - long.mean()).to_numpy(),
                high,
                changed,
            ])
            transformed[values.isna().to_numpy()] = np.nan
            result[indices] = transformed
        for i, suffix in enumerate(SUFFIXES):
            output[f"{feature}_{suffix}"] = result[:, i]
        print(f"Built {feature}", flush=True)
    return pd.DataFrame(output)


def latest_version(datasets: list[str]) -> str:
    """Select the latest numbered Classic Unity release with all required files."""
    versions = {name.split("/")[0] for name in datasets if re.fullmatch(r"v\d+\.\d+/features.json", name)}
    versions = [v for v in versions if int(v[1:].split(".")[0]) >= 6 and all(
        f"{v}/{name}" in datasets for name in ["train.parquet", "validation.parquet", "live.parquet"]
    )]
    if not versions:
        raise ValueError("No Unity dataset found; specify --data-version explicitly")
    return max(versions, key=lambda v: tuple(map(int, v[1:].split("."))))


def main() -> None:
    from numerapi import NumerAPI

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-version", help="default: latest available Unity version")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--tickers", nargs="+", help="optional exact numerai_ticker values for a small run")
    parser.add_argument("--output", type=Path, default=Path("data/time_series_features.parquet"))
    args = parser.parse_args()
    api = NumerAPI()
    version = args.data_version or latest_version(api.list_datasets())
    directory = args.data_dir / version
    directory.mkdir(parents=True, exist_ok=True)

    def download(name: str, refresh: bool = False) -> Path:
        path = directory / name
        if refresh or not path.exists():
            api.download_dataset(f"{version}/{name}", dest_path=str(path))
        return path

    metadata = json.loads(download("features.json").read_text())
    # Country is categorical metadata, also excluded in the Unity model notebook.
    features = [f for f in metadata["feature_sets"]["medium"] if f != "feature_country"]
    filters = [("numerai_ticker", "in", args.tickers)] if args.tickers else None
    # Keep every historical date, including test rows and rows without targets.
    # Refresh live on each run; it changes from round to round.
    data = pd.concat([
        pd.read_parquet(download(f"{split}.parquet", refresh=split == "live"), columns=KEYS + features, filters=filters)
        for split in ["train", "validation", "live"]
    ], ignore_index=True)
    # Overlapping splits may repeat the same observation. Reject conflicting values.
    data = data.drop_duplicates()
    result = build_features(data, features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    print(f"Wrote {len(result)} rows and {4 * len(features)} features from {version} to {args.output}")


if __name__ == "__main__":
    main()
