"""Reproduce the Unity transform comparison with an explicit, recorded configuration.

    python reproduce.py --config reproduction_config.json --source-post /path/to/post.html

The post does not specify its model configuration or exact date cutoffs. The
provided config is an independently specified attempt using the Unity notebook's
model and 200,000-row sampling recipe; it is not the missing original configuration.
Large downloads, a disk-backed feature matrix, models and predictions stay in the
cache. Compact results include file hashes, coverage, per-date scores and a chart.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numba import njit
from numerapi import NumerAPI
from numerai_tools.scoring import neutral_contribution, neutral_correlation
from threadpoolctl import threadpool_limits

from time_series import KEYS, SUFFIXES, build_features, rank_and_bin


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@njit(cache=True)
def raw_transforms(values, days, ticker_starts):
    """Exact example semantics, accelerated for Unity's numeric 0-4 inputs.

    Five rolling value counts provide average-tie ranks and the current high.
    The latest occurrence of each value supplies the high's date. Inputs with
    values outside 0-4 must be rejected by the caller, never rounded or clipped.
    """
    result = np.full((len(values), 4), np.nan, dtype=np.float32)
    for group in range(len(ticker_starts) - 1):
        start, stop = ticker_starts[group], ticker_starts[group + 1]
        left_long = left_short = start
        frequencies = np.zeros(5, dtype=np.int64)
        seen = np.full(5, -1, dtype=np.int64)
        long_sum = short_sum = 0.0
        long_count = short_count = 0
        previous = np.nan
        changed = first_valid = -1
        for i in range(start, stop):
            value = values[i]
            if not np.isnan(value):
                level = int(value)
                frequencies[level] += 1
                seen[level] = i
                long_sum += value
                short_sum += value
                long_count += 1
                short_count += 1
                if first_valid == -1:
                    first_valid = days[i]
            while days[left_long] <= days[i] - 364:
                old = values[left_long]
                if not np.isnan(old):
                    frequencies[int(old)] -= 1
                    long_sum -= old
                    long_count -= 1
                left_long += 1
            while days[left_short] <= days[i] - 91:
                old = values[left_short]
                if not np.isnan(old):
                    short_sum -= old
                    short_count -= 1
                left_short += 1
            if np.isnan(value):
                previous = np.nan
                changed = -1
                continue
            if not np.isnan(previous) and previous != value:
                changed = days[i]
            if changed != -1:
                result[i, 3] = (days[i] - changed) / 7
            previous = value
            if days[i] - first_valid < 364 or long_count < 52:
                continue
            lower = 0
            for level in range(int(value)):
                lower += frequencies[level]
            result[i, 0] = (lower + (frequencies[int(value)] + 1) / 2) / long_count
            if short_count >= 13:
                result[i, 1] = short_sum / short_count - long_sum / long_count
            for level in range(4, -1, -1):
                if frequencies[level]:
                    result[i, 2] = (days[i] - days[seen[level]]) / 7
                    break
    return result


@njit(cache=True)
def bin_transforms(raw, date_order, date_starts):
    """Average-tie percentile ranks per date, ceil(rank*5)-1; 255 means NaN."""
    result = np.full(raw.shape, 255, dtype=np.uint8)
    for group in range(len(date_starts) - 1):
        rows = date_order[date_starts[group]:date_starts[group + 1]]
        for column in range(raw.shape[1]):
            observed = rows[~np.isnan(raw[rows, column])]
            ordered = observed[np.argsort(raw[observed, column])]
            left = 0
            while left < len(ordered):
                right = left + 1
                while right < len(ordered) and raw[ordered[right], column] == raw[ordered[left], column]:
                    right += 1
                percentile = ((left + 1 + right) / 2) / len(ordered)
                level = np.uint8(np.ceil(percentile * 5) - 1)
                for i in range(left, right):
                    result[ordered[i], column] = level
                left = right
    return result


def boundaries(sorted_values):
    return np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1, len(sorted_values)]


def check_optimized_path(frame, feature):
    """Compare the full accelerated path with the public example on these rows."""
    frame = frame[KEYS + [feature]].sort_values(KEYS).reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"])
    tickers = frame.numerai_ticker.to_numpy()
    days = frame.date.to_numpy(dtype="datetime64[D]").astype("int64")
    ticker_starts = boundaries(tickers)
    order = np.argsort(days, kind="stable")
    date_starts = boundaries(days[order])
    values = frame[feature].to_numpy(dtype="float64", na_value=np.nan)
    raw = raw_transforms(values, days, ticker_starts)
    reference = build_features(frame, [feature])
    np.testing.assert_allclose(raw, reference.iloc[:, 2:].to_numpy(), rtol=0, atol=0, equal_nan=True)
    expected = rank_and_bin(reference).iloc[:, 2:].to_numpy()
    expected = np.where(np.isnan(expected), 255, expected).astype("uint8")
    np.testing.assert_array_equal(bin_transforms(raw, order, date_starts), expected)


def validate_kernel():
    """Exercise ties, NaNs, daily short history, calendar gaps and ticker resets."""
    rng = np.random.default_rng(7)
    pieces = []
    for ticker, frequency in [("A", "7D"), ("B", "D"), ("C", "14D")]:
        values = rng.integers(0, 5, 160).astype(float)
        values[:55] = 2
        values[[12, 56, 90]] = np.nan
        pieces.append(pd.DataFrame({"numerai_ticker": ticker, "date": pd.date_range("2020-01-01", periods=160, freq=frequency), "feature_x": values}))
    check_optimized_path(pd.concat(pieces).sample(frac=1, random_state=7), "feature_x")


def read_columns(paths, columns):
    return pd.concat([pd.read_parquet(path, columns=columns) for path in paths], ignore_index=True)


def prepare(config, data_dir, cache_dir):
    paths = [data_dir / "train.parquet", data_dir / "validation.parquet"]
    metadata = json.loads((data_dir / "features.json").read_text())
    base = metadata["feature_sets"][config["base_feature_set"]]
    medium = [f for f in metadata["feature_sets"][config["transform_feature_set"]] if f != "feature_country"]
    state_path = cache_dir / "matrix_manifest.json"
    builder_hash = sha256(Path(__file__).with_name("time_series.py"))
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["builder_sha256"] != builder_hash or state["config"] != config:
            raise ValueError("Cached matrix belongs to a different builder/config; use a new cache directory")
        return pd.read_parquet(cache_dir / "row_metadata.parquet"), np.load(cache_dir / "features.npy", mmap_mode="r"), state
    meta = read_columns(paths, KEYS + [config["target"], "data_type"])
    meta["source_row"] = np.arange(len(meta))
    meta["split"] = np.repeat(["train", "validation"], [pq.ParquetFile(p).metadata.num_rows for p in paths])
    meta["date"] = pd.to_datetime(meta["date"])
    if meta[KEYS].isna().any().any() or meta.duplicated(KEYS).any():
        raise ValueError("Missing or duplicate ticker/date keys in historical data")
    order = meta.sort_values(KEYS).index.to_numpy()
    meta = meta.iloc[order].reset_index(drop=True)
    days = meta.date.to_numpy(dtype="datetime64[D]").astype("int64")
    ticker_starts = boundaries(meta.numerai_ticker.to_numpy())
    date_order = np.argsort(days, kind="stable")
    date_starts = boundaries(days[date_order])
    columns = base + [f"{feature}_{suffix}" for feature in medium for suffix in SUFFIXES]
    matrix = np.lib.format.open_memmap(cache_dir / "features.npy", mode="w+", dtype="uint8", shape=(len(meta), len(columns)), fortran_order=True)
    print(f"Preparing {len(meta):,} rows, {len(base)} base + {4 * len(medium)} transformed columns; {matrix.nbytes / 1e9:.1f} GB disk matrix", flush=True)
    countries = None
    continuous_columns = {}
    started = time.monotonic()
    sample_tickers = meta.numerai_ticker.drop_duplicates().iloc[[0, len(ticker_starts) // 2 - 1, -1]].tolist()
    sample_mask = meta.numerai_ticker.isin(sample_tickers).to_numpy()
    for offset in range(0, len(base), 32):
        batch = base[offset:offset + 32]
        frame = read_columns(paths, batch).iloc[order].reset_index(drop=True)
        for feature in batch:
            column = base.index(feature)
            if feature == "feature_country":
                categorical = pd.Categorical(frame[feature])
                countries = list(categorical.categories)
                codes = categorical.codes.astype("int16")
                if len(countries) >= 255:
                    raise ValueError("Country categories do not fit uint8 storage")
                matrix[:, column] = np.where(codes < 0, 255, codes).astype("uint8")
                continue
            values = frame[feature].to_numpy(dtype="float64", na_value=np.nan)
            valid = values[~np.isnan(values)]
            if not np.all((valid >= 0) & (valid <= 4) & (valid == np.floor(valid))):
                if feature in medium:
                    raise ValueError(f"{feature} is a medium feature outside the compiled kernel's integer 0-4 scale")
                # The all set also contains continuous factor features. Preserve
                # their original float32 values in separate column files; they
                # must not be rounded, clipped or binned to fit uint8 storage.
                filename = f"continuous_{column}.npy"
                np.save(cache_dir / filename, values.astype("float32"))
                continuous_columns[str(column)] = {"feature": feature, "file": filename}
                matrix[:, column] = 255
                continue
            matrix[:, column] = np.where(np.isnan(values), 255, values).astype("uint8")
            if feature in medium:
                if feature in medium[:3]:
                    sample = meta.loc[sample_mask, KEYS].copy()
                    sample[feature] = values[sample_mask]
                    check_optimized_path(sample, feature)
                raw = raw_transforms(values, days, ticker_starts)
                binned = bin_transforms(raw, date_order, date_starts)
                start = len(base) + medium.index(feature) * 4
                matrix[:, start:start + 4] = binned
        matrix.flush()
        print(f"Prepared {min(offset + 32, len(base))}/{len(base)} source features in {time.monotonic() - started:.0f}s", flush=True)
    meta.to_parquet(cache_dir / "row_metadata.parquet", index=False)
    state = {"config": config, "builder_sha256": builder_hash, "columns": columns, "base_columns": len(base), "transform_columns": 4 * len(medium), "rows": len(meta), "country_categories": countries, "continuous_columns": continuous_columns, "kernel_validation": "bit-exact float32 raw and uint8 bins on synthetic edge cases and three actual feature histories for three sampled tickers", "preparation_seconds": time.monotonic() - started}
    state_path.write_text(json.dumps(state, indent=2))
    return meta, matrix, state


def feature_frame(matrix, rows, columns, names, continuous=None):
    # Gather one contiguous column at a time from the Fortran-order matrix.
    # Row-major advanced indexing has a much larger memory-access cost here.
    values = np.empty((len(rows), columns), dtype="float32", order="F")
    for column in range(columns):
        values[:, column] = matrix[rows, column]
    values[values == 255] = np.nan
    for column, source in (continuous or {}).items():
        if column < columns:
            values[:, column] = source[rows]
    frame = pd.DataFrame(values, columns=names[:columns])
    if "feature_country" in frame:
        frame["feature_country"] = frame.feature_country.fillna(-1).astype("int16").astype("category")
    return frame


def train_and_predict(config, meta, matrix, state, cache_dir):
    continuous = {int(column): np.load(cache_dir / item["file"], mmap_mode="r") for column, item in state["continuous_columns"].items()}
    predictions = meta[KEYS + [config["target"], "data_type", "split"]].copy()
    coverage = []
    for period in ["original", "retrained"]:
        if period == "original":
            train_mask = meta.split.eq("train") & meta[config["target"]].notna()
            requested_start = config["validation_start"]
        else:
            train_mask = meta.date.le(config["retrain_through"]) & meta[config["target"]].notna()
            requested_start = config["retrain_validation_start"]
        # Match the notebook's pandas sample recipe on original Parquet order,
        # although the feature matrix itself is sorted by ticker and date.
        available = meta.loc[train_mask].sort_values("source_row").index
        sampled = pd.Series(available).sample(n=min(config["training_sample_rows"], len(available)), random_state=config["sample_seed"]).to_numpy()
        last_train = meta.loc[available, "date"].max()
        embargo_end = last_train + pd.offsets.BDay(config["embargo_business_days"])
        evaluation = meta.split.eq("validation") & meta.data_type.eq("validation") & meta[config["target"]].notna() & meta.date.ge(requested_start) & meta.date.le(config["validation_end"]) & meta.date.gt(embargo_end)
        coverage.append({"period": period, "available_train_rows": len(available), "sampled_train_rows": len(sampled), "sampled_row_indices_sha256": hashlib.sha256(sampled.tobytes()).hexdigest(), "first_train_date": str(meta.loc[available, "date"].min().date()), "last_train_date": str(last_train.date()), "embargo_end": str(embargo_end.date()), "first_evaluation_date": str(meta.loc[evaluation, "date"].min().date()), "last_evaluation_date": str(meta.loc[evaluation, "date"].max().date()), "evaluation_rows_before_benchmark_join": int(evaluation.sum()), "evaluation_dates_before_benchmark_join": int(meta.loc[evaluation, "date"].nunique())})
        np.save(cache_dir / f"{period}_sampled_rows.npy", sampled)
        for variant, count in [("unity", state["base_columns"]), ("unity_plus_transforms", len(state["columns"]))]:
            label = f"{period}_{variant}"
            path = cache_dir / f"{label}_predictions.parquet"
            if path.exists():
                saved = pd.read_parquet(path)
                predictions[label] = saved[label]
                continue
            print(f"Training {label}: {len(sampled):,} rows, {count} features", flush=True)
            started = time.monotonic()
            model = lgb.LGBMRegressor(**config["model"])
            model.fit(feature_frame(matrix, sampled, count, state["columns"], continuous), meta.loc[sampled, config["target"]].to_numpy())
            model.booster_.save_model(str(cache_dir / f"{label}_model.txt"))
            predictions[label] = np.nan
            for date, rows in meta.loc[evaluation].groupby("date").groups.items():
                predictions.loc[rows, label] = model.predict(feature_frame(matrix, rows.to_numpy(), count, state["columns"], continuous))
            predictions[KEYS + [label]].to_parquet(path, index=False)
            print(f"Finished {label} in {time.monotonic() - started:.0f}s", flush=True)
    return predictions, coverage


def score(config, predictions, data_dir, result_dir):
    benchmark = pd.read_parquet(data_dir / "validation_benchmark_models.parquet", columns=KEYS + [config["benchmark"]])
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    predictions = predictions.merge(benchmark, on=KEYS, how="left", validate="one_to_one")
    labels = [f"{p}_{v}" for p in ["original", "retrained"] for v in ["unity", "unity_plus_transforms"]]
    neutralizer_file = pq.ParquetFile(data_dir / "validation_neutralizer.parquet")
    neutralizer_columns = [c for c in neutralizer_file.schema.names if c.startswith("neutralizer_column_")]
    per_date = []
    # Neutralizer row groups are stored by date; streaming avoids rescanning a
    # multi-GB file for each date and keeps only one row group in memory.
    for batch in neutralizer_file.iter_batches(batch_size=50000, columns=KEYS + neutralizer_columns):
        neutralizers = batch.to_pandas()
        neutralizers["date"] = pd.to_datetime(neutralizers["date"])
        # A row group may be split into multiple batches. Buffer by date below.
        if "pending" in locals():
            neutralizers = pd.concat([pending, neutralizers], ignore_index=True)
        last = neutralizers.date.max()
        pending = neutralizers[neutralizers.date.eq(last)]
        complete = neutralizers[neutralizers.date.ne(last)]
        for date, group in complete.groupby("date"):
            per_date.extend(score_date(date, group, predictions, config, labels, neutralizer_columns))
    if "pending" in locals():
        for date, group in pending.groupby("date"):
            per_date.extend(score_date(date, group, predictions, config, labels, neutralizer_columns))
    scores = pd.DataFrame(per_date).sort_values(["period", "variant", "date"])
    scores.to_csv(result_dir / "per_date_scores.csv", index=False)
    return scores


def score_date(date, neutralizers, predictions, config, labels, columns):
    selected = predictions[predictions.date.eq(date)]
    rows = []
    for period in ["original", "retrained"]:
        period_labels = [label for label in labels if label.startswith(period + "_")]
        valid = selected.dropna(subset=period_labels + [config["benchmark"], config["target"]])
        if valid.empty:
            continue
        joined = valid.merge(neutralizers[KEYS + columns], on=KEYS, how="inner", validate="one_to_one").set_index("numerai_ticker").sort_index()
        if joined[columns].isna().any().any() or len(joined) != len(valid):
            raise ValueError(f"Incomplete neutralizer coverage on {date}")
        ncorr = neutral_correlation(joined[period_labels], joined[columns], joined[config["target"]])
        bmc = neutral_contribution(joined[period_labels], joined[config["benchmark"]], joined[columns], joined[config["target"]])
        key_hash = hashlib.sha256("\n".join(joined.index.astype(str)).encode()).hexdigest()
        for label in period_labels:
            rows.append({"period": period, "variant": label[len(period) + 1:], "date": str(date.date()), "rows": len(joined), "ticker_keys_sha256": key_hash, "ncorr": float(ncorr[label]), "neutral_bmc": float(bmc[label]), "prediction_benchmark_correlation": float(joined[label].corr(joined[config["benchmark"]]))})
        print(f"Scored {period} {date.date()} ({len(joined):,} tickers)", flush=True)
    return rows


def record_run(config, config_path, manifest, source_post, cache_dir):
    """Refuse mixed caches and snapshot precisely the inputs used for this run."""
    packages = {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "pyarrow", "numerapi", "numerai-tools", "lightgbm", "scipy", "numba", "matplotlib", "threadpoolctl"]}
    sources = {"runner": sha256(__file__), "builder": sha256(Path(__file__).with_name("time_series.py")), "config": sha256(config_path)}
    signature = {"config": config, "data_manifest": manifest, "source_hashes": sources, "python": platform.python_version(), "platform": platform.platform(), "packages": packages, "source_post_sha256": sha256(source_post) if source_post else None}
    path = cache_dir / "run_signature.json"
    if path.exists() and json.loads(path.read_text()) != signature:
        raise ValueError("Cache signature differs (data, config, code or environment); use a new --cache-dir")
    if not path.exists():
        if (cache_dir / "matrix_manifest.json").exists():
            raise ValueError("Unsigned cached matrix; use a new --cache-dir")
        path.write_text(json.dumps(signature, indent=2))
        shutil.copyfile(__file__, cache_dir / "reproduce_snapshot.py")
        shutil.copyfile(Path(__file__).with_name("time_series.py"), cache_dir / "time_series_snapshot.py")
        shutil.copyfile(config_path, cache_dir / "config_snapshot.json")
    return signature


def summarize(scores, config):
    rows = []
    for (period, variant), group in scores.groupby(["period", "variant"]):
        row = {"period": period, "variant": variant, "dates": len(group), "first_date": group.date.min(), "last_date": group.date.max(), "ncorr": float(group.ncorr.mean()), "neutral_bmc": float(group.neutral_bmc.mean()), "neutral_bmc_last_200_dates": float(group.neutral_bmc.tail(200).mean()), "prediction_benchmark_correlation": float(group.prediction_benchmark_correlation.mean())}
        claimed = config["claimed_metrics"][period][variant]
        row["claimed"] = claimed
        row["matches_claimed_to_4_decimals"] = all(round(row[k], 4) == v for k, v in claimed.items())
        row["difference_from_claimed"] = {k: row[k] - v for k, v in claimed.items()}
        rows.append(row)
    return rows


def plot(scores, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex="col")
    for column, period in enumerate(["original", "retrained"]):
        for variant, group in scores[scores.period.eq(period)].groupby("variant"):
            for row, metric in enumerate(["ncorr", "neutral_bmc"]):
                axes[row, column].plot(pd.to_datetime(group.date), group[metric].cumsum(), label=variant.replace("_", " "))
                axes[row, column].set_title(f"{period}: cumulative {metric}")
                axes[row, column].grid(alpha=0.2)
        axes[0, column].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("reproduction_config.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).parent / "data/reproduction")
    parser.add_argument("--data-dir", type=Path, help="reuse existing downloads, e.g. data/v6.0")
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "reproduction_results")
    parser.add_argument("--source-post", type=Path, help="optional original HTML; its SHA256 is recorded")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir or args.cache_dir / config["data_version"]
    data_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    api = NumerAPI(show_progress_bars=False)
    manifest = []
    for name in ["features.json", "train.parquet", "validation.parquet", "validation_benchmark_models.parquet", "validation_neutralizer.parquet"]:
        path = data_dir / name
        if not path.exists():
            api.download_dataset(f"{config['data_version']}/{name}", dest_path=str(path), round_num=config["data_round"])
        item = {"name": name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        if name.endswith(".parquet"):
            file = pq.ParquetFile(path)
            item.update(rows=file.metadata.num_rows, columns=len(file.schema.names), row_groups=file.num_row_groups)
        manifest.append(item)
    signature = record_run(config, args.config, manifest, args.source_post, args.cache_dir)
    validate_kernel()
    meta, matrix, state = prepare(config, data_dir, args.cache_dir)
    predictions, coverage = train_and_predict(config, meta, matrix, state, args.cache_dir)
    with threadpool_limits(limits=config["blas_threads"], user_api="blas"):
        scores = score(config, predictions, data_dir, args.results_dir)
    result = {"purpose": "Independent reproduction attempt; original model config was not supplied", "config": config, "python": signature["python"], "platform": signature["platform"], "packages": signature["packages"], "builder_sha256": signature["source_hashes"]["builder"], "runner_sha256": signature["source_hashes"]["runner"], "config_sha256": signature["source_hashes"]["config"], "source_post_sha256": signature["source_post_sha256"], "data_manifest": manifest, "matrix": {k: state[k] for k in ["rows", "base_columns", "transform_columns", "continuous_columns", "preparation_seconds", "kernel_validation"]}, "training_and_evaluation_coverage": coverage, "metrics": summarize(scores, config), "total_seconds": time.monotonic() - started}
    (args.results_dir / "summary.json").write_text(json.dumps(result, indent=2))
    (args.results_dir / "config.json").write_text(json.dumps(config, indent=2))
    plot(scores, args.results_dir / "cumulative_scores.svg")
    print(json.dumps(result["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
