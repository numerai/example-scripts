"""Confirm the Unity transform comparison using every eligible training row.

    python reproduce_full.py --prepared-cache data/reproduction

Uses the same fixed 300-tree model as the sampled attempt. A LightGBM Sequence
streams the verified feature matrix instead of allocating a dense training
DataFrame. With no prepared cache, the features are built from the pinned input
files using the shared, validated preparation code.
"""

import argparse
import gc
import importlib.metadata
import json
import numbers
import platform
import resource
import shutil
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numerapi import NumerAPI
from threadpoolctl import threadpool_limits

import reproduce as shared
from plot_reproduction import plot_scores


def peak_gib():
    scale = 1024 ** 3 if sys.platform == "darwin" else 1024 ** 2
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def numeric_block(matrix, rows, count, continuous, dtype="float32"):
    values = np.empty((len(rows), count), dtype=dtype, order="F")
    for column in range(count):
        values[:, column] = matrix[rows, column]
    values[values == 255] = np.nan
    for column, source in continuous.items():
        if column < count:
            values[:, column] = source[rows]
    return values


class MatrixSequence(lgb.Sequence):
    """Block-cached scalar sampling plus bounded-memory batch construction."""

    batch_size = 4096

    def __init__(self, matrix, rows, count, continuous):
        self.matrix = matrix
        self.rows = np.asarray(rows)
        self.count = count
        self.continuous = continuous
        self.block_start = -1
        self.block = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        if isinstance(item, numbers.Integral):
            start = int(item) // self.batch_size * self.batch_size
            if start != self.block_start:
                self.block_start = start
                self.block = numeric_block(self.matrix, self.rows[start:start + self.batch_size], self.count, self.continuous)
            # LightGBM's Sequence sampler requires doubles. Returning an owned
            # row also avoids a second copy in its row iterator.
            return self.block[int(item) - start].astype("float64")
        return numeric_block(self.matrix, self.rows[item], self.count, self.continuous)


def load_continuous(cache, state):
    return {int(column): np.load(cache / item["file"], mmap_mode="r") for column, item in state["continuous_columns"].items()}


def native_parameters(config):
    parameters = dict(config["native_parameters"])
    iterations = parameters.pop("num_iterations")
    parameters.pop("categorical_feature", None)  # supplied explicitly to Dataset
    return parameters, iterations


def validate_native(config, meta, matrix, state, continuous):
    """Check native Sequence training against the sklearn wrapper on real data."""
    rows = meta.loc[meta.split.eq("train") & meta[config["target"]].notna()].sort_values("source_row").index
    rows = pd.Series(rows).sample(n=min(10000, len(rows)), random_state=config["sample_seed"]).to_numpy()
    holdout = rows[:1000]
    count = len(state["columns"])
    frame = shared.feature_frame(matrix, rows, count, state["columns"], continuous)
    target = meta.loc[rows, config["target"]].to_numpy()
    parameters, iterations = native_parameters(config)
    # Match the full requested 300 rounds; the small row subset keeps this check cheap.
    reference = lgb.LGBMRegressor(**config["model"]).fit(frame, target)
    sequence = MatrixSequence(matrix, rows, count, continuous)
    dataset = lgb.Dataset(sequence, label=target, feature_name=state["columns"], categorical_feature=[state["columns"].index("feature_country")], params=parameters, free_raw_data=True)
    native = lgb.train(parameters, dataset, num_boost_round=iterations)
    expected = reference.predict(shared.feature_frame(matrix, holdout, count, state["columns"], continuous))
    actual = native.predict(numeric_block(matrix, holdout, count, continuous))
    difference = float(np.max(np.abs(expected - actual)))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    print(f"Native/sklearn parity passed: {len(rows):,} actual rows, {count} columns, {iterations} rounds, max difference {difference:.3g}, peak RSS {peak_gib():.1f} GiB", flush=True)
    return {"rows": len(rows), "columns": count, "iterations": iterations, "maximum_absolute_prediction_difference": difference}


def reuse_prepared(cache, config, manifest):
    signature = json.loads((cache / "run_signature.json").read_text())
    state = json.loads((cache / "matrix_manifest.json").read_text())
    if signature["data_manifest"] != manifest:
        raise ValueError("Prepared input hashes/schemas differ from current data")
    if signature["python"] != platform.python_version():
        raise ValueError("Prepared Python version differs from the current environment")
    for package, version in signature["packages"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Prepared package version differs: {package}")
    if signature["source_hashes"]["runner"] != shared.sha256(shared.__file__) or state["builder_sha256"] != shared.sha256(Path(__file__).with_name("time_series.py")):
        raise ValueError("Prepared cache code does not match the imported preparation/builder")
    semantic_keys = ["data_version", "data_round", "target", "base_feature_set", "base_feature_storage", "transform_feature_set", "transform_history", "warmup", "binning"]
    for key in semantic_keys:
        if state["config"][key] != config[key]:
            raise ValueError(f"Prepared feature semantics differ: {key}")
    matrix = np.load(cache / "features.npy", mmap_mode="r")
    meta = pd.read_parquet(cache / "row_metadata.parquet")
    if matrix.shape != (len(meta), len(state["columns"])) or matrix.dtype != np.uint8:
        raise ValueError("Prepared matrix shape/dtype mismatch")
    return meta, matrix, state


def train_full(config, meta, matrix, state, continuous, cache):
    predictions = meta[shared.KEYS + [config["target"], "data_type", "split"]].copy()
    coverage, timings = [], []
    for period in ["original", "retrained"]:
        train_mask = meta.split.eq("train") if period == "original" else meta.date.le(config["retrain_through"])
        train_mask &= meta[config["target"]].notna()
        rows = meta.loc[train_mask].sort_values("source_row").index.to_numpy()
        last_train = meta.loc[rows, "date"].max()
        embargo = last_train + pd.offsets.BDay(config["embargo_business_days"])
        start = config["validation_start"] if period == "original" else config["retrain_validation_start"]
        evaluation = meta.split.eq("validation") & meta.data_type.eq("validation") & meta[config["target"]].notna() & meta.date.ge(start) & meta.date.gt(embargo) & meta.date.le(config["validation_end"])
        np.save(cache / f"{period}_training_rows.npy", rows)
        coverage.append({"period": period, "available_train_rows": len(rows), "used_train_rows": len(rows), "training_row_indices_sha256": shared.hashlib.sha256(rows.tobytes()).hexdigest(), "first_train_date": str(meta.loc[rows, "date"].min().date()), "last_train_date": str(last_train.date()), "embargo_end": str(embargo.date()), "first_evaluation_date": str(meta.loc[evaluation, "date"].min().date()), "last_evaluation_date": str(meta.loc[evaluation, "date"].max().date()), "evaluation_rows_before_benchmark_join": int(evaluation.sum()), "evaluation_dates_before_benchmark_join": int(meta.loc[evaluation, "date"].nunique())})
        for variant, count in [("unity", state["base_columns"]), ("unity_plus_transforms", len(state["columns"]))]:
            label = f"{period}_{variant}"
            prediction_path = cache / f"{label}_predictions.parquet"
            if prediction_path.exists():
                predictions[label] = pd.read_parquet(prediction_path, columns=[label])[label]
                continue
            parameters, iterations = native_parameters(config)
            sequence = MatrixSequence(matrix, rows, count, continuous)
            dataset = lgb.Dataset(sequence, label=meta.loc[rows, config["target"]].to_numpy(), feature_name=state["columns"][:count], categorical_feature=[state["columns"].index("feature_country")], params=parameters, free_raw_data=True)
            print(f"Constructing full Dataset {label}: {len(rows):,} rows, {count} columns", flush=True)
            started = time.monotonic()
            dataset.construct()
            construction_seconds = time.monotonic() - started
            print(f"Dataset {label} constructed in {construction_seconds:.0f}s, peak RSS {peak_gib():.1f} GiB", flush=True)
            trained_at = time.monotonic()

            def progress(environment):
                if (environment.iteration + 1) % 25 == 0:
                    print(f"{label}: {environment.iteration + 1}/{iterations} trees in {time.monotonic() - trained_at:.0f}s, peak RSS {peak_gib():.1f} GiB", flush=True)

            model = lgb.train(parameters, dataset, num_boost_round=iterations, callbacks=[progress])
            training_seconds = time.monotonic() - trained_at
            model.save_model(str(cache / f"{label}_model.txt"))
            # Release native training bins before full-universe prediction.
            del dataset, sequence
            gc.collect()
            predictions[label] = np.nan
            for _, group in meta.loc[evaluation].groupby("date"):
                predictions.loc[group.index, label] = model.predict(numeric_block(matrix, group.index.to_numpy(), count, continuous))
            predictions[shared.KEYS + [label]].to_parquet(prediction_path, index=False)
            timings.append({"model": label, "construction_seconds": construction_seconds, "training_seconds": training_seconds, "total_fit_and_prediction_seconds": time.monotonic() - started, "peak_rss_gib": peak_gib()})
            print(f"Finished full {label} in {time.monotonic() - started:.0f}s", flush=True)
            del model
            gc.collect()
    return predictions, coverage, timings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("full_training_config.json"))
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data/reproduction/v6.0")
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).parent / "data/full_training")
    parser.add_argument("--prepared-cache", type=Path, help="reuse a signed feature cache from reproduce.py")
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "reproduction_results/full")
    parser.add_argument("--source-post", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["training_sample_rows"] is not None:
        raise ValueError("This confirmation uses all training rows, not a sample")
    for path in [args.data_dir, args.cache_dir, args.results_dir]:
        path.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    manifest = []
    for name in ["features.json", "train.parquet", "validation.parquet", "validation_benchmark_models.parquet", "validation_neutralizer.parquet"]:
        path = args.data_dir / name
        if not path.exists():
            NumerAPI(show_progress_bars=False).download_dataset(f"{config['data_version']}/{name}", dest_path=str(path), round_num=config["data_round"])
        item = {"name": name, "bytes": path.stat().st_size, "sha256": shared.sha256(path)}
        if name.endswith(".parquet"):
            file = pq.ParquetFile(path)
            item.update(rows=file.metadata.num_rows, columns=len(file.schema.names), row_groups=file.num_row_groups)
        manifest.append(item)
    signature = {"config": config, "data_manifest": manifest, "sources": {name: shared.sha256(path) for name, path in {"full_runner": __file__, "shared_runner": shared.__file__, "builder": Path(__file__).with_name("time_series.py"), "plotter": Path(__file__).with_name("plot_reproduction.py"), "config": args.config}.items()}, "python": platform.python_version(), "platform": platform.platform(), "packages": dict(sorted((distribution.metadata["Name"], distribution.version) for distribution in importlib.metadata.distributions())), "source_post_sha256": shared.sha256(args.source_post) if args.source_post else None}
    signature_path = args.cache_dir / "run_signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError("Full-run cache signature differs; use a new --cache-dir")
    if not signature_path.exists() and any(
        list(args.cache_dir.glob(pattern)) for pattern in [
            "*_model.txt", "*_predictions.parquet", "matrix_manifest.json",
            "row_metadata.parquet", "*_training_rows.npy",
        ]
    ):
        raise ValueError("Unsigned full-run artifacts found; use a new --cache-dir")
    signature_path.write_text(json.dumps(signature, indent=2))
    for path in [Path(__file__), Path(shared.__file__), Path(__file__).with_name("time_series.py"), Path(__file__).with_name("plot_reproduction.py"), args.config]:
        shutil.copyfile(path, args.cache_dir / f"snapshot_{path.name}")
    shared.validate_kernel()
    if args.prepared_cache:
        meta, matrix, state = reuse_prepared(args.prepared_cache, config, manifest)
        feature_cache = args.prepared_cache
    else:
        meta, matrix, state = shared.prepare(config, args.data_dir, args.cache_dir)
        feature_cache = args.cache_dir
    continuous = load_continuous(feature_cache, state)
    parity = validate_native(config, meta, matrix, state, continuous)
    gc.collect()
    predictions, coverage, timings = train_full(config, meta, matrix, state, continuous, args.cache_dir)
    with threadpool_limits(limits=config["blas_threads"], user_api="blas"):
        scores = shared.score(config, predictions, args.data_dir, args.results_dir)
    result = {"purpose": "Full-training confirmation; same fixed 300-tree model, all eligible rows, no tuning", "config": config, "signature": signature, "data_manifest": manifest, "native_sklearn_parity": parity, "prepared_matrix": {"reused": bool(args.prepared_cache), "builder_sha256": state["builder_sha256"], "base_columns": state["base_columns"], "transform_columns": state["transform_columns"], "rows": state["rows"], "continuous_columns": len(state["continuous_columns"])}, "training_and_evaluation_coverage": coverage, "model_timings": timings, "metrics": shared.summarize(scores, config), "total_seconds": time.monotonic() - started, "peak_rss_gib": peak_gib()}
    (args.results_dir / "summary.json").write_text(json.dumps(result, indent=2))
    (args.results_dir / "config.json").write_text(json.dumps(config, indent=2))
    plot_scores(scores, args.results_dir / "cumulative_scores.svg")
    files = []
    for path in sorted(args.cache_dir.glob("*_model.txt")) + sorted(args.cache_dir.glob("*_predictions.parquet")) + sorted(args.cache_dir.glob("*_training_rows.npy")):
        files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": shared.sha256(path)})
    (args.results_dir / "output_manifest.json").write_text(json.dumps({"generated_after_run": True, "run_signature_sha256": shared.sha256(signature_path), "files": files}, indent=2))
    print(json.dumps(result["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
