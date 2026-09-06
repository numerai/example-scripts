"""Build downsampled_full parquet without loading the entire v5.3 table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from numerapi import NumerAPI

from agents.code.metrics import numerai_metrics
from agents.code.modeling.utils.constants import NUMERAI_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build lean downsampled v5.3 files.")
    parser.add_argument("--data-version", default="v5.3")
    parser.add_argument("--era-step", type=int, default=4)
    return parser.parse_args()


def _unique_eras(path: Path) -> list:
    eras = pd.read_parquet(path, columns=["era"])["era"]
    return sorted(eras.unique(), key=lambda x: int(x))


def _read_kept_eras(path: Path, keep_eras: set[str], extra_filter=None) -> pd.DataFrame:
    table = pq.read_table(path, filters=[("era", "in", list(keep_eras))])
    df = table.to_pandas()
    if extra_filter is not None:
        df = extra_filter(df)
    if df.index.name and df.index.name not in df.columns:
        df = df.reset_index()
    return df


def main() -> None:
    args = parse_args()
    data_dir = NUMERAI_DIR / args.data_version
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "validation.parquet"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("train.parquet and validation.parquet must already exist.")

    train_eras = _unique_eras(train_path)
    val_eras = _unique_eras(val_path)
    all_eras = sorted(set(train_eras) | set(val_eras), key=lambda x: int(x))
    keep_eras = {era for i, era in enumerate(all_eras) if i % args.era_step == 0}
    print(f"Keeping {len(keep_eras)} / {len(all_eras)} eras")

    train = _read_kept_eras(train_path, keep_eras)
    validation = _read_kept_eras(
        val_path,
        keep_eras,
        extra_filter=lambda df: df[df["data_type"] == "validation"] if "data_type" in df.columns else df,
    )
    full = pd.concat([train, validation], ignore_index=True)
    full = full.drop(columns=["data_type"], errors="ignore")
    out = data_dir / "downsampled_full.parquet"
    full.to_parquet(out, index=False)
    print(f"Wrote {out} rows={len(full)} cols={full.shape[1]}")

    napi = NumerAPI()
    bench_path = numerai_metrics.ensure_full_benchmark_models(napi, args.data_version)
    ids = full["id"].dropna().unique()
    benchmark = pd.read_parquet(bench_path)
    if "id" in benchmark.columns:
        benchmark = benchmark.set_index("id")
    benchmark = benchmark.loc[benchmark.index.intersection(ids)]
    bench_out = data_dir / "downsampled_full_benchmark_models.parquet"
    benchmark.to_parquet(bench_out)
    print(f"Wrote {bench_out} rows={len(benchmark)}")


if __name__ == "__main__":
    main()
