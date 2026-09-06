"""Blend OOF prediction files with per-era rank-gauss weights and score BMC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from agents.code.metrics import numerai_metrics
from agents.code.modeling.utils.constants import (
    DEFAULT_BENCHMARK_MODEL,
    DEFAULT_TARGET_COL,
)
from agents.code.modeling.utils.ensemble import blend_rank_gauss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank-gauss blend prediction files.")
    parser.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        type=Path,
        help="Prediction parquet files to blend.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        required=True,
        type=float,
        help="Blend weights, one per predictions file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Experiment directory (writes predictions/ and results/).",
    )
    parser.add_argument(
        "--results-name",
        type=str,
        required=True,
        help="Stem for the blended predictions and results files.",
    )
    parser.add_argument("--pred-col", type=str, default="prediction")
    parser.add_argument("--target-col", type=str, default=DEFAULT_TARGET_COL)
    parser.add_argument("--era-col", type=str, default="era")
    parser.add_argument("--id-col", type=str, default="id")
    parser.add_argument("--data-version", type=str, default="v5.3")
    parser.add_argument("--benchmark-model", type=str, default=DEFAULT_BENCHMARK_MODEL)
    parser.add_argument("--benchmark-data-path", type=Path, default=None)
    return parser.parse_args()


def _load_aligned(
    paths: list[Path],
    pred_col: str,
    target_col: str,
    era_col: str,
    id_col: str,
) -> pd.DataFrame:
    frames = []
    for i, path in enumerate(paths):
        columns = [pred_col, target_col, era_col, id_col]
        frame = pd.read_parquet(path, columns=columns).rename(
            columns={pred_col: f"pred_{i}"}
        )
        frames.append(frame)
    merged = frames[0]
    for i, frame in enumerate(frames[1:], start=1):
        merged = merged.merge(
            frame[[id_col, f"pred_{i}"]],
            on=id_col,
            how="inner",
        )
    return merged


def main() -> None:
    args = parse_args()
    if len(args.predictions) != len(args.weights):
        raise ValueError("--predictions and --weights must have the same length.")

    merged = _load_aligned(
        args.predictions,
        args.pred_col,
        args.target_col,
        args.era_col,
        args.id_col,
    )
    pred_cols = [f"pred_{i}" for i in range(len(args.predictions))]
    blended = blend_rank_gauss(
        merged,
        pred_cols,
        args.weights,
        era_col=args.era_col,
        out_col="prediction",
    )
    out = merged[[args.id_col, args.era_col, args.target_col]].copy()
    out["prediction"] = blended.to_numpy()

    predictions_dir = args.output_dir / "predictions"
    results_dir = args.output_dir / "results"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = predictions_dir / f"{args.results_name}.parquet"
    out.to_parquet(predictions_path, index=False)

    summaries = numerai_metrics.summarize_prediction_file_with_bmc(
        predictions_path,
        ["prediction"],
        args.target_col,
        args.data_version,
        benchmark_model=args.benchmark_model,
        benchmark_data_path=args.benchmark_data_path,
        era_col=args.era_col,
        id_col=args.id_col,
    )
    results = {
        "model": {
            "type": "RankGaussBlend",
            "params": {
                "weights": args.weights,
                "sources": [str(path) for path in args.predictions],
            },
        },
        "data": {"target": args.target_col, "data_version": args.data_version},
        "benchmark": {
            "model": args.benchmark_model,
            "file": str(args.benchmark_data_path) if args.benchmark_data_path else None,
        },
        "output": {
            "output_dir": str(args.output_dir),
            "predictions_file": f"predictions/{args.results_name}.parquet",
        },
        "metrics": {
            "corr": summaries["corr"].loc["prediction"].to_dict(),
            "bmc": summaries["bmc"].loc["prediction"].to_dict(),
            "bmc_last_200_eras": summaries["bmc_last_200_eras"]
            .loc["prediction"]
            .to_dict(),
        },
    }
    results_path = results_dir / f"{args.results_name}.json"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"Saved blended predictions to {predictions_path}")
    print(f"Saved blended results to {results_path}")


if __name__ == "__main__":
    main()
