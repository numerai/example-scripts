"""Summarize Numerai agent results into readable tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

AGENTS_DIR = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize agent results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=AGENTS_DIR / "results",
        help="Directory containing results JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Experiment output directory (uses <output-dir>/results).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows shown in the leaderboard.",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="bmc_mean",
        help="Column to sort by (default: bmc_mean).",
    )
    return parser.parse_args()


def _load_results(path: Path) -> dict:
    data = json.loads(path.read_text())
    metrics = data.get("metrics", {})
    bmc = metrics.get("bmc_last_200_eras") or metrics.get("bmc") or {}
    corr = metrics.get("corr") or {}
    return {
        "name": path.stem,
        "feature_set": data.get("data", {}).get("feature_set"),
        "model_type": data.get("model", {}).get("type"),
        "bmc_mean": bmc.get("mean"),
        "bmc_sharpe": bmc.get("sharpe"),
        "bmc_consistency": bmc.get("consistency"),
        "bmc_drawdown": bmc.get("max_drawdown"),
        "bmc_avg_corr_with_benchmark": bmc.get("avg_corr_with_benchmark"),
        "corr_mean": corr.get("mean"),
        "corr_sharpe": corr.get("sharpe"),
    }


def _format_table(df: pd.DataFrame) -> str:
    return df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    if args.output_dir is not None:
        results_dir = args.output_dir / "results"
    rows = []
    missing = []

    for path in sorted(results_dir.glob("*.json")):
        row = _load_results(path)
        if row["bmc_mean"] is None:
            missing.append(path.name)
        rows.append(row)

    if not rows:
        print(f"No results found in {results_dir}")
        return

    df = pd.DataFrame(rows)
    leaderboard = df.dropna(subset=["bmc_mean"]).copy()
    leaderboard = leaderboard.sort_values(
        by=args.sort, ascending=False, na_position="last"
    )
    if args.limit is not None:
        leaderboard = leaderboard.head(args.limit)

    print("BMC last 200 eras leaderboard (sorted by bmc_mean):")
    print(
        _format_table(
            leaderboard[
                [
                    "name",
                    "feature_set",
                    "model_type",
                    "bmc_mean",
                    "bmc_sharpe",
                    "bmc_consistency",
                    "bmc_avg_corr_with_benchmark",
                    "corr_mean",
                ]
            ]
        )
    )

    if missing:
        print("\nMissing BMC metrics:")
        for name in missing:
            print(f"- {name}")


if __name__ == "__main__":
    main()
