"""Display experiment results and plot per-era correlation curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd
from matplotlib import ticker

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required. Install with `.venv/bin/pip install matplotlib`."
    ) from exc

from agents.code.metrics import numerai_metrics

AGENTS_DIR = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display experiment metrics and plot OOF correlation curves."
    )
    parser.add_argument(
        "base_model",
        type=str,
        help="Baseline model name (results/predictions file stem).",
    )
    parser.add_argument(
        "experiment_models",
        nargs="+",
        help="One or more experiment model names to compare.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=AGENTS_DIR / "experiments" / "feature_set_experiments",
        help="Experiment output directory (contains results/ and predictions/).",
    )
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        default=AGENTS_DIR / "baselines",
        help="Baselines output directory (contains results/ and predictions/).",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="target",
        help="Target column name in predictions file.",
    )
    parser.add_argument(
        "--era-col",
        type=str,
        default="era",
        help="Era column name in predictions file.",
    )
    parser.add_argument(
        "--pred-col",
        type=str,
        default="prediction",
        help="Prediction column name in predictions file.",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="id",
        help="ID column name in predictions file.",
    )
    parser.add_argument(
        "--dark",
        action="store_true",
        help="Use a dark theme for plots.",
    )
    parser.add_argument(
        "--base-benchmark-model",
        type=str,
        default=None,
        help=(
            "Use benchmark model predictions as the baseline instead of a results file. "
            "Example: v52_lgbm_ender20. If set, base_model can be any label (e.g. 'benchmark')."
        ),
    )
    parser.add_argument(
        "--benchmark-data-path",
        type=Path,
        default=None,
        help="Optional benchmark parquet path override.",
    )
    parser.add_argument(
        "--benchmark-data-version",
        type=str,
        default=None,
        help="Override data version when loading benchmark predictions.",
    )
    parser.add_argument(
        "--start-era",
        type=int,
        default=575,
        help="Minimum era to include in plots (inclusive).",
    )
    parser.add_argument(
        "--max-xticks",
        type=int,
        default=12,
        help="Maximum number of x-axis ticks to show.",
    )
    return parser.parse_args()


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text)


def _sort_era_index(series: pd.Series) -> pd.Series:
    try:
        order = sorted(series.index, key=lambda x: int(x))
        return series.loc[order]
    except (TypeError, ValueError):
        return series.sort_index()


def _resolve_results_path(
    output_dir: Path, name: str, fallback_dir: Path | None = None
) -> Path | None:
    for base_dir in [output_dir, fallback_dir]:
        if base_dir is None:
            continue
        path = base_dir / "results" / f"{name}.json"
        if path.exists():
            return path
    return None


def _resolve_predictions_path(
    output_dir: Path, name: str, fallback_dir: Path | None = None
) -> Path:
    for base_dir in [output_dir, fallback_dir]:
        if base_dir is None:
            continue
        results_path = _resolve_results_path(base_dir, name)
        if results_path:
            data = json.loads(results_path.read_text())
            rel_path = data.get("output", {}).get("predictions_file")
            if rel_path:
                candidate = base_dir / rel_path
                if candidate.exists():
                    return candidate
        default_path = base_dir / "predictions" / f"{name}.parquet"
        if default_path.exists():
            return default_path
    raise FileNotFoundError(f"Predictions not found for {name}.")


def _load_predictions(
    path: Path,
    pred_col: str,
    target_col: str,
    era_col: str,
    id_col: str,
) -> pd.DataFrame:
    required = [pred_col, target_col, era_col, id_col]
    return pd.read_parquet(path, columns=required)


def _per_era_corr(
    df: pd.DataFrame, pred_col: str, target_col: str, era_col: str
) -> pd.Series:
    per_era = numerai_metrics.per_era_corr(df, [pred_col], target_col, era_col=era_col)
    if isinstance(per_era, pd.DataFrame):
        series = per_era[pred_col]
    else:
        series = per_era.squeeze()
    return _sort_era_index(series)


def _per_era_bmc(
    df: pd.DataFrame,
    pred_col: str,
    target_col: str,
    era_col: str,
    id_col: str,
    benchmark: pd.DataFrame,
    benchmark_col: str,
) -> pd.Series:
    enriched = numerai_metrics.attach_benchmark_predictions(
        df,
        benchmark,
        benchmark_col,
        era_col=era_col,
        id_col=id_col,
    )
    per_era = numerai_metrics.per_era_bmc(
        enriched,
        [pred_col],
        benchmark_col,
        target_col,
        era_col=era_col,
    )
    if isinstance(per_era, pd.DataFrame):
        series = per_era[pred_col]
    else:
        series = per_era.squeeze()
    return _sort_era_index(series)


def _load_metrics(results_path: Path) -> dict:
    data = json.loads(results_path.read_text())
    metrics = data.get("metrics", {})
    bmc = metrics.get("bmc", {})
    bmc_last = metrics.get("bmc_last_200_eras", {})
    small = metrics.get("small_bmc", {})
    small_last = metrics.get("small_bmc_last200", {})
    corr = metrics.get("corr", {})
    return {
        "model": results_path.stem,
        "feature_set": data.get("data", {}).get("feature_set"),
        "bmc_mean": bmc.get("mean"),
        "bmc_sharpe": bmc.get("sharpe"),
        "bmc_drawdown": bmc.get("max_drawdown"),
        "bmc_avg_corr_bench": bmc.get("avg_corr_with_benchmark"),
        "bmc_last200_mean": bmc_last.get("mean"),
        "bmc_last200_sharpe": bmc_last.get("sharpe"),
        "bmc_last200_drawdown": bmc_last.get("max_drawdown"),
        "bmc_last200_avg_corr_bench": bmc_last.get("avg_corr_with_benchmark"),
        "small_bmc_mean": small.get("mean"),
        "small_bmc_sharpe": small.get("sharpe"),
        "small_bmc_drawdown": small.get("max_drawdown"),
        "small_bmc_avg_corr_bench": small.get("avg_corr_with_benchmark"),
        "small_bmc_last200_mean": small_last.get("mean"),
        "small_bmc_last200_sharpe": small_last.get("sharpe"),
        "small_bmc_last200_drawdown": small_last.get("max_drawdown"),
        "small_bmc_last200_avg_corr_bench": small_last.get(
            "avg_corr_with_benchmark"
        ),
        "corr_mean": corr.get("mean"),
        "corr_sharpe": corr.get("sharpe"),
    }


def _summary_row(summary: pd.Series, prefix: str) -> dict:
    return {
        f"{prefix}_mean": summary.get("mean"),
        f"{prefix}_sharpe": summary.get("sharpe"),
        f"{prefix}_drawdown": summary.get("max_drawdown"),
    }


def _metrics_from_predictions(
    name: str,
    df: pd.DataFrame,
    pred_col: str,
    target_col: str,
    era_col: str,
    id_col: str,
    benchmark: pd.DataFrame,
    benchmark_col: str,
) -> dict:
    corr = numerai_metrics.per_era_corr(df, [pred_col], target_col, era_col=era_col)
    corr_summary = numerai_metrics.summarize_scores(corr).loc[pred_col]

    bmc_df = numerai_metrics.per_era_bmc(
        df,
        [pred_col],
        benchmark_col,
        target_col,
        era_col=era_col,
    )
    bmc_summary = numerai_metrics.summarize_scores(bmc_df).loc[pred_col]
    bmc_last = numerai_metrics.summarize_scores(
        numerai_metrics._last_n_eras(bmc_df, 200)
    ).loc[pred_col]

    benchmark_corr = numerai_metrics.per_era_pred_corr(
        df, [pred_col], benchmark_col, era_col=era_col
    )
    benchmark_corr_mean = benchmark_corr.mean().get(pred_col)
    benchmark_corr_last = numerai_metrics._last_n_eras(benchmark_corr, 200).mean().get(
        pred_col
    )

    row = {"model": name, "feature_set": None}
    row.update(_summary_row(bmc_summary, "bmc"))
    row["bmc_avg_corr_bench"] = benchmark_corr_mean
    row.update(_summary_row(bmc_last, "bmc_last200"))
    row["bmc_last200_avg_corr_bench"] = benchmark_corr_last
    row.update(_summary_row(corr_summary, "corr"))
    row.update(
        {
            "small_bmc_mean": np.nan,
            "small_bmc_sharpe": np.nan,
            "small_bmc_drawdown": np.nan,
            "small_bmc_avg_corr_bench": np.nan,
            "small_bmc_last200_mean": np.nan,
            "small_bmc_last200_sharpe": np.nan,
            "small_bmc_last200_drawdown": np.nan,
            "small_bmc_last200_avg_corr_bench": np.nan,
        }
    )
    return row


def _format_table(df: pd.DataFrame) -> str:
    return df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )


def _apply_dark_theme(ax):
    ax.set_facecolor("black")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.grid(True, alpha=0.3, color="white")


def _filter_by_start_era(series: pd.Series, start_era: int | None) -> pd.Series:
    if start_era is None:
        return series

    def _coerce_era(value) -> int | None:
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        if not digits:
            return None
        return int(digits)

    mask_list = []
    for item in series.index:
        era_value = _coerce_era(item)
        mask_list.append(era_value is None or era_value >= start_era)
    mask = np.array(mask_list, dtype=bool)
    return series[mask]


def _set_xticks(ax, index, max_ticks: int | None):
    if max_ticks is None or max_ticks <= 0:
        ax.tick_params(axis="x", rotation=45)
        return
    idx = list(index)
    if not idx:
        return
    if len(idx) <= max_ticks:
        ax.tick_params(axis="x", rotation=45)
        return
    step = max(1, len(idx) // max_ticks)
    ticks = idx[::step]
    if ticks and ticks[-1] != idx[-1]:
        ticks.append(idx[-1])
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], rotation=45, ha="right")


def _plot_curves(
    base_name: str,
    base_cumsum: pd.Series,
    model_cumsums: dict[str, pd.Series],
    bmc_cumsums: dict[str, pd.Series],
    output_dir: Path,
    dark_mode: bool = False,
    max_xticks: int | None = None,
) -> Path:
    rows = 3 if bmc_cumsums else 2
    fig, axes = plt.subplots(
        rows, 1, figsize=(12, 4 * rows), sharex=True, facecolor="black" if dark_mode else None
    )
    if rows == 1:
        axes = [axes]

    base_color = "white" if dark_mode else None
    base_alpha = 0.5 if dark_mode else 1.0
    model_colors: dict[str, str] = {}
    if dark_mode and len(model_cumsums) == 1:
        only_model = next(iter(model_cumsums))
        model_colors[only_model] = "orange"
    axes[0].plot(
        base_cumsum.index,
        base_cumsum.values,
        label=base_name,
        linewidth=2,
        color=base_color,
        alpha=base_alpha,
    )
    for name, series in model_cumsums.items():
        axes[0].plot(
            series.index,
            series.values,
            label=name,
            linewidth=2 if name in model_colors else 1.5,
            color=model_colors.get(name),
            alpha=0.9 if dark_mode else 0.8,
        )
    axes[0].set_title("Cumulative per-era correlation (OOF)")
    axes[0].set_ylabel("Cumsum corr")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for name, series in model_cumsums.items():
        common = base_cumsum.index.intersection(series.index)
        delta = series.loc[common] - base_cumsum.loc[common]
        axes[1].plot(
            delta.index,
            delta.values,
            label=f"{name} - {base_name}",
            linewidth=2 if name in model_colors else 1.5,
            color=model_colors.get(name),
            alpha=0.9 if dark_mode else 0.8,
        )
    axes[1].axhline(0, color="white" if dark_mode else "black", linewidth=1, alpha=0.5)
    axes[1].set_title("Delta vs baseline (cumsum corr)")
    axes[1].set_ylabel("Delta cumsum corr")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if bmc_cumsums:
        axes[2].plot(
            bmc_cumsums[base_name].index,
            bmc_cumsums[base_name].values,
            label=base_name,
            linewidth=2,
            color=base_color,
            alpha=base_alpha,
        )
        for name, series in bmc_cumsums.items():
            if name == base_name:
                continue
            axes[2].plot(
                series.index,
                series.values,
                label=name,
                linewidth=2 if name in model_colors else 1.5,
                color=model_colors.get(name),
                alpha=0.9 if dark_mode else 0.8,
            )
        axes[2].set_title("Cumulative per-era BMC (OOF)")
        axes[2].set_ylabel("Cumsum BMC")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    if dark_mode:
        for ax in axes:
            _apply_dark_theme(ax)
            legend = ax.get_legend()
            if legend is not None:
                legend.get_frame().set_facecolor("black")
                legend.get_frame().set_edgecolor("white")
                for text in legend.get_texts():
                    text.set_color("white")

    for ax in axes:
        _set_xticks(ax, base_cumsum.index, max_xticks)

    plt.tight_layout()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    stem = _slug(f"{base_name}_vs_{next(iter(model_cumsums))}")
    if len(model_cumsums) > 1:
        stem = f"{stem}_plus_{len(model_cumsums) - 1}"
    if dark_mode:
        stem = f"{stem}_dark"
    plot_path = plots_dir / f"{stem}.png"
    fig.savefig(plot_path, dpi=150)
    return plot_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    baselines_dir = args.baselines_dir.resolve()

    metrics_rows = []

    use_benchmark_base = args.base_benchmark_model is not None
    if args.base_model == "benchmark" and args.base_benchmark_model is None:
        args.base_benchmark_model = "v52_lgbm_ender20"
        use_benchmark_base = True

    base_results_path = None
    if not use_benchmark_base:
        base_results_path = _resolve_results_path(
            output_dir, args.base_model, baselines_dir
        )
        if not base_results_path:
            raise FileNotFoundError(
                f"Results not found for {args.base_model} in {output_dir / 'results'} or {baselines_dir / 'results'}"
            )
        metrics_rows.append(_load_metrics(base_results_path))

    for name in args.experiment_models:
        results_path = _resolve_results_path(output_dir, name)
        if not results_path:
            raise FileNotFoundError(
                f"Results not found for {name} in {output_dir / 'results'}"
            )
        metrics_rows.append(_load_metrics(results_path))

    reference_results_path = base_results_path
    if reference_results_path is None:
        if not args.experiment_models:
            raise ValueError("Benchmark baseline requires at least one experiment model.")
        reference_results_path = _resolve_results_path(
            output_dir, args.experiment_models[0]
        )
        if not reference_results_path:
            raise FileNotFoundError(
                f"Results not found for {args.experiment_models[0]} in {output_dir / 'results'}"
            )

    reference_data = json.loads(reference_results_path.read_text())
    data_version = (
        args.benchmark_data_version
        or reference_data.get("data", {}).get("data_version", "v5.2")
    )
    benchmark_model = args.base_benchmark_model or reference_data.get(
        "benchmark", {}
    ).get("model", "v52_lgbm_ender20")

    if args.benchmark_data_path is not None:
        benchmark, benchmark_col = numerai_metrics.load_benchmark_predictions_from_path(
            args.benchmark_data_path,
            benchmark_model,
            era_col=args.era_col,
            id_col=args.id_col,
        )
    else:
        benchmark, benchmark_col = numerai_metrics.load_benchmark_predictions(
            data_version,
            benchmark_model=benchmark_model,
            era_col=args.era_col,
        )

    if use_benchmark_base:
        template_path = _resolve_predictions_path(
            output_dir, args.experiment_models[0], baselines_dir
        )
        template_df = _load_predictions(
            template_path, args.pred_col, args.target_col, args.era_col, args.id_col
        )
        attached = numerai_metrics.attach_benchmark_predictions(
            template_df,
            benchmark,
            benchmark_col,
            era_col=args.era_col,
            id_col=args.id_col,
        )
        base_df = attached.copy()
        base_df[args.pred_col] = base_df[benchmark_col].to_numpy()
        base_name = benchmark_model
        metrics_rows.insert(
            0,
            _metrics_from_predictions(
                base_name,
                base_df,
                args.pred_col,
                args.target_col,
                args.era_col,
                args.id_col,
                benchmark,
                benchmark_col,
            ),
        )
    else:
        base_path = _resolve_predictions_path(
            output_dir, args.base_model, baselines_dir
        )
        base_df = _load_predictions(
            base_path, args.pred_col, args.target_col, args.era_col, args.id_col
        )
        base_name = args.base_model

    metrics_df = pd.DataFrame(metrics_rows)
    print("Model metrics:")
    print(_format_table(metrics_df))

    base_corr = _per_era_corr(base_df, args.pred_col, args.target_col, args.era_col)
    base_corr = _filter_by_start_era(base_corr, args.start_era)

    model_corrs = {}
    for name in args.experiment_models:
        pred_path = _resolve_predictions_path(output_dir, name)
        df = _load_predictions(
            pred_path, args.pred_col, args.target_col, args.era_col, args.id_col
        )
        corr = _per_era_corr(df, args.pred_col, args.target_col, args.era_col)
        corr = _filter_by_start_era(corr, args.start_era)
        model_corrs[name] = corr

    common_eras = base_corr.index
    for corr in model_corrs.values():
        common_eras = common_eras.intersection(corr.index)
    if common_eras.empty:
        raise ValueError("No overlapping eras across base and experiment models.")
    base_corr = base_corr.loc[common_eras]
    base_cumsum = base_corr.cumsum()
    model_cumsums = {
        name: corr.loc[common_eras].cumsum() for name, corr in model_corrs.items()
    }

    bmc_cumsums = {}
    base_bmc = _per_era_bmc(
        base_df,
        args.pred_col,
        args.target_col,
        args.era_col,
        args.id_col,
        benchmark,
        benchmark_col,
    )
    base_bmc = _filter_by_start_era(base_bmc, args.start_era)
    model_bmcs = {}
    for name in args.experiment_models:
        pred_path = _resolve_predictions_path(output_dir, name)
        df = _load_predictions(
            pred_path, args.pred_col, args.target_col, args.era_col, args.id_col
        )
        bmc = _per_era_bmc(
            df,
            args.pred_col,
            args.target_col,
            args.era_col,
            args.id_col,
            benchmark,
            benchmark_col,
        )
        bmc = _filter_by_start_era(bmc, args.start_era)
        model_bmcs[name] = bmc

    common_bmc_eras = base_bmc.index
    for bmc in model_bmcs.values():
        common_bmc_eras = common_bmc_eras.intersection(bmc.index)
    if common_bmc_eras.empty:
        raise ValueError("No overlapping eras for BMC across base and experiment models.")
    base_bmc = base_bmc.loc[common_bmc_eras]
    bmc_cumsums[base_name] = base_bmc.cumsum()
    for name, bmc in model_bmcs.items():
        bmc_cumsums[name] = bmc.loc[common_bmc_eras].cumsum()

    plot_path = _plot_curves(
        base_name,
        base_cumsum,
        model_cumsums,
        bmc_cumsums,
        output_dir,
        dark_mode=args.dark,
        max_xticks=args.max_xticks,
    )
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
