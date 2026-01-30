"""Plot cumulative Numerai correlation between benchmark models."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Iterable, Sequence

import pandas as pd
import pyarrow.parquet as pq
from numerapi import NumerAPI

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required. Install with `.venv/bin/pip install matplotlib`."
    ) from exc

from agents.code.metrics import numerai_metrics
from agents.code.modeling.utils.constants import NUMERAI_DIR

AGENTS_DIR = Path(__file__).resolve().parents[2]


DEFAULT_DATA_VERSION_V52 = "v5.2"
DEFAULT_DATA_VERSION_V51 = "v5.1"
DEFAULT_DATA_VERSION_V5 = "v5.0"
DEFAULT_MODELS = ["v52_teager20", "v51_teager20", "v5_teager20"]

ALIASES: dict[str, Sequence[str]] = {
    "v52_teager20": ("v52_lgbm_teager2b20", "v52_lgbm_teager20"),
    "v51_teager20": ("v51_lgbm_teager2b20", "v51_lgbm_teager20", "v51_teager20"),
    "v5_teager20": ("v5_lgbm_teager2b20", "v5_lgbm_teager20", "v5_teager20"),
    "v52_ender20": ("v52_lgbm_ender20",),
    "ender20": ("v52_lgbm_ender20",),
    "v52_cyrus": ("v52_lgbm_cyrusd20",),
    "cyrus": ("v52_lgbm_cyrusd20",),
    "cyrusd20": ("v52_lgbm_cyrusd20",),
}

TARGET_ALIASES: dict[str, Sequence[str]] = {
    "ender20": ("target_ender_20", "target_ender20"),
    "v52_ender20": ("target_ender_20",),
    "cyrus": ("target_cyrusd_20", "target_cyrus_20", "target_cyrus20"),
    "cyrusd20": ("target_cyrusd_20",),
    "v52_cyrus": ("target_cyrusd_20",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot cumulative per-era numerai_corr for benchmark models vs ender/cyrus."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Optional v5.2 benchmark parquet path override.",
    )
    parser.add_argument(
        "--v51-path",
        type=Path,
        default=None,
        help="Optional v5.1 benchmark parquet path override.",
    )
    parser.add_argument(
        "--v5-path",
        type=Path,
        default=None,
        help="Optional v5 benchmark parquet path override.",
    )
    parser.add_argument(
        "--v52-version",
        type=str,
        default=DEFAULT_DATA_VERSION_V52,
        help="Data version for v52 benchmark models.",
    )
    parser.add_argument(
        "--v51-version",
        type=str,
        default=DEFAULT_DATA_VERSION_V51,
        help="Data version for v51 benchmark models.",
    )
    parser.add_argument(
        "--v5-version",
        type=str,
        default=DEFAULT_DATA_VERSION_V5,
        help="Data version for v5 benchmark models.",
    )
    parser.add_argument(
        "--target-path",
        type=Path,
        default=None,
        help="Optional v5.2 target parquet path override.",
    )
    parser.add_argument(
        "--v51-target-path",
        type=Path,
        default=None,
        help="Optional v5.1 target parquet path override.",
    )
    parser.add_argument(
        "--v5-target-path",
        type=Path,
        default=None,
        help="Optional v5 target parquet path override.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Benchmark model columns (or aliases) to plot.",
    )
    parser.add_argument(
        "--ender-col",
        type=str,
        default="ender20",
        help="Target column (or alias) for ender20 correlations.",
    )
    parser.add_argument(
        "--cyrus-col",
        type=str,
        default="cyrus",
        help="Target column (or alias) for cyrus correlations.",
    )
    parser.add_argument(
        "--era-col",
        type=str,
        default="era",
        help="Era column name in the benchmark parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the plot image.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively.",
    )
    parser.add_argument(
        "--list-columns",
        action="store_true",
        help="List available columns in the parquet(s) and exit.",
    )
    return parser.parse_args()


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text)


def _available_columns(path: Path) -> list[str]:
    return pq.read_schema(path).names


def _resolve_column(name: str, columns: Iterable[str]) -> str:
    if name in columns:
        return name
    candidates = ALIASES.get(name, [])
    for candidate in candidates:
        if candidate in columns:
            return candidate
    matches = [col for col in columns if name in col]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ValueError(
            f"Column '{name}' matched multiple columns: {matches}. "
            "Pass an explicit column name."
        )
    raise ValueError(
        f"Column '{name}' not found. Use --list-columns to inspect available columns."
    )


def _resolve_target_column(name: str, columns: Iterable[str]) -> str:
    if name in columns:
        return name
    candidates = TARGET_ALIASES.get(name, [])
    for candidate in candidates:
        if candidate in columns:
            return candidate
    matches = [col for col in columns if name in col]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ValueError(
            f"Target '{name}' matched multiple columns: {matches}. "
            "Pass an explicit column name."
        )
    raise ValueError(
        f"Target '{name}' not found. Use --list-columns to inspect available columns."
    )


def _infer_version(name: str, default_version: str, args: argparse.Namespace) -> str:
    lowered = name.lower()
    if lowered in {"ender20", "cyrus", "cyrusd20", "v52_ender20", "v52_cyrus"}:
        return args.v52_version
    if lowered.startswith("v52_"):
        return args.v52_version
    if lowered.startswith("v51_"):
        return args.v51_version
    if lowered.startswith("v5_"):
        return args.v5_version
    return default_version


def _load_version_frame(
    data_version: str,
    model_names: Sequence[str],
    era_col: str,
    id_col: str,
    napi: NumerAPI,
    path_override: Path | None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    paths: list[Path]
    if path_override is not None:
        path = path_override.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Benchmark parquet not found: {path}")
        paths = [path]
    else:
        full_path = (NUMERAI_DIR / data_version / "full_benchmark_models.parquet").resolve()
        if full_path.exists():
            paths = [full_path]
        else:
            train_path = (NUMERAI_DIR / data_version / "train_benchmark_models.parquet").resolve()
            val_path = (
                NUMERAI_DIR / data_version / "validation_benchmark_models.parquet"
            ).resolve()
            if not train_path.exists():
                train_path.parent.mkdir(parents=True, exist_ok=True)
                napi.download_dataset(
                    f"{data_version}/train_benchmark_models.parquet",
                    dest_path=str(train_path),
                )
            if not val_path.exists():
                val_path.parent.mkdir(parents=True, exist_ok=True)
                napi.download_dataset(
                    f"{data_version}/validation_benchmark_models.parquet",
                    dest_path=str(val_path),
                )
            paths = [train_path, val_path]

    columns = _available_columns(paths[0])
    resolved: dict[str, str] = {}
    for name in model_names:
        col = _resolve_column(name, columns)
        resolved.setdefault(col, name)
    read_cols = [era_col]
    if id_col in columns:
        read_cols.append(id_col)
    read_cols.extend(resolved.keys())
    read_cols = list(dict.fromkeys(read_cols))

    frames = []
    for path in paths:
        df = pd.read_parquet(path, columns=read_cols)
        if id_col not in df.columns:
            if df.index.name == id_col:
                df = df.reset_index()
            else:
                raise ValueError(
                    f"Expected '{id_col}' column in benchmark parquet: {path}"
                )
        frames.append(df)
    df = pd.concat(frames, axis=0, ignore_index=True)
    return df, resolved


def _load_targets_frame(
    data_version: str,
    target_names: Sequence[str],
    era_col: str,
    id_col: str,
    napi: NumerAPI,
    path_override: Path | None,
    fallback_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    paths: list[Path]
    if path_override is not None:
        path = path_override.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Target parquet not found: {path}")
        paths = [path]
    else:
        full_path = (NUMERAI_DIR / data_version / "full.parquet").resolve()
        if full_path.exists():
            paths = [full_path]
        elif fallback_path is not None and fallback_path.exists():
            print(
                f"Warning: using targets from {fallback_path} for {data_version}."
            )
            paths = [fallback_path]
        else:
            train_path = (NUMERAI_DIR / data_version / "train.parquet").resolve()
            val_path = (NUMERAI_DIR / data_version / "validation.parquet").resolve()
            if not train_path.exists():
                train_path.parent.mkdir(parents=True, exist_ok=True)
                napi.download_dataset(
                    f"{data_version}/train.parquet", dest_path=str(train_path)
                )
            if not val_path.exists():
                val_path.parent.mkdir(parents=True, exist_ok=True)
                napi.download_dataset(
                    f"{data_version}/validation.parquet", dest_path=str(val_path)
                )
            paths = [train_path, val_path]

    columns = _available_columns(paths[0])
    resolved: dict[str, str] = {}
    for name in target_names:
        col = _resolve_target_column(name, columns)
        resolved.setdefault(name, col)
    read_cols = [era_col]
    if id_col in columns:
        read_cols.append(id_col)
    read_cols.extend(resolved.values())
    read_cols = list(dict.fromkeys(read_cols))

    frames = []
    for path in paths:
        df = pd.read_parquet(path, columns=read_cols)
        if id_col not in df.columns:
            if df.index.name == id_col:
                df = df.reset_index()
            else:
                raise ValueError(f"Expected '{id_col}' column in target parquet: {path}")
        frames.append(df)
    df = pd.concat(frames, axis=0, ignore_index=True)
    return df, resolved


def _plot_panel(ax, cumsum: pd.DataFrame, labels: dict[str, str], title: str) -> None:
    for col in cumsum.columns:
        label = labels.get(col, col)
        ax.plot(cumsum.index, cumsum[col].values, label=label, linewidth=2)
    ax.set_title(title)
    ax.set_ylabel("Cumsum corr")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _sort_era_index(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.index.dtype == object:
        try:
            order = sorted(scores.index, key=lambda x: int(x))
            return scores.loc[order]
        except (TypeError, ValueError):
            return scores.sort_index()
    return scores.sort_index()


def _path_override(version: str, args: argparse.Namespace) -> Path | None:
    if version == args.v52_version:
        return args.data_path
    if version == args.v51_version:
        return args.v51_path
    if version == args.v5_version:
        return args.v5_path
    return None


def _target_path_override(version: str, args: argparse.Namespace) -> Path | None:
    if version == args.v52_version:
        return args.target_path
    if version == args.v51_version:
        return args.v51_target_path
    if version == args.v5_version:
        return args.v5_target_path
    return None


def main() -> None:
    args = parse_args()
    if len(set(args.models)) != len(args.models):
        raise ValueError("Duplicate model names provided.")

    napi = NumerAPI()
    version_models: dict[str, list[str]] = {}
    for name in args.models:
        version = _infer_version(name, args.v52_version, args)
        version_models.setdefault(version, []).append(name)

    if args.list_columns:
        for version in version_models:
            path_override = _path_override(version, args)
            if path_override is not None:
                path = path_override.expanduser().resolve()
                if not path.exists():
                    raise FileNotFoundError(f"Benchmark parquet not found: {path}")
            else:
                full_path = Path(f"{version}/full_benchmark_models.parquet")
                if full_path.exists():
                    path = full_path
                else:
                    path = Path(f"{version}/train_benchmark_models.parquet")
                    if not path.exists():
                        napi.download_dataset(str(path))
            cols = _available_columns(path)
            print(f"{version}:")
            print("\n".join(cols))
        return

    resolved_by_label: dict[str, str] = {}
    ender_title = None
    cyrus_title = None
    per_era_ender_frames = []
    per_era_cyrus_frames = []
    fallback_targets = None
    if args.target_path is not None:
        fallback_targets = args.target_path.expanduser().resolve()
    else:
        default_fallback = Path(f"{args.v52_version}/full.parquet")
        if default_fallback.exists():
            fallback_targets = default_fallback

    for version, names in version_models.items():
        df_version, resolved = _load_version_frame(
            version,
            names,
            args.era_col,
            "id",
            napi,
            _path_override(version, args),
        )
        targets_df, target_resolved = _load_targets_frame(
            version,
            [args.ender_col, args.cyrus_col],
            args.era_col,
            "id",
            napi,
            _target_path_override(version, args),
            fallback_targets,
        )
        ender_target = target_resolved.get(args.ender_col)
        cyrus_target = target_resolved.get(args.cyrus_col)
        if ender_target is None or cyrus_target is None:
            raise ValueError(
                f"Failed to resolve target columns for {version}."
            )
        if ender_title is None:
            ender_title = ender_target
        if cyrus_title is None:
            cyrus_title = cyrus_target
        df_version = df_version.merge(
            targets_df,
            on=["id", args.era_col],
            how="inner",
        )
        frame_models = []
        for col, label in resolved.items():
            existing = resolved_by_label.get(label)
            if existing and existing != col:
                raise ValueError(
                    f"Model '{label}' resolved to multiple columns: {existing}, {col}"
                )
            resolved_by_label[label] = col
            frame_models.append(col)
        if frame_models:
            per_era_ender = numerai_metrics.per_era_corr(
                df_version, frame_models, ender_target, era_col=args.era_col
            )
            per_era_cyrus = numerai_metrics.per_era_corr(
                df_version, frame_models, cyrus_target, era_col=args.era_col
            )
            per_era_ender_frames.append(per_era_ender)
            per_era_cyrus_frames.append(per_era_cyrus)

    model_labels = {resolved_by_label[name]: name for name in args.models}

    per_era_ender = pd.concat(per_era_ender_frames, axis=1)
    per_era_cyrus = pd.concat(per_era_cyrus_frames, axis=1)
    per_era_ender = _sort_era_index(per_era_ender)
    per_era_cyrus = _sort_era_index(per_era_cyrus)
    cumsum_ender = per_era_ender.cumsum()
    cumsum_cyrus = per_era_cyrus.cumsum()
    cumsum_ender = _sort_era_index(cumsum_ender)
    cumsum_cyrus = _sort_era_index(cumsum_cyrus)

    base_label = next(
        (name for name in args.models if name.lower().startswith("v5_")), None
    )
    if base_label is None:
        raise ValueError("Diff plot requires a v5.* model in --models.")
    base_col = resolved_by_label[base_label]
    base_ender = per_era_ender.get(base_col)
    base_cyrus = per_era_cyrus.get(base_col)
    if base_ender is None or base_cyrus is None:
        raise ValueError("Failed to locate v5.* base series for diff plots.")

    base_ender = base_ender.reindex(per_era_ender.index).fillna(0.0)
    base_cyrus = base_cyrus.reindex(per_era_cyrus.index).fillna(0.0)
    diff_ender = per_era_ender.sub(base_ender, axis=0)
    diff_cyrus = per_era_cyrus.sub(base_cyrus, axis=0)
    diff_ender[base_col] = 0.0
    diff_cyrus[base_col] = 0.0
    diff_cumsum_ender = diff_ender.cumsum()
    diff_cumsum_cyrus = diff_cyrus.cumsum()

    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    _plot_panel(
        axes[0],
        cumsum_ender,
        model_labels,
        f"Cumulative numerai_corr vs {ender_title or args.ender_col}",
    )
    _plot_panel(
        axes[1],
        cumsum_cyrus,
        model_labels,
        f"Cumulative numerai_corr vs {cyrus_title or args.cyrus_col}",
    )
    _plot_panel(
        axes[2],
        diff_cumsum_ender,
        model_labels,
        f"Cumulative diff vs {base_label} ({ender_title or args.ender_col})",
    )
    _plot_panel(
        axes[3],
        diff_cumsum_cyrus,
        model_labels,
        f"Cumulative diff vs {base_label} ({cyrus_title or args.cyrus_col})",
    )
    axes[3].set_xlabel("Era")
    plt.tight_layout()

    if args.output is None:
        plots_dir = AGENTS_DIR / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        stem = _slug(
            "benchmark_corr_cumsum_with_diffs_"
            f"{ender_title or args.ender_col}_vs_{cyrus_title or args.cyrus_col}"
        )
        output = plots_dir / f"{stem}.png"
    else:
        output = args.output
        output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Saved plot to {output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
