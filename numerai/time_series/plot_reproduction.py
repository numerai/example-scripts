"""Render the recorded per-date comparison with readable yearly date labels.

    python plot_reproduction.py reproduction_results/sampled/per_date_scores.csv
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def plot_scores(scores: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex="col")
    variants = [("unity", "Unity", "#2673b8"), ("unity_plus_transforms", "Unity + transforms", "#dc6d1b")]
    for column, (period, title) in enumerate([
        ("original", "Original training"), ("retrained", "Retrained through 2022"),
    ]):
        for variant, label, color in variants:
            group = scores[scores.period.eq(period) & scores.variant.eq(variant)].sort_values("date")
            for row, metric in enumerate(["ncorr", "neutral_bmc"]):
                axes[row, column].plot(pd.to_datetime(group.date), group[metric].cumsum(), color=color, label=label)
        axes[0, column].set_title(title)
        axes[1, column].set_xlabel("Validation date")
        for row, ylabel in enumerate(["Cumulative NCORR", "Cumulative neutral BMC"]):
            axis = axes[row, column]
            axis.set_ylabel(ylabel)
            axis.xaxis.set_major_locator(mdates.YearLocator())
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            axis.tick_params(labelsize=9)
            axis.grid(alpha=0.2)
            axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scores", type=Path)
    parser.add_argument("--output", type=Path, help="default: cumulative_scores.svg beside the CSV")
    args = parser.parse_args()
    plot_scores(pd.read_csv(args.scores), args.output or args.scores.with_name("cumulative_scores.svg"))


if __name__ == "__main__":
    main()
