from __future__ import annotations

import argparse
from pathlib import Path

from .constants import DEFAULT_CONFIG_PATH
from .pipeline import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the base Numerai model."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the config file (.py or .json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base output directory for results and predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(args.config, args.output_dir)
