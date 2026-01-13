from __future__ import annotations

import json
import unittest
from pathlib import Path

from agents.modeling.utils.pipeline import run_training


class TestSmallLgbmEnder20Baseline(unittest.TestCase):
    def test_pipeline_metrics(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        config_path = (
            repo_root
            / "agents"
            / "baselines"
            / "configs"
            / "small_lgbm_ender20_baseline.py"
        )
        _, results_path = run_training(config_path)

        with open(results_path, "r", encoding="utf-8") as results_file:
            results = json.load(results_file)

        corr_mean = results["metrics"]["corr"]["mean"]
        bmc_mean = results["metrics"]["bmc"]["mean"]
        avg_corr_with_benchmark = results["metrics"]["bmc"]["avg_corr_with_benchmark"]

        self.assertGreater(corr_mean, 0.005)
        self.assertLess(corr_mean, 0.03)
        self.assertGreater(avg_corr_with_benchmark, 0.4)
        self.assertLess(avg_corr_with_benchmark, 0.8)
        self.assertGreater(bmc_mean, -0.005)
        self.assertLess(bmc_mean, 0.005)
