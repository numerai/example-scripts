import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = ROOT / "agents"
sys.path.insert(0, str(AGENTS_DIR))

from agent_base_model import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_config,
    resolve_small_bmc_baseline_path,
)
import show_experiment  # noqa: E402


class TestConfigLayout(unittest.TestCase):
    def test_default_config_path_exists(self):
        self.assertTrue(
            DEFAULT_CONFIG_PATH.exists(),
            f"Default config missing: {DEFAULT_CONFIG_PATH}",
        )

    def test_load_all_configs(self):
        config_dirs = [
            AGENTS_DIR / "baselines" / "configs",
            *sorted((AGENTS_DIR / "experiments").glob("*/configs")),
        ]
        config_files = []
        for config_dir in config_dirs:
            if config_dir.exists():
                config_files.extend(sorted(config_dir.glob("*.py")))

        self.assertTrue(config_files, "No config files discovered.")

        for path in config_files:
            with self.subTest(path=path):
                config = load_config(path)
                self.assertIsInstance(config, dict)
                for key in ["data", "model", "training", "output", "preprocessing"]:
                    self.assertIn(key, config)
                self.assertIn("type", config["model"])
                self.assertIn("params", config["model"])
                for key in ["data_version", "feature_set", "target_col", "era_col"]:
                    self.assertIn(key, config["data"])

    def test_resolve_small_bmc_baseline_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            baselines_dir = tmp / "baselines"
            predictions_dir = tmp / "predictions"
            baselines_dir_preds = baselines_dir / "predictions"
            baselines_dir_preds.mkdir(parents=True)
            predictions_dir.mkdir(parents=True)

            baseline_name = "small_lgbm_ender20_baseline"
            baseline_path = baselines_dir_preds / f"{baseline_name}.parquet"
            baseline_path.touch()

            resolved = resolve_small_bmc_baseline_path(
                baseline_name,
                predictions_dir,
                baselines_dir,
            )
            self.assertEqual(resolved, baseline_path)

            override = tmp / "override.parquet"
            override.touch()
            resolved_override = resolve_small_bmc_baseline_path(
                baseline_name,
                predictions_dir,
                baselines_dir,
                override_path=str(override),
            )
            self.assertEqual(resolved_override, override.resolve())

    def test_show_experiment_path_resolution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            output_dir = tmp / "experiments" / "catboost_small_baseline"
            baselines_dir = tmp / "baselines"
            (output_dir / "results").mkdir(parents=True)
            (output_dir / "predictions").mkdir(parents=True)
            (baselines_dir / "results").mkdir(parents=True)
            (baselines_dir / "predictions").mkdir(parents=True)

            base_name = "small_lgbm_ender20_baseline"
            base_results = baselines_dir / "results" / f"{base_name}.json"
            base_preds = baselines_dir / "predictions" / f"{base_name}.parquet"
            base_preds.touch()
            base_results.write_text(
                json.dumps(
                    {"output": {"predictions_file": f"predictions/{base_name}.parquet"}}
                )
            )

            exp_name = "small_catboost_baseline"
            exp_results = output_dir / "results" / f"{exp_name}.json"
            exp_preds = output_dir / "predictions" / f"{exp_name}.parquet"
            exp_preds.touch()
            exp_results.write_text(
                json.dumps(
                    {"output": {"predictions_file": f"predictions/{exp_name}.parquet"}}
                )
            )

            resolved_results = show_experiment._resolve_results_path(
                output_dir, base_name, baselines_dir
            )
            self.assertEqual(resolved_results, base_results)

            resolved_base_preds = show_experiment._resolve_predictions_path(
                output_dir, base_name, baselines_dir
            )
            self.assertEqual(resolved_base_preds, base_preds)

            resolved_exp_preds = show_experiment._resolve_predictions_path(
                output_dir, exp_name, baselines_dir
            )
            self.assertEqual(resolved_exp_preds, exp_preds)


if __name__ == "__main__":
    unittest.main()
