"""Small deterministic checks for time windows, ticker isolation, and causality."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from time_series import build_features, latest_version, main


class TimeSeriesTests(unittest.TestCase):
    def build(self, values, dates=None, tickers=None):
        return build_features(pd.DataFrame({
            "numerai_ticker": tickers if tickers is not None else ["A"] * len(values),
            "date": dates if dates is not None else pd.date_range("2025-01-01", periods=len(values), freq="7D"),
            "feature_x": values,
        }), ["feature_x"])

    def test_ties_and_changes(self):
        result = self.build([0, 1, 1, 0])
        np.testing.assert_allclose(result.feature_x_rank_52w, [1, 1, 5 / 6, 3 / 8])
        np.testing.assert_allclose(result.feature_x_weeks_since_high_52w, [0, 0, 0, 1])
        np.testing.assert_allclose(result.feature_x_weeks_since_change, [np.nan, 0, 1, 0], equal_nan=True)

    def test_elapsed_time_and_window_boundaries(self):
        result = self.build([10, 0, 2, 1], ["2025-01-01", "2025-04-02", "2025-12-31", "2026-01-07"])
        # The first value is exactly 13 weeks old at row 1, and 52 at row 2.
        np.testing.assert_allclose(result.feature_x_mean_13w_minus_52w, [0, -5, 1, 0.5])
        np.testing.assert_allclose(result.feature_x_rank_52w, [1, 0.5, 1, 2 / 3])
        np.testing.assert_allclose(result.feature_x_weeks_since_high_52w, [0, 13, 0, 1])

    def test_missing_values_break_unchanged_run(self):
        result = self.build([1, 2, np.nan, 2, 2, 1])
        self.assertTrue(result.filter(like="feature_x_").iloc[2].isna().all())
        np.testing.assert_allclose(result.feature_x_weeks_since_change, [np.nan, 0, np.nan, np.nan, np.nan, 0], equal_nan=True)

    def test_future_and_other_tickers_do_not_affect_history(self):
        past = self.build([1, 2, 1])
        mixed = self.build(
            [99, 1, 1, -100, 2],
            ["2025-01-08", "2025-01-15", "2025-01-01", "2025-01-22", "2025-01-08"],
            ["B", "A", "A", "A", "A"],
        )
        pd.testing.assert_frame_equal(past, mixed[mixed.numerai_ticker.eq("A")].iloc[:3].reset_index(drop=True))

    def test_duplicates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "one observation"):
            self.build([1, 2], ["2025-01-01", "2025-01-01"])

    def test_latest_version_is_numeric(self):
        files = [f"{v}/{name}" for v in ["v5.3", "v6.9", "v6.10"] for name in [
            "features.json", "train.parquet", "validation.parquet", "live.parquet",
        ]]
        self.assertEqual(latest_version(files + ["v7.0/features.json"]), "v6.10")

    def test_cli_carries_history_across_splits_and_writes_joinable_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "v6.0"
            directory.mkdir()
            (directory / "features.json").write_text(json.dumps({
                "feature_sets": {"medium": ["feature_x", "feature_country"]},
            }))
            for split, date, value in [
                ("train", "2025-01-01", 1),
                ("validation", "2025-01-08", 2),
                ("live", "2025-01-15", 1),
            ]:
                pd.DataFrame({"numerai_ticker": ["A"], "date": [date], "feature_x": [value]}).to_parquet(
                    directory / f"{split}.parquet", index=False,
                )
            downloads = []

            class FakeAPI:
                def download_dataset(self, name, dest_path):
                    downloads.append(name)  # fixtures already provide the downloaded bytes

            output = Path(tmp) / "features.parquet"
            with patch.dict(sys.modules, {"numerapi": types.SimpleNamespace(NumerAPI=FakeAPI)}), patch.object(sys, "argv", [
                "time_series.py", "--data-version", "v6.0", "--data-dir", tmp, "--output", str(output),
            ]):
                main()
            result = pd.read_parquet(output)
            np.testing.assert_allclose(result.feature_x_rank_52w, [1, 1, 0.5])
            np.testing.assert_allclose(result.feature_x_weeks_since_high_52w, [0, 0, 1])
            self.assertEqual(downloads, ["v6.0/live.parquet"])
            self.assertEqual(len(result.columns), 6)
            live = pd.read_parquet(directory / "live.parquet")
            live["date"] = pd.to_datetime(live["date"])
            joined = live.merge(result, on=["numerai_ticker", "date"], validate="one_to_one")
            self.assertEqual(len(joined), 1)


if __name__ == "__main__":
    unittest.main()
