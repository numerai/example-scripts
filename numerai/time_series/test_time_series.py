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

from time_series import ages, build_features, latest_version, main, rank_and_bin


class TimeSeriesTests(unittest.TestCase):
    def build(self, values, dates=None, tickers=None):
        return build_features(pd.DataFrame({
            "numerai_ticker": tickers if tickers is not None else ["A"] * len(values),
            "date": dates if dates is not None else pd.date_range("2025-01-01", periods=len(values), freq="7D"),
            "feature_x": values,
        }), ["feature_x"])

    def test_ties_and_changes(self):
        result = self.build([0] * 52 + [1, 1, 0]).iloc[-3:]
        np.testing.assert_allclose(result.feature_x_rank_52w, [1, 103 / 104, 51 / 104])
        np.testing.assert_allclose(result.feature_x_weeks_since_high_52w, [0, 0, 1])
        np.testing.assert_allclose(result.feature_x_weeks_since_change, [0, 1, 0])

    def test_elapsed_time_and_window_boundaries(self):
        # The initial spike expires at exactly 52 weeks. At row 52 the short
        # mean is mean(40..52), and the long mean is mean(1..52).
        result = self.build([100] + list(range(1, 54))).iloc[52:]
        np.testing.assert_allclose(result.feature_x_mean_13w_minus_52w, [19.5, 19.5])
        np.testing.assert_allclose(result.feature_x_rank_52w, [1, 1])
        np.testing.assert_allclose(result.feature_x_weeks_since_high_52w, [0, 0])
        high, _ = ages(np.array([10., 0., 2., 1.]), np.array([0, 91, 364, 371]))
        np.testing.assert_allclose(high, [0, 13, 0, 1])

    def test_short_and_incomplete_histories_stay_missing(self):
        result = self.build(list(range(53)))
        self.assertTrue(result.iloc[:52, 2:5].isna().all().all())
        self.assertTrue(result.iloc[52, 2:5].notna().all())
        incomplete = list(range(54))
        incomplete[10] = np.nan
        self.assertTrue(self.build(incomplete).iloc[52:, 2:5].isna().all().all())
        # Fifty-two daily observations are not a year of weekly history.
        daily = self.build(list(range(53)), pd.date_range("2025-01-01", periods=53))
        self.assertTrue(daily.iloc[:, 2:5].isna().all().all())

    def test_missing_values_break_unchanged_run(self):
        result = self.build([1, 2, np.nan, 2, 2, 1])
        self.assertTrue(result.filter(like="feature_x_").iloc[2].isna().all())
        np.testing.assert_allclose(result.feature_x_weeks_since_change, [np.nan, 0, np.nan, np.nan, np.nan, 0], equal_nan=True)

    def test_future_and_other_tickers_do_not_affect_history(self):
        dates = list(pd.date_range("2025-01-01", periods=55, freq="7D"))
        past = self.build(list(range(55)), dates)
        mixed = self.build(
            [-100] + list(reversed(range(55))) + [99],
            [dates[-1] + pd.Timedelta(weeks=1)] + list(reversed(dates)) + [dates[-1]],
            ["A"] * 56 + ["B"],
        )
        pd.testing.assert_frame_equal(past, mixed[mixed.numerai_ticker.eq("A")].iloc[:55].reset_index(drop=True))

    def test_bins_use_only_the_current_date_and_preserve_ties_and_missing(self):
        data = pd.DataFrame({
            "numerai_ticker": list("ABCDEFGHIJ") * 2,
            "date": ["2025-01-01"] * 10 + ["2025-01-08"] * 10,
            "feature_x_rank_52w": list(range(10)) + list(range(100, 110)),
        })
        first = rank_and_bin(data.iloc[:10].copy())
        binned = rank_and_bin(data)
        pd.testing.assert_frame_equal(first, binned.iloc[:10])
        np.testing.assert_array_equal(first.feature_x_rank_52w, np.repeat(range(5), 2))
        tied = rank_and_bin(pd.DataFrame({
            "numerai_ticker": list("ABCDE"), "date": ["2025-01-01"] * 5,
            "feature_x_rank_52w": [1, 1, 2, 3, np.nan],
        }))
        np.testing.assert_allclose(tied.feature_x_rank_52w, [1, 1, 3, 4, np.nan], equal_nan=True)

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
            dates = pd.date_range("2024-01-03", periods=54, freq="7D").strftime("%Y-%m-%d")
            for split, rows in [
                ("train", list(range(52))), ("validation", [52]), ("live", [53]),
            ]:
                pd.DataFrame([
                    {"numerai_ticker": ticker, "date": dates[i], "feature_x": sign * i}
                    for ticker, sign in [("A", 1), ("B", -1)] for i in rows
                ]).to_parquet(
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
            mature = result[result.date >= pd.Timestamp(dates[52])]
            np.testing.assert_allclose(mature.feature_x_rank_52w, [4, 4, 2, 2])
            np.testing.assert_allclose(mature.feature_x_weeks_since_high_52w, [2, 2, 4, 4])
            self.assertTrue(result.loc[result.date < pd.Timestamp(dates[52]), "feature_x_rank_52w"].isna().all())
            self.assertEqual(downloads, ["v6.0/live.parquet"])
            self.assertEqual(len(result.columns), 6)
            live = pd.read_parquet(directory / "live.parquet")
            live["date"] = pd.to_datetime(live["date"])
            joined = live.merge(result, on=["numerai_ticker", "date"], validate="one_to_one")
            self.assertEqual(len(joined), 2)


if __name__ == "__main__":
    unittest.main()
