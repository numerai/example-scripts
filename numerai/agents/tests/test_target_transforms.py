from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from agents.code.modeling.utils.target_transforms import apply_target_transform


class TestTargetTransforms(unittest.TestCase):
    def test_residual_to_benchmark_is_orthogonal_per_era(self) -> None:
        rng = np.random.default_rng(7)
        n = 2000
        eras = np.array(["0001"] * (n // 2) + ["0002"] * (n - n // 2))
        benchmark = rng.normal(size=n)

        # Different slopes per era + intercept
        y = np.empty(n, dtype="float64")
        y[: n // 2] = 2.0 * benchmark[: n // 2] + 1.0 + rng.normal(scale=0.1, size=n // 2)
        y[n // 2 :] = -3.0 * benchmark[n // 2 :] - 0.5 + rng.normal(
            scale=0.1, size=n - n // 2
        )

        X = pd.DataFrame({"era": eras, "v52_lgbm_ender20": benchmark})
        y = pd.Series(y, name="target")

        transformed = apply_target_transform(
            y,
            X,
            {
                "type": "residual_to_benchmark",
                "benchmark_col": "v52_lgbm_ender20",
                "era_col": "era",
                "per_era": True,
                "fit_intercept": True,
                "proportion": 1.0,
            },
        )

        for era in ("0001", "0002"):
            mask = X["era"].to_numpy() == era
            corr = np.corrcoef(transformed[mask], benchmark[mask])[0, 1]
            self.assertLess(abs(float(corr)), 1e-10)

    def test_subtract_benchmark_zscore_is_per_era_and_scaled(self) -> None:
        eras = np.array(["0001", "0001", "0002", "0002"])
        benchmark = np.array([1.0, 3.0, 10.0, 14.0])
        X = pd.DataFrame({"era": eras, "v52_lgbm_ender20": benchmark})
        y = pd.Series([0.5, 0.5, 0.5, 0.5], name="target")

        transformed = apply_target_transform(
            y,
            X,
            {"type": "subtract_benchmark", "benchmark_col": "v52_lgbm_ender20"},
        )

        expected = pd.Series([0.57, 0.43, 0.57, 0.43], name="target")
        self.assertTrue(np.allclose(transformed.to_numpy(), expected.to_numpy()))

    def test_subtract_benchmark_zscore_zero_std_is_noop(self) -> None:
        eras = np.array(["0001", "0001", "0001", "0001"])
        benchmark = np.array([2.0, 2.0, 2.0, 2.0])
        X = pd.DataFrame({"era": eras, "v52_lgbm_ender20": benchmark})
        y = pd.Series([0.1, 0.2, 0.3, 0.4], name="target")

        transformed = apply_target_transform(
            y,
            X,
            {
                "type": "subtract_benchmark_zscore",
                "benchmark_col": "v52_lgbm_ender20",
                "scale": 0.07,
            },
        )
        self.assertTrue(np.allclose(transformed.to_numpy(), y.to_numpy()))

    def test_proportion_blends_with_original_target(self) -> None:
        rng = np.random.default_rng(1337)
        n = 1000
        eras = np.array(["0001"] * n)
        benchmark = rng.normal(size=n)
        y = 1.5 * benchmark + 0.2 + rng.normal(scale=0.3, size=n)

        X = pd.DataFrame({"era": eras, "v52_lgbm_ender20": benchmark})
        y = pd.Series(y, name="target")

        y_same = apply_target_transform(
            y,
            X,
            {
                "type": "residual_to_benchmark",
                "benchmark_col": "v52_lgbm_ender20",
                "era_col": "era",
                "per_era": True,
                "fit_intercept": True,
                "proportion": 0.0,
            },
        )
        self.assertTrue(np.allclose(y_same.to_numpy(), y.to_numpy()))

        y_resid = apply_target_transform(
            y,
            X,
            {
                "type": "residual_to_benchmark",
                "benchmark_col": "v52_lgbm_ender20",
                "era_col": "era",
                "per_era": True,
                "fit_intercept": True,
                "proportion": 1.0,
            },
        )
        y_half = apply_target_transform(
            y,
            X,
            {
                "type": "residual_to_benchmark",
                "benchmark_col": "v52_lgbm_ender20",
                "era_col": "era",
                "per_era": True,
                "fit_intercept": True,
                "proportion": 0.5,
            },
        )
        expected = 0.5 * y + 0.5 * y_resid
        self.assertTrue(np.allclose(y_half.to_numpy(), expected.to_numpy()))
