from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from agents.code.modeling.export_pkl import build_predict_fn
from agents.code.modeling.utils.data import drop_null_target_rows
from agents.code.modeling.utils.ensemble import blend_rank_gauss
from agents.code.modeling.utils.model_factory import build_model
from agents.code.modeling.utils.numerai_cv import era_cv_splits


class TestEnder60Eval(unittest.TestCase):
    def test_drop_null_target_rows(self) -> None:
        df = pd.DataFrame(
            {
                "target_ender_60": [0.5, np.nan, 0.25],
                "era": ["0001", "0002", "0003"],
            }
        )
        cleaned = drop_null_target_rows(df, ["target_ender_60"])
        self.assertEqual(list(cleaned["era"]), ["0001", "0003"])

    def test_era_cv_default_embargo_is_16(self) -> None:
        eras = [f"{i:04d}" for i in range(1, 81)]
        splits = era_cv_splits(eras, n_splits=4, min_train_size=0)
        train_eras, val_eras = splits[1]
        self.assertEqual(val_eras[0], "0021")
        self.assertEqual(train_eras[-1], "0004")

    def test_rank_gauss_blend_is_per_era(self) -> None:
        df = pd.DataFrame(
            {
                "era": ["0001"] * 4 + ["0002"] * 4,
                "a": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
                "b": [1.0, 1.5, 2.5, 8.0, 12.0, 18.0, 33.0, 41.0],
            }
        )
        blended = blend_rank_gauss(df, ["a", "b"], [0.7, 0.3], era_col="era")
        self.assertEqual(len(blended), 8)
        self.assertTrue(np.isfinite(blended.to_numpy()).all())
        for era in ("0001", "0002"):
            part = blended[df["era"] == era]
            self.assertAlmostEqual(float(part.std(ddof=0)), 1.0, places=6)


class TestXGBFactory(unittest.TestCase):
    def test_build_xgb_multioutput_predicts_named_head(self) -> None:
        try:
            import xgboost  # noqa: F401
        except ImportError:
            self.skipTest("xgboost is not installed")

        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(80, 3)), columns=["f1", "f2", "f3"])
        y = pd.DataFrame(
            {
                "target_ender_60": X["f1"] + rng.normal(scale=0.1, size=80),
                "target_teager2b_60": -X["f2"] + rng.normal(scale=0.1, size=80),
            }
        )
        model = build_model(
            "XGBRegressor",
            {
                "n_estimators": 8,
                "max_depth": 2,
                "multi_strategy": "multi_output_tree",
                "tree_method": "hist",
            },
            {
                "output_targets": ["target_ender_60", "target_teager2b_60"],
                "predict_target": "target_ender_60",
            },
            feature_cols=["f1", "f2", "f3"],
        )
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(preds.shape, (80,))
        self.assertGreater(float(np.corrcoef(preds, y["target_ender_60"])[0, 1]), 0.5)


class TestExportPredict(unittest.TestCase):
    def test_predict_ranks_residual_scores_into_unit_interval(self) -> None:
        class ResidualStub:
            def predict(self, X):
                return np.linspace(-0.03, 0.03, len(X))

        live = pd.DataFrame(
            {
                "f1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "era": ["0001"] * 3 + ["0002"] * 3,
            }
        )
        predict = build_predict_fn(ResidualStub(), ["f1"])
        out = predict(live, None)
        values = out["prediction"].to_numpy()
        self.assertEqual(list(out.index), list(live.index))
        self.assertTrue(np.isfinite(values).all())
        self.assertGreaterEqual(float(values.min()), 0.0)
        self.assertLessEqual(float(values.max()), 1.0)
        era1 = out.loc[live["era"] == "0001", "prediction"].to_numpy()
        self.assertGreater(era1[2], era1[0])


if __name__ == "__main__":
    unittest.main()
