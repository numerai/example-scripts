"""Validate the optional reproduction runner after installing its requirements.

    python validate_reproduction.py

This is separate from the lightweight example's test discovery because the
runner requires LightGBM, Numba and the scoring dependencies.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reproduce import feature_frame, record_run, validate_kernel


class ReproductionChecks(unittest.TestCase):
    def test_optimized_transforms_and_bins_match_example(self):
        validate_kernel()

    def test_continuous_values_and_missing_sentinel_stay_distinct(self):
        matrix = np.array([[255, 255, 0], [2, 255, 1], [4, 255, 0], [255, 255, 1]], dtype="uint8", order="F")
        continuous = np.array([255.0, -0.9, np.nan, 3.8186681], dtype="float32")
        frame = feature_frame(matrix, np.arange(4), 3, ["feature_x", "feature_factor", "feature_country"], {1: continuous})
        np.testing.assert_array_equal(frame.feature_factor.to_numpy(), continuous)
        np.testing.assert_allclose(frame.feature_x, [np.nan, 2, 4, np.nan], equal_nan=True)
        self.assertEqual(frame.feature_country.astype(int).tolist(), [0, 1, 0, 1])

    def test_changed_data_or_configuration_cannot_reuse_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            config_path = directory / "input_config.json"
            config = {"sample_seed": 7}
            config_path.write_text(json.dumps(config))
            data = [{"name": "train.parquet", "sha256": "first"}]
            first = record_run(config, config_path, data, None, directory)
            self.assertEqual(first, record_run(config, config_path, data, None, directory))
            with self.assertRaisesRegex(ValueError, "Cache signature differs"):
                record_run(config, config_path, [{"name": "train.parquet", "sha256": "changed"}], None, directory)
            with self.assertRaisesRegex(ValueError, "Cache signature differs"):
                record_run({"sample_seed": 8}, config_path, data, None, directory)

    def test_unsigned_cached_matrix_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            config_path = directory / "input_config.json"
            config_path.write_text("{}")
            (directory / "matrix_manifest.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Unsigned cached matrix"):
                record_run({}, config_path, [], None, directory)


if __name__ == "__main__":
    unittest.main()
