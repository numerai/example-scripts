from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = (
    BASE_DIR / "baselines" / "configs" / "small_lgbm_ender20_baseline.py"
)
DEFAULT_OUTPUT_DIR = BASE_DIR
DEFAULT_BASELINES_DIR = BASE_DIR / "baselines"
DEFAULT_BENCHMARK_MODEL = "v52_lgbm_ender20"
DEFAULT_SMALL_BMC_BASELINE = "small_lgbm_ender20_baseline"
DEFAULT_LGBM_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "max_depth": 5,
    "num_leaves": 2**5 - 1,
    "colsample_bytree": 0.1,
    "min_data_in_leaf": 10000,
    "n_jobs": 30,
    "random_state": 1337,
}
