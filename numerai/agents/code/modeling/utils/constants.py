from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
NUMERAI_DIR = BASE_DIR.parent
REPO_DIR = NUMERAI_DIR.parent
DEFAULT_CONFIG_PATH = (
    BASE_DIR / "baselines" / "configs" / "small_lgbm_ender60_baseline.py"
)
DEFAULT_OUTPUT_DIR = BASE_DIR
DEFAULT_BASELINES_DIR = BASE_DIR / "baselines"
DEFAULT_BENCHMARK_MODEL = "v53_lgbm_ender60"
DEFAULT_TARGET_COL = "target_ender_60"
DEFAULT_EMBARGO_ERAS = 16
