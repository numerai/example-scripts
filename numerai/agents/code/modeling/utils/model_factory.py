from __future__ import annotations

from agents.code.modeling.models.lgbm_regressor import LGBMRegressor


def build_model(
    model_type: str,
    model_params: dict,
    model_config: dict | None = None,
    *,
    feature_cols: list[str] | None = None,
):
    if model_type != "LGBMRegressor":
        raise ValueError(
            f"Unsupported model type: {model_type}. Only LGBMRegressor is available."
        )
    return LGBMRegressor(feature_cols=feature_cols, **model_params)
