from __future__ import annotations

def build_model(
    model_type: str,
    model_params: dict,
    model_config: dict | None = None,
    *,
    feature_cols: list[str] | None = None,
):
    if model_type == "LGBMRegressor":
        from agents.code.modeling.models.lgbm_regressor import LGBMRegressor

        return LGBMRegressor(feature_cols=feature_cols, **model_params)
    if model_type == "ResidualMLPRegressor":
        from agents.code.modeling.models.residual_mlp_regressor import (
            ResidualMLPRegressor,
        )

        return ResidualMLPRegressor(feature_cols=feature_cols, **model_params)
    if model_type == "ResidualTorchMLPRegressor":
        from agents.code.modeling.models.residual_torch_mlp_regressor import (
            ResidualTorchMLPRegressor,
        )

        return ResidualTorchMLPRegressor(feature_cols=feature_cols, **model_params)
    raise ValueError(
        "Unsupported model type: "
        f"{model_type}. Supported types: LGBMRegressor, ResidualMLPRegressor, "
        "ResidualTorchMLPRegressor."
    )
