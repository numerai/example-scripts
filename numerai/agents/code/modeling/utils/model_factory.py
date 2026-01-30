from __future__ import annotations

def build_model(
    model_type: str,
    model_params: dict,
    model_config: dict | None = None,
    *,
    feature_cols: list[str] | None = None,
):
    model_config = model_config or {}
    if model_type == "LGBMRegressor":
        from agents.code.modeling.models.lgbm_regressor import LGBMRegressor
        model = LGBMRegressor(feature_cols=feature_cols, **model_params)
    else:
        raise ValueError(
            "Unsupported model type: "
            f"{model_type}. Supported types: LGBMRegressor"
        )

    target_transform = model_config.get("target_transform")
    if target_transform:
        from agents.code.modeling.utils.target_transforms import TargetTransformWrapper

        model = TargetTransformWrapper(model, target_transform)
    return model
