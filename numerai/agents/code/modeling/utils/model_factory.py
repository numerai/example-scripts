from __future__ import annotations

from agents.code.modeling.models.lgbm_regressor import LGBMRegressor


def build_model(
    model_type: str,
    model_params: dict,
    model_config: dict | None = None,
    *,
    feature_cols: list[str] | None = None,
):
    if model_type == "LGBMRegressor":
        return LGBMRegressor(feature_cols=feature_cols, **model_params)
    elif model_type == "CatBoostRegressor":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(**model_params)
    elif model_type in {"TabPFNRegressor", "TabPFNClassifier"}:
        try:
            import tabpfn
        except ImportError as exc:
            raise ImportError(
                "tabpfn is required for TabPFN models. Install with `.venv/bin/pip install tabpfn`."
            ) from exc
        model_class = getattr(tabpfn, model_type, None)
        if model_class is None:
            raise ValueError(
                f"tabpfn does not expose {model_type}; upgrade tabpfn or choose a supported model."
            )
        return model_class(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
