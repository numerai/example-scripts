from __future__ import annotations

import lightgbm as lgb

from agents.modeling.models.lgbm_regressor import LGBMRegressor
from agents.modeling.models.torch_mlp import TorchMLPRegressor


def build_model(model_type: str, model_params: dict):
    if model_type == "LGBMRegressor":
        return LGBMRegressor(**model_params)
    if model_type == "LGBMRanker":
        return lgb.LGBMRanker(**model_params)
    if model_type == "TorchMLPRegressor":
        return TorchMLPRegressor(**model_params)
    if model_type == "CatBoostRegressor":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(**model_params)
    if model_type == "MLPRegressor":
        try:
            from sklearn.neural_network import MLPRegressor
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for MLPRegressor. Install with `.venv/bin/pip install scikit-learn`."
            ) from exc

        return MLPRegressor(**model_params)
    if model_type in {"TabPFNRegressor", "TabPFNClassifier"}:
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
    raise ValueError(f"Unsupported model type: {model_type}")
