from __future__ import annotations


class LGBMRegressor:
    """Minimal wrapper exposing fit/predict for the training pipeline."""

    def __init__(self, **params):
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "lightgbm is required for LGBMRegressor. Install with `.venv/bin/pip install lightgbm`."
            ) from exc
        self._model = lgb.LGBMRegressor(**params)

    def fit(self, X, y, **kwargs):
        self._model.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def __getattr__(self, name: str):
        return getattr(self._model, name)
