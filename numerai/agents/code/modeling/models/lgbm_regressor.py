from __future__ import annotations


class LGBMRegressor:
    """Minimal wrapper exposing fit/predict for the training pipeline."""

    def __init__(self, feature_cols: list[str] | None = None, **params):
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "lightgbm is required for LGBMRegressor. Install with `.venv/bin/pip install lightgbm`."
            ) from exc
        self._lgb = lgb
        self._params = dict(params)
        self._model = lgb.LGBMRegressor(**params)
        self._feature_cols = feature_cols

    def fit(self, X, y, **kwargs):
        X = self._filter_features(X, self._feature_cols)
        try:
            self._model.fit(X, y, **kwargs)
        except self._lgb.basic.LightGBMError as exc:
            if self._should_fallback_to_cpu(exc):
                print("LightGBM GPU not available; retrying with device_type='cpu'.")
                self._params["device_type"] = "cpu"
                self._model = self._lgb.LGBMRegressor(**self._params)
                self._model.fit(X, y, **kwargs)
            else:
                raise
        return self

    def predict(self, X):
        X = self._filter_features(X, self._feature_cols)
        return self._model.predict(X)

    def _should_fallback_to_cpu(self, exc: Exception) -> bool:
        message = str(exc)
        device_type = str(self._params.get("device_type", "")).lower()
        return (
            device_type == "gpu"
            and "GPU Tree Learner was not enabled" in message
        )

    @staticmethod
    def _filter_features(X, feature_cols):
        if not feature_cols or not hasattr(X, "columns"):
            return X
        missing = [col for col in feature_cols if col not in X.columns]
        if missing:
            raise ValueError(
                f"Missing feature columns for LGBMRegressor: {missing[:5]}"
                + ("..." if len(missing) > 5 else "")
            )
        return X[feature_cols]

    def __getattr__(self, name: str):
        return getattr(self._model, name)
