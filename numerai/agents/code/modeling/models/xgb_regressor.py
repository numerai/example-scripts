from __future__ import annotations

import numpy as np


class XGBRegressor:
    """XGBoost wrapper with optional vector-leaf multi-output regression."""

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        output_targets: list[str] | None = None,
        predict_target: str | None = None,
        **params,
    ):
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for XGBRegressor. Install with `.venv/bin/pip install xgboost`."
            ) from exc
        self._xgb = xgb
        self._params = dict(params)
        self._feature_cols = feature_cols
        self._output_targets = list(output_targets) if output_targets else None
        self._predict_target = predict_target
        if self._predict_target is None and self._output_targets:
            self._predict_target = self._output_targets[0]
        self._model = xgb.XGBRegressor(**params)

    def fit(self, X, y, **kwargs):
        X = self._filter_features(X, self._feature_cols)
        y_fit = self._prepare_y(X, y)
        self._model.fit(X, y_fit, **kwargs)
        return self

    def predict(self, X):
        X = self._filter_features(X, self._feature_cols)
        preds = np.asarray(self._model.predict(X))
        if preds.ndim == 2:
            idx = self._predict_index()
            return preds[:, idx]
        return preds.ravel()

    def _prepare_y(self, X, y):
        if self._output_targets:
            if hasattr(y, "columns"):
                missing = [col for col in self._output_targets if col not in y.columns]
                if missing:
                    raise ValueError(f"Missing output target columns in y: {missing}")
                return y[self._output_targets]
            if hasattr(X, "columns"):
                missing = [col for col in self._output_targets if col not in X.columns]
                if not missing:
                    return X[self._output_targets]
            if len(self._output_targets) == 1:
                return y
            raise ValueError(
                "XGBRegressor.output_targets requires a DataFrame y or those columns in X."
            )
        if hasattr(y, "columns"):
            cols = list(y.columns)
            self._output_targets = cols
            if self._predict_target is None:
                self._predict_target = cols[0]
            return y
        return y

    def _predict_index(self) -> int:
        if not self._output_targets:
            return 0
        target = self._predict_target or self._output_targets[0]
        if target in self._output_targets:
            return self._output_targets.index(target)
        return 0

    @staticmethod
    def _filter_features(X, feature_cols):
        if not feature_cols or not hasattr(X, "columns"):
            return X
        missing = [col for col in feature_cols if col not in X.columns]
        if missing:
            raise ValueError(
                f"Missing feature columns for XGBRegressor: {missing[:5]}"
                + ("..." if len(missing) > 5 else "")
            )
        return X[feature_cols]

    def __getattr__(self, name: str):
        return getattr(self._model, name)
