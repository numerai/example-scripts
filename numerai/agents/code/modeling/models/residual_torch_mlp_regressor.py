from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


class ResidualTorchMLPRegressor:
    """Train a torch MLP on centered residuals: target - scaled(benchmark)."""

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        *,
        benchmark_model: str | None = None,
        benchmark_col: str | None = None,
        benchmark_mean: float = 0.5000751057731074,
        benchmark_std: float = 0.007366944227953757,
        benchmark_clip: float = 1e-4,
        center_residual: bool = True,
        use_feature_cols: bool = True,
        extra_feature_cols: Iterable[str] | None = None,
        drop_benchmark_from_features: bool = True,
        hidden_layer_sizes: Sequence[int] = (256, 128),
        activation: str = "relu",
        dropout: float = 0.0,
        batch_size: int = 2048,
        prediction_batch_size: int | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        gradient_clip_norm: float | None = None,
        max_epochs: int = 50,
        patience: int = 5,
        val_fraction: float = 0.1,
        val_split: str = "row",
        era_col: str = "era",
        random_state: int = 1337,
        device: str | None = None,
        num_workers: int = 0,
        scale_features: bool = True,
        impute_strategy: str | None = "median",
    ):
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError(
                "torch is required for ResidualTorchMLPRegressor. "
                "Install with `.venv/bin/pip install torch`."
            ) from exc

        self._torch = torch
        self._nn = nn
        self._feature_cols = feature_cols
        self._benchmark_model = benchmark_model
        self._benchmark_col = benchmark_col
        self._benchmark_mean = float(benchmark_mean)
        self._benchmark_std = float(benchmark_std)
        self._benchmark_clip = float(benchmark_clip)
        self._center_residual = bool(center_residual)
        self._use_feature_cols = use_feature_cols
        self._extra_feature_cols = list(extra_feature_cols) if extra_feature_cols else []
        self._drop_benchmark_from_features = drop_benchmark_from_features
        self._hidden_layer_sizes = tuple(int(size) for size in hidden_layer_sizes)
        self._activation = activation
        self._dropout = float(dropout)
        self._train_batch_size = int(batch_size)
        self._pred_batch_size = int(prediction_batch_size or batch_size)
        self._learning_rate = float(learning_rate)
        self._weight_decay = float(weight_decay)
        self._gradient_clip_norm = (
            float(gradient_clip_norm) if gradient_clip_norm is not None else None
        )
        self._max_epochs = int(max_epochs)
        self._patience = int(patience)
        self._val_fraction = float(val_fraction)
        self._val_split = str(val_split).lower()
        if self._val_split not in {"row", "era"}:
            raise ValueError(
                "val_split must be one of: 'row', 'era'. "
                f"Got '{self._val_split}'."
            )
        self._era_col = str(era_col)
        self._random_state = int(random_state)
        self._device = self._resolve_device(device)
        self._num_workers = int(num_workers)
        self._scale_features = bool(scale_features)
        self._impute_strategy = impute_strategy

        self._net = None
        self._input_cols: list[str] | None = None
        self._impute_values: np.ndarray | None = None
        self._scale_mean: np.ndarray | None = None
        self._scale_std: np.ndarray | None = None
        self._residual_mean: float = 0.0

    def fit(self, X, y, **kwargs):
        del kwargs
        benchmark_pred, benchmark_col = self._extract_benchmark_pred(X)
        self._benchmark_col = benchmark_col
        era_values = self._extract_era_values(X) if self._val_split == "era" else None
        y_values = self._as_array(y)
        benchmark_scaled = self._scale_benchmark(benchmark_pred)

        valid_mask = np.isfinite(y_values) & np.isfinite(benchmark_scaled)
        if not np.any(valid_mask):
            raise ValueError(
                "ResidualTorchMLPRegressor has no finite target/benchmark pairs."
            )
        residual = y_values - benchmark_scaled
        if self._center_residual:
            self._residual_mean = float(np.mean(residual[valid_mask]))
            if np.isfinite(self._residual_mean):
                residual = residual - self._residual_mean

        X_features, input_cols = self._select_features(X, benchmark_col)
        mask = np.isfinite(residual)
        if mask.sum() == 0:
            raise ValueError("ResidualTorchMLPRegressor has no finite residual targets.")
        if mask.sum() < len(residual):
            X_features = self._subset_rows(X_features, mask)
            residual = residual[mask]
            if era_values is not None:
                era_values = era_values[mask]

        self._input_cols = input_cols
        X_values = self._prepare_X(X_features, fit=True)
        y_values = residual.astype(np.float32)

        torch = self._torch
        torch.manual_seed(self._random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._random_state)

        X_train, y_train, X_val, y_val = self._train_val_split(
            X_values, y_values, era_values=era_values
        )
        train_loader = self._build_loader(X_train, y_train, shuffle=True)
        val_loader = (
            self._build_loader(X_val, y_val, shuffle=False)
            if X_val is not None
            else None
        )

        self._net = self._build_network(X_train.shape[1]).to(self._device)
        optimizer = torch.optim.Adam(
            self._net.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )
        loss_fn = torch.nn.MSELoss()

        best_state = None
        best_loss = float("inf")
        epochs_no_improve = 0

        for _ in range(self._max_epochs):
            self._train_epoch(train_loader, loss_fn, optimizer)
            if val_loader is None:
                continue
            val_loss = self._eval_loss(val_loader, loss_fn)
            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self._net.state_dict().items()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self._patience:
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        return self

    def predict(self, X):
        _, benchmark_col = self._extract_benchmark_pred(X)
        X_features = self._select_features_for_predict(X, benchmark_col)
        X_values = self._prepare_X(X_features, fit=False)

        preds = self._predict_batches(X_values)
        return preds

    def _predict_batches(self, X_values: np.ndarray) -> np.ndarray:
        torch = self._torch
        self._net.eval()
        preds = []
        with torch.no_grad():
            loader = self._build_loader(X_values, None, shuffle=False)
            for X_batch, _ in loader:
                X_batch = X_batch.to(self._device)
                batch_preds = self._net(X_batch).squeeze(-1)
                preds.append(batch_preds.detach().cpu().numpy())
        return np.concatenate(preds, axis=0)

    def _train_epoch(self, loader, loss_fn, optimizer) -> None:
        self._net.train()
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self._device)
            y_batch = y_batch.to(self._device)
            optimizer.zero_grad(set_to_none=True)
            preds = self._net(X_batch).squeeze(-1)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            if self._gradient_clip_norm:
                self._torch.nn.utils.clip_grad_norm_(
                    self._net.parameters(), self._gradient_clip_norm
                )
            optimizer.step()

    def _eval_loss(self, loader, loss_fn) -> float:
        self._net.eval()
        total_loss = 0.0
        total_count = 0
        with self._torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)
                preds = self._net(X_batch).squeeze(-1)
                loss = loss_fn(preds, y_batch)
                batch_size = y_batch.shape[0]
                total_loss += float(loss.item()) * batch_size
                total_count += batch_size
        return total_loss / max(total_count, 1)

    def _build_loader(self, X, y, *, shuffle: bool):
        torch = self._torch
        X_tensor = torch.tensor(X, dtype=torch.float32)
        if y is None:
            y_tensor = torch.zeros(X_tensor.shape[0], dtype=torch.float32)
        else:
            y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        batch_size = self._train_batch_size if shuffle else self._pred_batch_size
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self._num_workers,
        )

    def _build_network(self, input_dim: int):
        activation_layer = self._activation_layer()
        layers = []
        in_dim = input_dim
        for size in self._hidden_layer_sizes:
            layers.append(self._nn.Linear(in_dim, size))
            layers.append(activation_layer())
            if self._dropout > 0:
                layers.append(self._nn.Dropout(self._dropout))
            in_dim = size
        layers.append(self._nn.Linear(in_dim, 1))
        return self._nn.Sequential(*layers)

    def _activation_layer(self):
        name = self._activation.lower()
        if name == "relu":
            return self._nn.ReLU
        if name == "gelu":
            return self._nn.GELU
        if name == "tanh":
            return self._nn.Tanh
        if name == "elu":
            return self._nn.ELU
        raise ValueError(f"Unsupported activation: {self._activation}")

    def _train_val_split(
        self, X: np.ndarray, y: np.ndarray, *, era_values: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        if self._val_fraction <= 0 or len(y) < 2:
            return X, y, None, None
        rng = np.random.default_rng(self._random_state)
        if era_values is not None and self._val_split == "era":
            unique_eras = np.unique(era_values)
            if len(unique_eras) < 2:
                era_values = None
            else:
                rng.shuffle(unique_eras)
                val_era_count = max(1, int(len(unique_eras) * self._val_fraction))
                val_eras = set(unique_eras[:val_era_count])
                val_mask = np.array([era in val_eras for era in era_values], dtype=bool)
                if 0 < val_mask.sum() < len(y):
                    train_idx = np.where(~val_mask)[0]
                    val_idx = np.where(val_mask)[0]
                    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]
                era_values = None

        indices = np.arange(len(y))
        rng.shuffle(indices)
        val_size = max(1, int(len(y) * self._val_fraction))
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]
        return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    def _prepare_X(self, X, *, fit: bool) -> np.ndarray:
        if hasattr(X, "columns"):
            X = self._coerce_numeric(X)
            X_values = X.to_numpy(dtype=np.float32)
        else:
            X_values = np.asarray(X, dtype=np.float32)

        if self._impute_strategy:
            if fit or self._impute_values is None:
                self._impute_values = self._compute_impute_values(X_values)
            X_values = self._apply_impute(X_values, self._impute_values)

        if self._scale_features:
            if fit or self._scale_mean is None or self._scale_std is None:
                self._scale_mean = np.mean(X_values, axis=0)
                self._scale_std = np.std(X_values, axis=0)
                self._scale_std = np.where(self._scale_std == 0, 1.0, self._scale_std)
            X_values = (X_values - self._scale_mean) / self._scale_std

        return X_values

    def _compute_impute_values(self, X_values: np.ndarray) -> np.ndarray:
        values = X_values.copy()
        values[~np.isfinite(values)] = np.nan
        if self._impute_strategy == "median":
            impute = np.nanmedian(values, axis=0)
        elif self._impute_strategy == "mean":
            impute = np.nanmean(values, axis=0)
        else:
            raise ValueError(f"Unsupported impute_strategy: {self._impute_strategy}")
        impute = np.where(np.isfinite(impute), impute, 0.0).astype(np.float32)
        return impute

    @staticmethod
    def _apply_impute(X_values: np.ndarray, impute: np.ndarray) -> np.ndarray:
        mask = ~np.isfinite(X_values)
        if mask.any():
            X_values = X_values.copy()
            X_values[mask] = np.take(impute, np.where(mask)[1])
        return X_values

    def _extract_benchmark_pred(self, X) -> tuple[np.ndarray, str]:
        if not hasattr(X, "columns"):
            raise ValueError(
                "ResidualTorchMLPRegressor expects a pandas DataFrame with benchmark columns."
            )
        benchmark_col = self._resolve_benchmark_col(X.columns)
        benchmark_pred = self._as_array(X[benchmark_col])
        return benchmark_pred, benchmark_col

    def _extract_era_values(self, X) -> np.ndarray | None:
        if not hasattr(X, "columns"):
            return None
        if self._era_col not in X.columns:
            raise ValueError(
                f"Era column '{self._era_col}' not found in inputs for era-wise split."
            )
        return np.asarray(X[self._era_col])

    def _scale_benchmark(self, benchmark_pred: np.ndarray) -> np.ndarray:
        benchmark_pred = np.asarray(benchmark_pred, dtype=float)
        scaled = np.full_like(benchmark_pred, np.nan, dtype=float)
        finite_mask = np.isfinite(benchmark_pred)
        if not np.any(finite_mask):
            return scaled
        clipped = np.clip(
            benchmark_pred[finite_mask],
            self._benchmark_clip,
            1.0 - self._benchmark_clip,
        )
        try:
            from scipy.special import ndtri
        except ImportError as exc:
            raise ImportError(
                "scipy is required to inverse-normalize benchmark predictions. "
                "Install with `.venv/bin/pip install scipy`."
            ) from exc
        z_scores = ndtri(clipped)
        scaled[finite_mask] = self._benchmark_mean + self._benchmark_std * z_scores
        return scaled

    def _resolve_benchmark_col(self, columns: Iterable[str]) -> str:
        if self._benchmark_col:
            if self._benchmark_col not in columns:
                raise ValueError(
                    f"Benchmark column '{self._benchmark_col}' not found in inputs."
                )
            return self._benchmark_col
        if not self._benchmark_model:
            raise ValueError(
                "benchmark_model or benchmark_col must be set for ResidualTorchMLPRegressor."
            )
        if self._benchmark_model in columns:
            return self._benchmark_model
        suffix = f"_{self._benchmark_model}"
        matches = [col for col in columns if col.endswith(suffix)]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(
                f"Benchmark model '{self._benchmark_model}' not found in columns."
            )
        raise ValueError(
            f"Benchmark model '{self._benchmark_model}' matched multiple columns: {matches}"
        )

    def _select_features(self, X, benchmark_col: str) -> tuple[pd.DataFrame, list[str]]:
        if not hasattr(X, "columns"):
            raise ValueError("ResidualTorchMLPRegressor requires pandas DataFrame inputs.")

        if self._use_feature_cols and self._feature_cols:
            cols = list(self._feature_cols)
            for col in self._extra_feature_cols:
                if col not in cols:
                    cols.append(col)
        else:
            cols = list(X.columns)

        if self._drop_benchmark_from_features and benchmark_col in cols:
            cols.remove(benchmark_col)

        missing = [col for col in cols if col not in X.columns]
        if missing:
            raise ValueError(
                f"Missing feature columns for ResidualTorchMLPRegressor: {missing[:5]}"
                + ("..." if len(missing) > 5 else "")
            )
        if not cols:
            raise ValueError(
                "ResidualTorchMLPRegressor has no input features after filtering."
            )
        return X[cols], cols

    def _select_features_for_predict(self, X, benchmark_col: str) -> pd.DataFrame:
        if self._input_cols is None:
            X_features, _ = self._select_features(X, benchmark_col)
            return X_features
        if not hasattr(X, "columns"):
            raise ValueError("ResidualTorchMLPRegressor requires pandas DataFrame inputs.")
        missing = [col for col in self._input_cols if col not in X.columns]
        if missing:
            raise ValueError(
                f"Missing feature columns for ResidualTorchMLPRegressor: {missing[:5]}"
                + ("..." if len(missing) > 5 else "")
            )
        return X[self._input_cols]

    @staticmethod
    def _coerce_numeric(X: pd.DataFrame) -> pd.DataFrame:
        non_numeric = [
            col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])
        ]
        if not non_numeric:
            return X
        X = X.copy()
        for col in non_numeric:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        return X

    @staticmethod
    def _subset_rows(X, mask: np.ndarray):
        if hasattr(X, "iloc"):
            return X.iloc[mask]
        return X[mask]

    @staticmethod
    def _as_array(values) -> np.ndarray:
        if hasattr(values, "to_numpy"):
            return values.to_numpy(dtype=float)
        return np.asarray(values, dtype=float)

    def _resolve_device(self, device: str | None):
        if device:
            return self._torch.device(device)
        if self._torch.backends.mps.is_available():
            return self._torch.device("mps")
        if self._torch.cuda.is_available():
            return self._torch.device("cuda")
        return self._torch.device("cpu")

    def __getattr__(self, name: str):
        return getattr(self._net, name)
