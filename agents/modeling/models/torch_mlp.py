from __future__ import annotations

import numpy as np


def _resolve_torch_activation(name: str, torch_module):
    mapping = {
        "relu": torch_module.nn.ReLU,
        "gelu": torch_module.nn.GELU,
        "silu": torch_module.nn.SiLU,
        "tanh": torch_module.nn.Tanh,
        "leaky_relu": torch_module.nn.LeakyReLU,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported activation '{name}'.")
    return mapping[key]


class TorchMLPRegressor:
    def __init__(
        self,
        hidden_layer_sizes=(256, 128),
        activation="relu",
        dropout=0.0,
        batch_size=4096,
        learning_rate=1e-3,
        weight_decay=0.0,
        max_epochs=20,
        device="cuda",
        seed=1337,
        early_stopping=True,
        patience=3,
        val_fraction=0.1,
        amp=True,
        num_workers=0,
        pin_memory=None,
        max_grad_norm=None,
        feature_center=None,
        target_center=None,
        lr_schedule=None,
        lr_schedule_params=None,
    ):
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is required for TorchMLPRegressor. Install a CUDA-enabled build to use GPU."
            ) from exc
        self.torch = torch
        self.device = str(device)
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but not available. Install CUDA-enabled torch or set device='cpu'."
            )
        if isinstance(hidden_layer_sizes, int):
            hidden_layer_sizes = (hidden_layer_sizes,)
        self.hidden_layer_sizes = tuple(int(x) for x in hidden_layer_sizes)
        self.activation = activation
        self.dropout = float(dropout)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.max_epochs = int(max_epochs)
        self.seed = seed
        self.early_stopping = bool(early_stopping)
        self.patience = int(patience)
        self.val_fraction = float(val_fraction)
        self.amp = bool(amp)
        self.num_workers = int(num_workers)
        self.pin_memory = pin_memory
        self.max_grad_norm = (
            float(max_grad_norm) if max_grad_norm is not None else None
        )
        self.feature_center = (
            float(feature_center) if feature_center is not None else None
        )
        self.target_center = (
            float(target_center) if target_center is not None else None
        )
        self.lr_schedule = lr_schedule
        self.lr_schedule_params = lr_schedule_params or {}
        self._device = torch.device(self.device)
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        if self._device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.model = None

    def _build_model(self, input_dim: int):
        torch = self.torch
        activation_cls = _resolve_torch_activation(self.activation, torch)
        layers = []
        in_dim = int(input_dim)
        for hidden in self.hidden_layer_sizes:
            layers.append(torch.nn.Linear(in_dim, hidden))
            layers.append(activation_cls())
            if self.dropout > 0:
                layers.append(torch.nn.Dropout(self.dropout))
            in_dim = hidden
        layers.append(torch.nn.Linear(in_dim, 1))
        self.model = torch.nn.Sequential(*layers).to(self._device)

    def _make_loader(self, X, y=None, weights=None, shuffle=False):
        torch = self.torch
        X_tensor = torch.from_numpy(X).float()
        if y is None:
            dataset = torch.utils.data.TensorDataset(X_tensor)
        else:
            y_tensor = torch.from_numpy(y).float().view(-1, 1)
            if weights is None:
                dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
            else:
                w_tensor = torch.from_numpy(weights).float().view(-1, 1)
                dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor, w_tensor)
        if self.pin_memory is None:
            pin_memory = self._device.type == "cuda"
        else:
            pin_memory = bool(self.pin_memory)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
        )

    def _weighted_mse_sum(self, preds, targets, weights):
        torch = self.torch
        loss = (preds - targets) ** 2
        if weights is None:
            return loss.sum(), torch.tensor(loss.numel(), device=loss.device)
        loss = loss * weights
        return loss.sum(), weights.sum().clamp_min(1e-8)

    def fit(self, X, y, sample_weight=None):
        torch = self.torch
        X_np = np.asarray(X, dtype="float32")
        if self.feature_center is not None:
            X_np = X_np - self.feature_center
        y_np = np.asarray(y, dtype="float32")
        if self.target_center is not None:
            y_np = y_np - self.target_center
        weights_np = None
        if sample_weight is not None:
            weights_np = np.asarray(sample_weight, dtype="float32")
        if self.model is None:
            self._build_model(X_np.shape[1])

        if self.early_stopping and self.val_fraction > 0:
            rng = np.random.default_rng(self.seed)
            indices = rng.permutation(X_np.shape[0])
            val_size = int(X_np.shape[0] * self.val_fraction)
            if val_size < 1:
                train_idx, val_idx = indices, None
            else:
                val_idx = indices[:val_size]
                train_idx = indices[val_size:]
        else:
            train_idx, val_idx = None, None

        if train_idx is None:
            X_train, y_train, w_train = X_np, y_np, weights_np
        else:
            X_train = X_np[train_idx]
            y_train = y_np[train_idx]
            w_train = weights_np[train_idx] if weights_np is not None else None

        train_loader = self._make_loader(
            X_train, y_train, w_train, shuffle=True
        )

        val_loader = None
        if val_idx is not None:
            X_val = X_np[val_idx]
            y_val = y_np[val_idx]
            w_val = weights_np[val_idx] if weights_np is not None else None
            val_loader = self._make_loader(X_val, y_val, w_val, shuffle=False)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        schedule_type = (
            str(self.lr_schedule).lower() if self.lr_schedule is not None else "none"
        )
        scheduler = None
        if schedule_type not in {"none", "", "null"}:
            params = dict(self.lr_schedule_params)
            if schedule_type == "cosine":
                t_max = int(params.get("t_max", self.max_epochs))
                eta_min = float(params.get("eta_min", 0.0))
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=t_max, eta_min=eta_min
                )
            elif schedule_type == "step":
                step_size = int(params.get("step_size", 10))
                gamma = float(params.get("gamma", 0.1))
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=step_size, gamma=gamma
                )
            elif schedule_type == "plateau":
                factor = float(params.get("factor", 0.5))
                patience = int(params.get("patience", 2))
                min_lr = float(params.get("min_lr", 0.0))
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=factor,
                    patience=patience,
                    min_lr=min_lr,
                )
            elif schedule_type == "cosine_warm_restarts":
                t0 = int(params.get("t0", 10))
                t_mult = int(params.get("t_mult", 2))
                eta_min = float(params.get("eta_min", 0.0))
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=t0, T_mult=t_mult, eta_min=eta_min
                )
            elif schedule_type == "one_cycle":
                max_lr = float(params.get("max_lr", self.learning_rate))
                pct_start = float(params.get("pct_start", 0.3))
                anneal_strategy = params.get("anneal_strategy", "cos")
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=max_lr,
                    epochs=self.max_epochs,
                    steps_per_epoch=len(train_loader),
                    pct_start=pct_start,
                    anneal_strategy=anneal_strategy,
                )
            else:
                raise ValueError(f"Unsupported lr_schedule '{self.lr_schedule}'.")
        scaler = torch.cuda.amp.GradScaler(
            enabled=self.amp and self._device.type == "cuda"
        )

        best_val = float("inf")
        epochs_without_improve = 0

        for _ in range(self.max_epochs):
            self.model.train()
            last_loss = None
            for batch in train_loader:
                if len(batch) == 3:
                    batch_x, batch_y, batch_w = batch
                else:
                    batch_x, batch_y = batch
                    batch_w = None
                batch_x = batch_x.to(self._device, non_blocking=True)
                batch_y = batch_y.to(self._device, non_blocking=True)
                if batch_w is not None:
                    batch_w = batch_w.to(self._device, non_blocking=True)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(
                    enabled=self.amp and self._device.type == "cuda"
                ):
                    preds = self.model(batch_x).view(-1, 1)
                    loss_sum, weight_sum = self._weighted_mse_sum(
                        preds, batch_y, batch_w
                    )
                    loss = loss_sum / weight_sum
                last_loss = loss.detach().item()
                scaler.scale(loss).backward()
                if self.max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                scaler.step(optimizer)
                scaler.update()

            if val_loader is None:
                if scheduler is not None:
                    if schedule_type == "plateau":
                        scheduler.step(last_loss if last_loss is not None else 0.0)
                    else:
                        scheduler.step()
                continue

            self.model.eval()
            val_loss_sum = torch.tensor(0.0, device=self._device)
            val_weight_sum = torch.tensor(0.0, device=self._device)
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 3:
                        batch_x, batch_y, batch_w = batch
                    else:
                        batch_x, batch_y = batch
                        batch_w = None
                    batch_x = batch_x.to(self._device, non_blocking=True)
                    batch_y = batch_y.to(self._device, non_blocking=True)
                    if batch_w is not None:
                        batch_w = batch_w.to(self._device, non_blocking=True)
                    preds = self.model(batch_x).view(-1, 1)
                    loss_sum, weight_sum = self._weighted_mse_sum(
                        preds, batch_y, batch_w
                    )
                    val_loss_sum += loss_sum
                    val_weight_sum += weight_sum

            val_loss = (val_loss_sum / val_weight_sum).item()
            if val_loss < best_val:
                best_val = val_loss
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= self.patience:
                    break
            if scheduler is not None:
                if schedule_type == "plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

        return self

    def predict(self, X):
        torch = self.torch
        X_np = np.asarray(X, dtype="float32")
        if self.feature_center is not None:
            X_np = X_np - self.feature_center
        loader = self._make_loader(X_np, shuffle=False)
        preds = []
        self.model.eval()
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self._device, non_blocking=True)
                batch_preds = self.model(batch_x).view(-1).detach().cpu().numpy()
                preds.append(batch_preds)
        if not preds:
            return np.array([], dtype="float32")
        output = np.concatenate(preds)
        if self.target_center is not None:
            output = output + self.target_center
        return output
