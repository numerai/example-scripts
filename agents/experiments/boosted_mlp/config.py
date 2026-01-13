CONFIG = {
    "data": {
        "data_version": "v5.2",
        "embargo_eras": 13,
        "era_col": "era",
        "feature_set": "small",
        "target_col": "target",
        "id_col": "id",
        "full_data_path": "v5.2/downsampled_full.parquet",
        "benchmark_data_path": "v5.2/downsampled_full_benchmark_models.parquet",
        "small_bmc_baseline": "small_lgbm_ender20_baseline",
        "small_bmc_baseline_path": "agents/baselines/predictions/small_lgbm_ender20_baseline.parquet",
    },
    "model": {
        "type": "TorchMLPRegressor",
        "params": {
            "hidden_layer_sizes": (512, 256, 128),
            "activation": "gelu",
            "dropout": 0.1,
            "batch_size": 4096,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "max_epochs": 20,
            "early_stopping": True,
            "patience": 3,
            "val_fraction": 0.1,
        },
    },
    "output": {
        "output_dir": "agents/experiments/boosted_mlp",
        "results_name": "boosted_torch_mlp_baseline_residual",
    },
    "preprocessing": {
        "missing_value": 2.0,
        "nan_missing_all_twos": False,
    },
    "training": {
        "target_transform": {"type": "baseline_residual"},
        "prediction_transform": {"type": "add_baseline"},
        "cv": {
            "embargo": 13,
            "enabled": True,
            "min_train_size": 0,
            "mode": "expanding",
            "n_splits": 5,
        },
    },
}
