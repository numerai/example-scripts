CONFIG = {
    "data": {
        "data_version": "v5.2",
        "embargo_eras": 13,
        "era_col": "era",
        "id_col": "id",
        "feature_set": "medium",
        "target_col": "target",
        "benchmark_model": "v52_lgbm_ender20",
        "full_data_path": "v5.2/downsampled_full.parquet",
        "benchmark_data_path": "v5.2/downsampled_full_benchmark_models.parquet",
    },
    "model": {
        "type": "LGBMRegressor",
        "x_groups": ["features", "era", "benchmark_models"],
        "target_transform": {
            "type": "residual_to_benchmark",
            "benchmark_col": "v52_lgbm_ender20",
            "era_col": "era",
            "per_era": True,
            "fit_intercept": True,
            "proportion": 1.0,
        },
        "params": {
            "colsample_bytree": 0.1,
            "device_type": "gpu",
            "learning_rate": 0.01,
            "max_depth": 5,
            "min_data_in_leaf": 10000,
            "n_estimators": 2000,
            "n_jobs": 30,
            "num_leaves": 31,
            "random_state": 1337,
        },
    },
    "output": {
        "output_dir": "experiments/ender20_residual_target_lgbm",
        "results_name": "lgbm_ender20_residual_benchmark_scout_downsampled",
    },
    "preprocessing": {"missing_value": 2.0, "nan_missing_all_twos": False},
    "training": {
        "cv": {
            "embargo": 13,
            "enabled": True,
            "min_train_size": 0,
            "mode": "expanding",
            "n_splits": 5,
        }
    },
}

