CONFIG = {
    "data": {
        "data_version": "v5.3",
        "embargo_eras": 16,
        "era_col": "era",
        "id_col": "id",
        "feature_set": "medium",
        "target_col": "target_ender_60",
        "benchmark_model": "v53_lgbm_ender60",
        "full_data_path": "v5.3/downsampled_full.parquet",
        "benchmark_data_path": "v5.3/downsampled_full_benchmark_models.parquet",
    },
    "model": {
        "x_groups": ["features", "era", "benchmark_models"],
        "target_transform": {
            "type": "residual_to_benchmark",
            "benchmark_col": "v53_lgbm_ender60",
            "era_col": "era",
            "per_era": True,
            "fit_intercept": True,
            "proportion": 0.5,
        },
        "params": {
            "colsample_bytree": 0.1,
            "device_type": "cpu",
            "learning_rate": 0.02,
            "max_depth": 5,
            "min_data_in_leaf": 10000,
            "n_estimators": 400,
            "n_jobs": 8,
            "num_leaves": 31,
            "random_state": 1337,
        },
        "type": "LGBMRegressor",
    },
    "output": {"results_name": "residual_prop_050"},
    "preprocessing": {"missing_value": 2.0, "nan_missing_all_twos": False},
    "training": {
        "cv": {
            "embargo": 16,
            "enabled": True,
            "min_train_size": 0,
            "mode": "expanding",
            "n_splits": 5,
        }
    },
}
