CONFIG = {
    "data": {
        "data_version": "v5.2",
        "embargo_eras": 13,
        "era_col": "era",
        "feature_set": "all",
        "target_col": "target",
        "id_col": "id",
        "full_data_path": "v5.2/downsampled_full.parquet",
        "benchmark_data_path": "v5.2/downsampled_full_benchmark_models.parquet",
    },
    "model": {
        "params": {
            "colsample_bytree": 0.1,
            "learning_rate": 0.001,
            "max_depth": 10,
            "min_data_in_leaf": 10000,
            "n_estimators": 30000,
            "n_jobs": 30,
            "num_leaves": 1024,
            "random_state": 1337,
        },
        "type": "LGBMRegressor",
    },
    "output": {
        "output_dir": "agents/baselines",
        "results_name": "deep_lgbm_ender20_baseline_downsampled",
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
