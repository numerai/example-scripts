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
        "params": {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 5,
            "colsample_bytree": 0.1,
            "min_child_weight": 20,
            "tree_method": "hist",
            "n_jobs": 8,
            "random_state": 1337,
        },
        "type": "XGBRegressor",
    },
    "output": {"results_name": "xgb_scalar_ender60"},
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
