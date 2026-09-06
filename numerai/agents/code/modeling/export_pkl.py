"""Fit a winning config on all labeled rows and export a Numerai predict pickle."""

from __future__ import annotations

import argparse
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
from numerapi import NumerAPI

from agents.code.modeling.utils.config import load_config
from agents.code.modeling.utils.constants import DEFAULT_TARGET_COL, NUMERAI_DIR
from agents.code.modeling.utils.data import (
    attach_benchmark_models,
    drop_null_target_rows,
    load_features,
    load_full_data,
)
from agents.code.modeling.utils.model_data import build_x_cols, normalize_x_groups
from agents.code.modeling.utils.model_factory import build_model
from agents.code.modeling.utils.pipeline import resolve_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a Numerai model pickle.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def unwrap_estimator(model):
    """Return the inner sklearn/LightGBM estimator from local wrappers."""
    while hasattr(model, "_model"):
        model = model._model
    return model


def build_predict_fn(lgb_model, feature_order: list[str]):
    """Self-contained predict() for numerai_predict. No repo imports at serve time.

    Defined via exec so cloudpickle does not bind the function to this module
    (Numerai's container has no ``agents`` package). Raw residual scores are
    per-era percentile-ranked into (0, 1]; ranking is monotonic so CORR/BMC
    are unchanged.
    """
    namespace = {
        "np": np,
        "pd": pd,
        "lgb_model": lgb_model,
        "feature_order": list(feature_order),
    }
    exec(
        """
def predict(live_features, live_benchmark_models=None):
    X = live_features.reindex(columns=feature_order)
    scores = np.asarray(lgb_model.predict(X)).ravel()
    ranked = pd.Series(scores, index=live_features.index)
    if "era" in live_features.columns:
        ranked = ranked.groupby(live_features["era"], sort=False).rank(
            method="average", pct=True
        )
    else:
        ranked = ranked.rank(method="average", pct=True)
    return pd.DataFrame({"prediction": ranked}, index=live_features.index)
""",
        namespace,
    )
    return namespace["predict"]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    data_version = data_config.get("data_version", "v5.3")
    feature_set = data_config.get("feature_set", "medium")
    target_col = data_config.get("target_col", DEFAULT_TARGET_COL)
    era_col = data_config.get("era_col", "era")
    id_col = data_config.get("id_col", "id")
    napi = NumerAPI()
    features = load_features(napi, data_version, feature_set)
    extra_cols = list(data_config.get("extra_cols") or [])
    extra_cols.extend(model_config.get("output_targets") or [])
    full = load_full_data(
        napi,
        data_version,
        features,
        era_col,
        target_col,
        id_col,
        full_data_path=data_config.get("full_data_path"),
        extra_cols=extra_cols,
    )
    x_groups = normalize_x_groups(
        model_config.get("x_groups") or model_config.get("data_needed")
    )
    benchmark_cols: list[str] = []
    if "benchmark_models" in x_groups:
        full, benchmark_cols = attach_benchmark_models(
            full,
            napi,
            data_version,
            data_config.get("benchmark_data_path"),
            era_col,
            id_col,
        )
    drop_cols = [target_col, *list(model_config.get("output_targets") or [])]
    full = drop_null_target_rows(full, drop_cols)
    x_cols = build_x_cols(
        x_groups=x_groups,
        features=features,
        benchmark_cols=benchmark_cols,
        era_col=era_col,
        id_col=id_col,
    )
    model_type, model_params = resolve_model_config(model_config)
    model = build_model(
        model_type, model_params, model_config, feature_cols=features
    )
    y = (
        full[list(model_config["output_targets"])]
        if model_config.get("output_targets")
        else full[target_col]
    )
    model.fit(full[x_cols], y)

    predict = build_predict_fn(unwrap_estimator(model), list(features))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(cloudpickle.dumps(predict))
    print(f"Wrote pickle to {args.output}")
    print(f"Trained on {len(full)} rows from {NUMERAI_DIR / data_version}")


if __name__ == "__main__":
    main()
