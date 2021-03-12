#!/usr/bin/env python
"""
Example classifier on Numerai data using a xgboost regression.
To get started, install the required packages: pip install pandas numpy sklearn xgboost
"""

import csv
from pathlib import Path

import pandas as pd
import numpy as np
from xgboost import XGBRegressor

TARGET_NAME = f"target"
PREDICTION_NAME = f"prediction"

MODEL_FILE = Path("example_model.xgb")


# Submissions are scored by spearman correlation
def correlation(predictions, targets):
    ranked_preds = predictions.rank(pct=True, method="first")
    return np.corrcoef(ranked_preds, targets)[0, 1]


# convenience method for scoring
def score(df):
    return correlation(df[PREDICTION_NAME], df[TARGET_NAME])


# Payout is just the score cliped at +/-25%
def payout(scores):
    return scores.clip(lower=-0.25, upper=0.25)


# Read the csv file into a pandas Dataframe as float16 to save space
def read_csv(file_path):
    with open(file_path, 'r') as f:
        column_names = next(csv.reader(f))

    dtypes = {x: np.float16 for x in column_names if x.startswith(('feature', 'target'))}
    df = pd.read_csv(file_path, dtype=dtypes, index_col=0)

    # Memory constrained? Try this instead (slower, but more memory efficient)
    # see https://forum.numer.ai/t/saving-memory-with-uint8-features/254
    # dtypes = {f"target": np.float16}
    # to_uint8 = lambda x: np.uint8(float(x) * 4)
    # converters = {x: to_uint8 for x in column_names if x.startswith('feature')}
    # df = pd.read_csv(file_path, dtype=dtypes, converters=converters)

    return df


def main():
    print("Loading data...")
    # The training data is used to train your model how to predict the targets.
    training_data = read_csv("numerai_training_data.csv")
    # The tournament data is the data that Numerai uses to evaluate your model.
    tournament_data = read_csv("numerai_tournament_data.csv")

    feature_names = [
        f for f in training_data.columns if f.startswith("feature")
    ]
    print(f"Loaded {len(feature_names)} features")

    # This is the model that generates the included example predictions file.
    # Taking too long? Set learning_rate=0.1 and n_estimators=200 to make this run faster.
    # Remember to delete example_model.xgb if you change any of the parameters below.
    model = XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators=2000, n_jobs=-1, colsample_bytree=0.1)
    if MODEL_FILE.is_file():
        print("Loading pre-trained model...")
        model.load_model(MODEL_FILE)
    else:
        print("Training model...")
        model.fit(training_data[feature_names], training_data[TARGET_NAME])
        model.save_model(MODEL_FILE)

    # Generate predictions on both training and tournament data
    print("Generating predictions...")
    training_data[PREDICTION_NAME] = model.predict(training_data[feature_names])
    tournament_data[PREDICTION_NAME] = model.predict(tournament_data[feature_names])

    # Check the per-era correlations on the training set (in sample)
    train_correlations = training_data.groupby("era").apply(score)
    print(f"On training the correlation has mean {train_correlations.mean()} and std {train_correlations.std(ddof=0)}")
    print(f"On training the average per-era payout is {payout(train_correlations).mean()}")

    """Validation Metrics"""
    # Check the per-era correlations on the validation set (out of sample)
    validation_data = tournament_data[tournament_data.data_type == "validation"]
    validation_correlations = validation_data.groupby("era").apply(score)
    print(f"On validation the correlation has mean {validation_correlations.mean()} and "
          f"std {validation_correlations.std(ddof=0)}")
    print(f"On validation the average per-era payout is {payout(validation_correlations).mean()}")

    # Check the "sharpe" ratio on the validation set
    validation_sharpe = validation_correlations.mean() / validation_correlations.std(ddof=0)
    print(f"Validation Sharpe: {validation_sharpe}")

    print("checking max drawdown...")
    rolling_max = (validation_correlations + 1).cumprod().rolling(window=100,
                                                                  min_periods=1).max()
    daily_value = (validation_correlations + 1).cumprod()
    max_drawdown = -((rolling_max - daily_value) / rolling_max).max()
    print(f"max drawdown: {max_drawdown}")

    # Check the feature exposure of your validation predictions
    feature_exposures = validation_data[feature_names].apply(lambda d: correlation(validation_data[PREDICTION_NAME], d),
                                                             axis=0)
    max_per_era = validation_data.groupby("era").apply(
        lambda d: d[feature_names].corrwith(d[PREDICTION_NAME]).abs().max())
    max_feature_exposure = max_per_era.mean()
    print(f"Max Feature Exposure: {max_feature_exposure}")

    # Check feature neutral mean
    print("Calculating feature neutral mean...")
    feature_neutral_mean = get_feature_neutral_mean(validation_data)
    print(f"Feature Neutral Mean is {feature_neutral_mean}")

    # Load example preds to get MMC metrics
    example_preds = pd.read_csv("example_predictions.csv").set_index("id")["prediction"]
    validation_example_preds = example_preds.loc[validation_data.index]
    validation_data["ExamplePreds"] = validation_example_preds

    print("calculating MMC stats...")
    # MMC over validation
    mmc_scores = []
    corr_scores = []
    for _, x in validation_data.groupby("era"):
        series = neutralize_series(pd.Series(unif(x[PREDICTION_NAME])),
                                   pd.Series(unif(x["ExamplePreds"])))
        mmc_scores.append(np.cov(series, x[TARGET_NAME])[0, 1] / (0.29 ** 2))
        corr_scores.append(correlation(unif(x[PREDICTION_NAME]), x[TARGET_NAME]))

    val_mmc_mean = np.mean(mmc_scores)
    val_mmc_std = np.std(mmc_scores)
    val_mmc_sharpe = val_mmc_mean / val_mmc_std
    corr_plus_mmcs = [c + m for c, m in zip(corr_scores, mmc_scores)]
    corr_plus_mmc_sharpe = np.mean(corr_plus_mmcs) / np.std(corr_plus_mmcs)
    corr_plus_mmc_mean = np.mean(corr_plus_mmcs)
    corr_plus_mmc_sharpe_diff = corr_plus_mmc_sharpe - validation_sharpe

    print(
        f"MMC Mean: {val_mmc_mean}\n"
        f"Corr Plus MMC Sharpe:{corr_plus_mmc_sharpe}\n"
        f"Corr Plus MMC Diff:{corr_plus_mmc_sharpe_diff}"
    )

    # Check correlation with example predictions
    full_df = pd.concat([validation_example_preds, validation_data[PREDICTION_NAME], validation_data["era"]], axis=1)
    full_df.columns = ["example_preds", "prediction", "era"]
    per_era_corrs = full_df.groupby('era').apply(lambda d: correlation(unif(d["prediction"]), unif(d["example_preds"])))
    corr_with_example_preds = per_era_corrs.mean()
    print(f"Corr with example preds: {corr_with_example_preds}")

    # Save predictions as a CSV and upload to https://numer.ai
    tournament_data[PREDICTION_NAME].to_csv("submission.csv", header=True)


""" 
functions used for advanced metrics
"""


# to neutralize a column in a df by many other columns on a per-era basis
def neutralize(df,
               columns,
               extra_neutralizers=None,
               proportion=1.0,
               normalize=True,
               era_col="era"):
    # need to do this for lint to be happy bc [] is a "dangerous argument"
    if extra_neutralizers is None:
        extra_neutralizers = []
    unique_eras = df[era_col].unique()
    computed = []
    for u in unique_eras:
        print(u, end="\r")
        df_era = df[df[era_col] == u]
        scores = df_era[columns].values
        if normalize:
            scores2 = []
            for x in scores.T:
                x = (pd.Series(x).rank(method="first").values - .5) / len(x)
                scores2.append(x)
            scores = np.array(scores2).T
            extra = df_era[extra_neutralizers].values
            exposures = np.concatenate([extra], axis=1)
        else:
            exposures = df_era[extra_neutralizers].values

        scores -= proportion * exposures.dot(
            np.linalg.pinv(exposures.astype(np.float32)).dot(scores.astype(np.float32)))

        scores /= scores.std(ddof=0)

        computed.append(scores)

    return pd.DataFrame(np.concatenate(computed),
                        columns=columns,
                        index=df.index)


# to neutralize any series by any other series
def neutralize_series(series, by, proportion=1.0):
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)

    # this line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
    exposures = np.hstack(
        (exposures,
         np.array([np.mean(series)] * len(exposures)).reshape(-1, 1)))

    correction = proportion * (exposures.dot(
        np.linalg.lstsq(exposures, scores, rcond=None)[0]))
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized


def unif(df):
    x = (df.rank(method="first") - 0.5) / len(df)
    return pd.Series(x, index=df.index)


def get_feature_neutral_mean(df):
    feature_cols = [c for c in df.columns if c.startswith("feature")]
    df.loc[:, "neutral_sub"] = neutralize(df, [PREDICTION_NAME],
                                          feature_cols)[PREDICTION_NAME]
    scores = df.groupby("era").apply(
        lambda x: correlation(x["neutral_sub"], x[TARGET_NAME])).mean()
    return np.mean(scores)


if __name__ == '__main__':
    main()
