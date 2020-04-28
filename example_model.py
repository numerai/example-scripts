#!/usr/bin/env python
"""
Example classifier on Numerai data using a xgboost regression.
To get started, install the required packages: pip install pandas numpy sklearn xgboost
"""

import csv

import pandas as pd
import numpy as np
from xgboost import XGBRegressor

TOURNAMENT_NAME = "kazutsugi"
TARGET_NAME = f"target_{TOURNAMENT_NAME}"
PREDICTION_NAME = f"prediction_{TOURNAMENT_NAME}"


# Submissions are scored by spearman correlation
def score(df):
    # method="first" breaks ties based on order in array
    pct_ranks = df[PREDICTION_NAME].rank(pct=True, method="first")
    targets = df[TARGET_NAME]
    return np.corrcoef(targets, pct_ranks)[0, 1]


# Payout is just the score cliped at +/-25%
def payout(scores):
    return scores.clip(lower=-0.25, upper=0.25)


# Read the csv file into a pandas Dataframe
def read_csv(file_path):
    with open(file_path, 'r') as f:
        column_names = next(csv.reader(f))

    dtypes = {x: np.float16 for x in column_names if x.startswith(('feature', 'target'))}
    df = pd.read_csv(file_path, dtype=dtypes)

    # Memory constrained? Try this instead (slower, but more memory efficient)
    # see https://forum.numer.ai/t/saving-memory-with-uint8-features/254
    # dtypes = {f"target_{TOURNAMENT_NAME}": np.float16}
    # to_uint8 = lambda x: np.uint8(float(x) * 4)
    # converters = {x: to_uint8 for x in column_names if x.startswith('feature')}
    # df = pd.read_csv(file_path, dtype=dtypes, converters=converters)

    return df.set_index("id")


def main():
    print("Loading data...")
    # The training data is used to train your model how to predict the targets.
    training_data = read_csv("numerai_training_data_small.csv")
    # The tournament data is the data that Numerai uses to evaluate your model.
    tournament_data = read_csv("numerai_tournament_data.csv")

    feature_names = [
        f for f in training_data.columns if f.startswith("feature")
    ]
    print(f"Loaded {len(feature_names)} features")

    print("Training model...")
    # This is the model that generates the included example predictions file
    # Taking too long? Set learning_rate=0.01 and n_estimators=200 to make this run faster
    model = XGBRegressor(max_depth=5, learning_rate=0.001, n_estimators=20000, n_jobs=-1, colsample_bytree=0.1)
    model.fit(training_data[feature_names], training_data[TARGET_NAME])

    # Generate predictions on both training and tournament data
    print("Generating predictions...")
    training_data[PREDICTION_NAME] = model.predict(training_data[feature_names])
    tournament_data[PREDICTION_NAME] = model.predict(tournament_data[feature_names])

    # Check the per-era correlations on the training set (in sample)
    train_correlations = training_data.groupby("era").apply(score)
    print(f"On training the correlation has mean {train_correlations.mean()} and std {train_correlations.std()}")
    print(f"On training the average per-era payout is {payout(train_correlations).mean()}")

    # Check the per-era correlations on the validation set (out of sample)
    validation_data = tournament_data[tournament_data.data_type == "validation"]
    validation_correlations = validation_data.groupby("era").apply(score)
    print(f"On validation the correlation has mean {validation_correlations.mean()} and std {validation_correlations.std()}")
    print(f"On validation the average per-era payout is {payout(validation_correlations).mean()}")

    # Save predictions as a CSV and upload to https://numer.ai
    tournament_data[PREDICTION_NAME].to_csv(TOURNAMENT_NAME + "_submission.csv")


if __name__ == '__main__':
    main()
