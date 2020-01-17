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

BENCHMARK = 0
BAND = 0.2


# Submissions are scored by spearman correlation
def score(df):
    # method="first" breaks ties based on order in array
    return np.corrcoef(
        df[TARGET_NAME],
        df[PREDICTION_NAME].rank(pct=True, method="first")
    )[0, 1]


# The payout function
def payout(scores):
    return ((scores - BENCHMARK) / BAND).clip(lower=-1, upper=1)


# Read the csv file into a pandas Dataframe
def read_csv(file_path):
    with open(file_path, 'r') as f:
        column_names = next(csv.reader(f))
        dtypes = {x: np.float16 for x in column_names if
                  x.startswith(('feature', 'target'))}
    return pd.read_csv(file_path, dtype=dtypes)


def main():
    print("# Loading data...")
    # The training data is used to train your model how to predict the targets.
    training_data = read_csv("numerai_training_data.csv").set_index("id")
    # The tournament data is the data that Numerai uses to evaluate your model.
    tournament_data = read_csv("numerai_tournament_data.csv").set_index("id")

    feature_names = [f for f in training_data.columns if f.startswith("feature")]
    print(f"Loaded {len(feature_names)} features")

    print("Training model")
    # For faster experimentation you can decrease n_estimators to 200, for better performance increase to 20,000
    model = XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators=2000, n_jobs=-1, colsample_bytree=0.1)
    model.fit(training_data[feature_names], training_data[TARGET_NAME])

    print("Generating predictions")
    training_data[PREDICTION_NAME] = model.predict(training_data[feature_names])
    tournament_data[PREDICTION_NAME] = model.predict(tournament_data[feature_names])

    # Check the per-era correlations on the training set
    train_correlations = training_data.groupby("era").apply(score)
    print(f"On training the correlation has mean {train_correlations.mean()} and std {train_correlations.std()}")
    print(f"On training the average per-era payout is {payout(train_correlations).mean()}")

    # Check the per-era correlations on the validation set
    validation_data = tournament_data[tournament_data.data_type == "validation"]
    validation_correlations = validation_data.groupby("era").apply(score)
    print(
        f"On validation the correlation has mean {validation_correlations.mean()} and std {validation_correlations.std()}")
    print(f"On validation the average per-era payout is {payout(validation_correlations).mean()}")

    tournament_data[PREDICTION_NAME].to_csv(TOURNAMENT_NAME + "_submission.csv")
    # Now you can upload these predictions on https://numer.ai


if __name__ == '__main__':
    main()
