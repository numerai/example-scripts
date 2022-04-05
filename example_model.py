import pandas as pd
from lightgbm import LGBMRegressor
import gc
import json
from pathlib import Path

from numerapi import NumerAPI
from utils import (
    save_model,
    load_model,
    neutralize,
    get_biggest_change_features,
    validation_metrics,
    ERA_COL,
    DATA_TYPE_COL,
    TARGET_COL,
    EXAMPLE_PREDS_COL
)


# download all the things

napi = NumerAPI()

current_round = napi.get_current_round()

# Tournament data changes every week so we specify the round in their name. Training
# and validation data only change periodically, so no need to download them every time.
print('Downloading dataset files...')

Path("./v4").mkdir(parents=False, exist_ok=True)
napi.download_dataset("v4/train.parquet")
napi.download_dataset("v4/validation.parquet")
napi.download_dataset("v4/live.parquet")
napi.download_dataset("v4/validation_example_preds.parquet")
napi.download_dataset("v4/features.json")

print('Reading minimal training data')
# read the feature metadata and get a feature set (or all the features)
with open("v4/features.json", "r") as f:
    feature_metadata = json.load(f)
# features = list(feature_metadata["feature_stats"].keys()) # get all the features
# features = feature_metadata["feature_sets"]["small"] # get the small feature set
features = feature_metadata["feature_sets"]["medium"] # get the medium feature set
# read in just those features along with era and target columns
read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]

# note: sometimes when trying to read the downloaded data you get an error about invalid magic parquet bytes...
# if so, delete the file and rerun the napi.download_dataset to fix the corrupted file
training_data = pd.read_parquet('v4/train.parquet',
                                columns=read_columns)
validation_data = pd.read_parquet('v4/validation.parquet',
                                  columns=read_columns)
live_data = pd.read_parquet(f'v4/live.parquet',
                                  columns=read_columns)


# pare down the number of eras to every 4th era
# every_4th_era = training_data[ERA_COL].unique()[::4]
# training_data = training_data[training_data[ERA_COL].isin(every_4th_era)]

# getting the per era correlation of each feature vs the target
all_feature_corrs = training_data.groupby(ERA_COL).apply(
    lambda era: era[features].corrwith(era[TARGET_COL])
)

# find the riskiest features by comparing their correlation vs
# the target in each half of training data; we'll use these later
riskiest_features = get_biggest_change_features(all_feature_corrs, 50)

# "garbage collection" (gc) gets rid of unused data and frees up memory
gc.collect()

model_name = f"model_target"
print(f"Checking for existing model '{model_name}'")
model = load_model(model_name)
if not model:
    print(f"model not found, creating new one")
    params = {"n_estimators": 2000,
              "learning_rate": 0.01,
              "max_depth": 5,
              "num_leaves": 2 ** 5,
              "colsample_bytree": 0.1}

    model = LGBMRegressor(**params)

    # train on all of train and save the model so we don't have to train next time
    model.fit(training_data.filter(like='feature_', axis='columns'),
              training_data[TARGET_COL])
    print(f"saving new model: {model_name}")
    save_model(model, model_name)

gc.collect()

nans_per_col = live_data[live_data["data_type"] == "live"][features].isna().sum()

# check for nans and fill nans
if nans_per_col.any():
    total_rows = len(live_data[live_data["data_type"] == "live"])
    print(f"Number of nans per column this week: {nans_per_col[nans_per_col > 0]}")
    print(f"out of {total_rows} total rows")
    print(f"filling nans with 0.5")
    live_data.loc[:, features] = live_data.loc[:, features].fillna(0.5)

else:
    print("No nans in the features this week!")


# double check the feature that the model expects vs what is available to prevent our
# pipeline from failing if Numerai adds more data and we don't have time to retrain!
model_expected_features = model.booster_.feature_name()
if set(model_expected_features) != set(features):
    print(f"New features are available! Might want to retrain model {model_name}.")
validation_data.loc[:, f"preds_{model_name}"] = model.predict(
    validation_data.loc[:, model_expected_features])
live_data.loc[:, f"preds_{model_name}"] = model.predict(
    live_data.loc[:, model_expected_features])

gc.collect()

# neutralize our predictions to the riskiest features
validation_data[f"preds_{model_name}_neutral_riskiest_50"] = neutralize(
    df=validation_data,
    columns=[f"preds_{model_name}"],
    neutralizers=riskiest_features,
    proportion=1.0,
    normalize=True,
    era_col=ERA_COL
)

live_data[f"preds_{model_name}_neutral_riskiest_50"] = neutralize(
    df=live_data,
    columns=[f"preds_{model_name}"],
    neutralizers=riskiest_features,
    proportion=1.0,
    normalize=True,
    era_col=ERA_COL
)

model_to_submit = f"preds_{model_name}_neutral_riskiest_50"

# rename best model to "prediction" and rank from 0 to 1 to meet upload requirements
validation_data["prediction"] = validation_data[model_to_submit].rank(pct=True)
live_data["prediction"] = live_data[model_to_submit].rank(pct=True)
validation_data["prediction"].to_csv(f"validation_predictions_{current_round}.csv")
live_data["prediction"].to_csv(f"live_predictions_{current_round}.csv")

validation_preds = pd.read_parquet('v4/validation_example_preds.parquet')
validation_data[EXAMPLE_PREDS_COL] = validation_preds["prediction"]

# get some stats about each of our models to compare...
# fast_mode=True so that we skip some of the stats that are slower to calculate
validation_stats = validation_metrics(validation_data, [model_to_submit, f"preds_{model_name}"], example_col=EXAMPLE_PREDS_COL, fast_mode=True, target_col=TARGET_COL)
print(validation_stats[["mean", "sharpe"]].to_markdown())

print(f'''
Done! Next steps:
    1. Go to numer.ai/tournament (make sure you have an account)
    2. Submit validation_predictions_{current_round}.csv to the diagnostics tool
    3. Submit tournament_predictions_{current_round}.csv to the "Upload Predictions" button
''')
