import dill as pickle
import pandas as pd
import numpy as np
from halo import Halo
from numerapi import NumerAPI
from pathlib import Path
from xgboost import XGBRegressor
from utils import save_model, load_model, neutralize, get_biggest_change_features, validation_metrics, download_data

# read in all of the new datas
# TODO: run this without public and secret key
public_id = '***REMOVED***'
secret_key = '***REMOVED***'
napi = NumerAPI(public_id=public_id, secret_key=secret_key)

# TODO: dont download if file already present
with Halo(text='Downloading training data', spinner='dots'):
    download_data(napi, 'numerai_training_data.parquet', 'numerai_training_data.parquet', round=279)

with Halo(text='Downloading tournament data', spinner='dots'):
    download_data(napi, 'numerai_tournament_data.parquet', 'numerai_tournament_data.parquet', round=279)

with Halo(text='Downloading validation data', spinner='dots'):
    download_data(napi, 'numerai_validation_data.parquet', 'numerai_validation_data.parquet', round=279)

# training_data = pd.read_csv('numerai_training_data.csv')
# tournament_data = pd.read_csv('numerai_tournament_data.csv')
# validation_data = pd.read_csv('numerai_validation_data.csv')
# training_data = pd.read_csv('numerai_training_data.csv')
# tournament_data = pd.read_csv('numerai_tournament_data.csv')
# validation_data = pd.read_csv('numerai_validation_data.csv')

# placeholder until new targets are included in tournament file
training_data["target_nomi"] = training_data["target"]
tournament_data["target_nomi"] = tournament_data["target"]
validation_data["target_nomi"] = validation_data["target"]

EXAMPLE_PREDS_COL = "example_preds"
# placeholder until we are reading from example_preds correctly
training_data[EXAMPLE_PREDS_COL] = np.random.uniform(0, 1, len(training_data))
tournament_data[EXAMPLE_PREDS_COL] = np.random.uniform(0, 1, len(tournament_data))
validation_data[EXAMPLE_PREDS_COL] = np.random.uniform(0, 1, len(validation_data))

TARGET_COL = "target"
ERA_COL = "era"

MODEL_FILE = Path("example_model.xgb")

feature_cols = [c for c in training_data if c.startswith("feature_")]
targets = [c for c in training_data if c.startswith("target_")]

models = {}
for target in targets:
    model_name = f"example_model_{target}"
    print(f"doing {model_name}")
    load_model(MODEL_FILE)
    if not model:
        print(f"model not found, training new one")
        params = {"n_estimators": 2000,
                  "learning_rate": 0.01,
                  "max_depth": 5,
                  "colsample_bytree": 0.1}

        model = XGBRegressor(**params)

        # train on all of train, predict on val, predict on tournament
        model.fit(training_data.loc[:, feature_cols], training_data[target])
        pickle.dump(model, open(MODEL_FILE, "wb"))

    models[model_name] = model

# check for nans and fill nans
if tournament_data.loc[tournament_data["data_type"] == "live", feature_cols].isna().sum().sum():
    cols_w_nan = tournament_data.loc[tournament_data["data_type"] == "live", feature_cols].isna().sum()
    total_rows = tournament_data[tournament_data["data_type"] == "live"]
    print(f"number of nans per column this week: {cols_w_nan[cols_w_nan > 0]}")
    print(f"out of {total_rows} total rows")
    print(f"filling nans with 0.5")
    tournament_data.loc[:, feature_cols].fillna(0.5, inplace=True)
else:
    print("no nans in the features this week!")

# predict on the latest data!
for model_name, model in models.items():
    validation_data.loc[:, f"preds_{model_name}"] = model.predict(validation_data.loc[:, feature_cols])
    tournament_data.loc[:, f"preds_{model_name}"] = model.predict(tournament_data.loc[:, feature_cols])

# find the riskiest features
all_feature_corrs = training_data.groupby(ERA_COL).apply(lambda d: d[feature_cols].corrwith(d[TARGET_COL]))
riskiest_features = get_biggest_change_features(all_feature_corrs, 50)

pred_cols = []
for model_name, model in models.items():
    # add the base preds as a pred col
    pred_cols.append(f"preds_{model_name}")

    # neutralize our predictions to the riskiest features only
    validation_data[f"preds_{model_name}_neutral_riskiest_50"] = neutralize(df=validation_data,
                                                                            columns=[f"preds_{model_name}"],
                                                                            neutralizers=riskiest_features,
                                                                            proportion=1.0,
                                                                            normalize=True,
                                                                            era_col=ERA_COL)

    tournament_data[f"preds_{model_name}_neutral_riskiest_50"] = neutralize(df=tournament_data,
                                                                            columns=[f"preds_{model_name}"],
                                                                            neutralizers=riskiest_features,
                                                                            proportion=1.0,
                                                                            normalize=True,
                                                                            era_col=ERA_COL)
    pred_cols.append(f"preds_{model_name}_neutral_riskiest_50")

# average all of our models of all neutralization types
validation_data["ensemble"] = sum([validation_data[pred_col] for pred_col in pred_cols])/len(pred_cols)
tournament_data["ensemble"] = sum([tournament_data[pred_col] for pred_col in pred_cols])/len(pred_cols)

# calculate validation locally
validation_stats = validation_metrics(validation_data, pred_cols+["ensemble"], example_col=EXAMPLE_PREDS_COL)
print(validation_stats)