
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from halo import Halo
from numerapi import NumerAPI
from xgboost import XGBRegressor
import gc
from utils import save_model, load_model, neutralize, get_biggest_change_features, unif, validation_metrics, download_data


def read_parquet(file_path):
    pfile = pq.read_table(file_path)
    column_names = pfile.column_names

    dtypes = {x: np.float16 for x in column_names if x.startswith(('feature', 'target'))}

    # Memory constrained? Try this instead (slower, but more memory efficient)
    # see https://forum.numer.ai/t/saving-memory-with-uint8-features/254
    # dtypes = {f"target": np.float16}
    # to_uint8 = lambda x: np.uint8(float(x) * 4)
    # converters = {x: to_uint8 for x in column_names if x.startswith('feature')}
    df = pd.read_parquet(file_path).astype(np.float16, errors='ignore')

    return df

# TODO: run this without public and secret key
public_id = 'V22H76F7UGZXRFHUK7EWRG53TJC34OVW'
secret_key = '6YCUIF523ALJIZE3GKIZU7BROOJRURHZQ3GAAJ4NAZHH7Z5GUWOAQKIY3LNZW753'
napi = NumerAPI(public_id=public_id, secret_key=secret_key)

# read in all of the new datas
with Halo(text='Downloading training data', spinner='dots'):
    download_data(napi, 'numerai_training_data.parquet', 'numerai_training_data.parquet', round=279)

with Halo(text='Downloading tournament data', spinner='dots'):
    download_data(napi, 'numerai_tournament_data.parquet', 'numerai_tournament_data.parquet', round=279)

with Halo(text='Downloading validation data', spinner='dots'):
    download_data(napi, 'numerai_validation_data.parquet', 'numerai_validation_data.parquet', round=279)

with Halo(text='Reading parquet data', spinner='dots'):
    training_data = read_parquet('numerai_training_data.parquet')
    tournament_data = read_parquet('numerai_tournament_data.parquet')
    validation_data = read_parquet('numerai_validation_data.parquet')

# placeholder until new targets are included in tournament file
training_data["target_nomi"] = training_data["target"]
tournament_data["target_nomi"] = tournament_data["target"]
validation_data["target_nomi"] = validation_data["target"]

EXAMPLE_PREDS_COL = "example_preds"
# placeholder until we are reading from example_preds correctly
with Halo(text='Downloading validation data', spinner='dots'):
    download_data(napi, 'example_predictions.csv', 'example_predictions.csv', round=279)
    download_data(napi, 'example_validation_predictions.csv', 'example_validation_predictions.csv', round=279)
    example_preds = pd.read_csv('example_predictions.csv')
    validation_preds = pd.read_csv('example_validation_predictions.csv')

    tournament_data[EXAMPLE_PREDS_COL] = example_preds[f"prediction"]
    validation_data[EXAMPLE_PREDS_COL] = validation_preds[f"prediction"]

TARGET_COL = "target"
ERA_COL = "era"

feature_cols = [c for c in training_data if c.startswith("feature_")]
targets = [c for c in training_data if c.startswith("target_")]

models = {}
for target in targets:
    gc.collect()
    model_name = f"example_model_{target}"
    print(f"doing {model_name}")
    model = load_model(model_name)
    if not model:
        with Halo(text='Training model', spinner='dots'):
            params = {"n_estimators": 20,
                      "learning_rate": 0.01,
                      "max_depth": 2,
                      "colsample_bytree": 0.1,
                      "n_jobs": 1}

            model = XGBRegressor(**params)

            # train on all of train, predict on val, predict on tournament
            # model.fit(training_data.loc[:, feature_cols], training_data[target])
            model.fit(training_data.loc[:1000, feature_cols], training_data[target])
            save_model(model, model_name)

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