import pandas as pd
from lightgbm import LGBMRegressor
import gc
from numerapi import NumerAPI
from halo import Halo
from utils import save_model, load_model, neutralize, get_biggest_change_features, validation_metrics, download_data

napi = NumerAPI()

# read in all of the new datas
with Halo(text='Downloading training data', spinner='dots'):
    download_data(napi, 'numerai_training_data.parquet', 'numerai_training_data.parquet', round=280)

with Halo(text='Downloading tournament data', spinner='dots'):
    download_data(napi, 'numerai_tournament_data.parquet', 'numerai_tournament_data.parquet', round=280)

with Halo(text='Downloading validation data', spinner='dots'):
    download_data(napi, 'numerai_validation_data.parquet', 'numerai_validation_data.parquet', round=280)

with Halo(text='Downloading example tournament preds', spinner='dots'):
    download_data(napi, 'example_predictions.parquet', 'example_predictions.parquet', round=280)

with Halo(text='Downloading example validation preds', spinner='dots'):
    download_data(napi, 'example_validation_predictions.parquet', 'example_validation_predictions.parquet', round=280)

with Halo(text='Reading parquet data', spinner='dots'):
    training_data = pd.read_parquet('numerai_training_data.parquet')
    tournament_data = pd.read_parquet('numerai_tournament_data.parquet')
    validation_data = pd.read_parquet('numerai_validation_data.parquet')
    example_preds = pd.read_parquet('example_predictions.parquet')
    validation_preds = pd.read_parquet('example_validation_predictions.parquet')

EXAMPLE_PREDS_COL = "example_preds"
validation_data[EXAMPLE_PREDS_COL] = validation_preds["prediction"]

TARGET_COL = "target"
ERA_COL = "era"

feature_cols = [c for c in training_data if c.startswith("feature_")]
targets = [c for c in training_data if c.startswith("target_")]

gc.collect()

models = {}
for target in targets:
    model_name = f"example_model_{target}"
    print(f"predicting {model_name}")
    model = load_model(model_name)
    if not model:
        print(f"model not found, training new one")
        params = {"n_estimators": 2000,
                  "learning_rate": 0.01,
                  "max_depth": 5,
                  "colsample_bytree": 0.1}

        model = LGBMRegressor(**params)

        # train on all of train, predict on val, predict on tournament
        with Halo(text='Training model', spinner='dots'):
            model.fit(training_data.loc[:, feature_cols], training_data[target])
            print(f"saving new model: {model_name}")
            save_model(model, model_name)

    models[model_name] = model

# check for nans and fill nans
if tournament_data.loc[tournament_data["data_type"] == "live", feature_cols].isna().sum().sum():
    cols_w_nan = tournament_data.loc[tournament_data["data_type"] == "live", feature_cols].isna().sum()
    total_rows = tournament_data[tournament_data["data_type"] == "live"]
    print(f"Number of nans per column this week: {cols_w_nan[cols_w_nan > 0]}")
    print(f"out of {total_rows} total rows")
    print(f"filling nans with 0.5")
    tournament_data.loc[:, feature_cols].fillna(0.5, inplace=True)
else:
    print("No nans in the features this week!")

# predict on the latest data!
with Halo(text='Predicting on latest data', spinner='dots'):
    for model_name, model in models.items():
        print(model_name)
        validation_data.loc[:, f"preds_{model_name}"] = model.predict(validation_data.loc[:, feature_cols])
        tournament_data.loc[:, f"preds_{model_name}"] = model.predict(tournament_data.loc[:, feature_cols])

# find the riskiest features
all_feature_corrs = training_data.groupby(ERA_COL).apply(lambda d: d[feature_cols].corrwith(d[TARGET_COL]))
riskiest_features = get_biggest_change_features(all_feature_corrs, 50)

pred_cols = []
with Halo(text='Neutralizing to risky features', spinner='dots'):
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
validation_data["ensemble"] = sum([validation_data[pred_col] for pred_col in pred_cols]) / len(pred_cols)
tournament_data["ensemble"] = sum([tournament_data[pred_col] for pred_col in pred_cols]) / len(pred_cols)

# rename ensemble to prediction to meet diagnostic/submission file requirements
validation_data["prediction"] = validation_data["ensemble"].rank(pct=True)
tournament_data["prediction"] = tournament_data["ensemble"].rank(pct=True)
validation_data["prediction"].to_csv("validation_predictions.csv")
tournament_data["prediction"].to_csv("tournament_predictions.csv")

# calculate validation locally
validation_stats = validation_metrics(validation_data, pred_cols + ["ensemble"], example_col=EXAMPLE_PREDS_COL)

print(validation_stats.to_markdown())
