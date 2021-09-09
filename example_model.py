import pandas as pd
from lightgbm import LGBMRegressor
import gc
from numerapi import NumerAPI
from halo import Halo
from utils import save_model, load_model, neutralize, get_biggest_change_features, validation_metrics, download_data


napi = NumerAPI()
spinner = Halo(text='', spinner='dots')

current_round = napi.get_current_round(tournament=8)  # tournament 8 is the primary Numerai Tournament

# read in all of the new datas
# tournament data and example predictions change every week so we specify the round in their names
# training and validation data only change periodically, so no need to download them over again every single week
napi.download_dataset("numerai_training_data.parquet", "numerai_training_data.parquet")
napi.download_dataset("numerai_tournament_data.parquet", f"numerai_tournament_data_{current_round}.parquet")
napi.download_dataset("numerai_validation_data.parquet", f"numerai_validation_data.parquet")
napi.download_dataset("example_predictions.parquet", f"example_predictions_{current_round}.parquet")
napi.download_dataset("example_validation_predictions.parquet", "example_validation_predictions.parquet")

spinner.start('Reading parquet data')
training_data = pd.read_parquet('numerai_training_data.parquet')
tournament_data = pd.read_parquet(f'numerai_tournament_data_{current_round}.parquet')
validation_data = pd.read_parquet('numerai_validation_data.parquet')
example_preds = pd.read_parquet(f'example_predictions_{current_round}.parquet')
validation_preds = pd.read_parquet('example_validation_predictions.parquet')
spinner.succeed()

EXAMPLE_PREDS_COL = "example_preds"
validation_data[EXAMPLE_PREDS_COL] = validation_preds["prediction"]

TARGET_COL = "target"
ERA_COL = "era"

# all feature columns start with the prefix "feature_"
feature_cols = [c for c in training_data if c.startswith("feature_")]

gc.collect()

model_name = f"model_target"
print(f"predicting {model_name}")
model = load_model(model_name)
if not model:
    print(f"model not found, training new one")
    params = {"n_estimators": 2000,
              "learning_rate": 0.01,
              "max_depth": 5,
              "num_leaves": 2 ** 5,
              "colsample_bytree": 0.1}

    model = LGBMRegressor(**params)

    # train on all of train, predict on val, predict on tournament, save the model so we don't have to train next time
    spinner.start('Training model')
    model.fit(training_data.loc[:, feature_cols], training_data[TARGET_COL])
    print(f"saving new model: {model_name}")
    save_model(model, model_name)
    spinner.succeed()

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
spinner.start('Predicting on latest data')
# double check the feature that the model expects vs what is available
# this prevents our pipeline from failing if Numerai adds more data and we don't have time to retrain!
model_expected_features = model.booster_.feature_name()
if set(model_expected_features) != set(feature_cols):
    print(f"New features are available! Might want to retrain model {model_name}.")
validation_data.loc[:, f"preds_{model_name}"] = model.predict(validation_data.loc[:, model_expected_features])
tournament_data.loc[:, f"preds_{model_name}"] = model.predict(tournament_data.loc[:, model_expected_features])
spinner.succeed()

spinner.start('Neutralizing to risky features')
# getting the per era correlation of each feature vs the target
all_feature_corrs = training_data.groupby(ERA_COL).apply(lambda d: d[feature_cols].corrwith(d[TARGET_COL]))

# find the riskiest features by comparing their correlation vs the target in half 1 and half 2 of training data
riskiest_features = get_biggest_change_features(all_feature_corrs, 50)

# neutralize our predictions to the riskiest features
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
spinner.succeed()


model_to_submit = f"preds_{model_name}_neutral_riskiest_50"
# rename best model to prediction and rank from 0 to 1 to meet diagnostic/submission file requirements
validation_data["prediction"] = validation_data[model_to_submit].rank(pct=True)
tournament_data["prediction"] = tournament_data[model_to_submit].rank(pct=True)
validation_data["prediction"].to_csv(f"validation_predictions_{current_round}.csv")
tournament_data["prediction"].to_csv(f"tournament_predictions_{current_round}.csv")

# get some stats about each of our models to compare...
# fast_mode=True so that we skip some of the stats that are slower to calculate
validation_stats = validation_metrics(validation_data, [model_to_submit], example_col=EXAMPLE_PREDS_COL, fast_mode=True)
print(validation_stats[["mean", "sharpe"]].to_markdown())





