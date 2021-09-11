import pandas as pd
from lightgbm import LGBMRegressor
import gc
from numerapi import NumerAPI
from utils import save_model, load_model, neutralize, get_biggest_change_features, validation_metrics, download_data, \
    load_model_config, save_model_config, get_time_series_cross_val_splits


EXAMPLE_PREDS_COL = "example_preds"
TARGET_COL = "target"
ERA_COL = "era"
# params we'll use to train all of our models.
# Ideal params would be more like 20000, 0.001, 6, 2**6, 0.1, but this is slow enough as it is
model_params = {"n_estimators": 2000,
                "learning_rate": 0.01,
                "max_depth": 5,
                "num_leaves": 2 ** 5,
                "colsample_bytree": 0.1}

# the amount of downsampling we'll use to speed up cross validation and full train.
# a value of 1 means no downsampling
# a value of 10 means use every 10th row
downsample_cross_val = 20
downsample_full_train = 1

# if model_selection_loop=True get OOS performance for training_data
# and use that to select best model
# if model_selection_loop=False, just predict on tournament data using existing models and model config
model_selection_loop = True
model_config_name = "advanced_example_model"

napi = NumerAPI()

current_round = napi.get_current_round(tournament=8)  # tournament 8 is the primary Numerai Tournament

print("Entering model selection loop.  This may take awhile.")
if model_selection_loop:
    model_config = {}
    print('downloading training_data')
    download_data(napi, 'numerai_training_data.parquet', 'numerai_training_data.parquet')

    print("reading training data from local file")
    training_data = pd.read_parquet('numerai_training_data.parquet')

    # keep track of some prediction columns
    ensemble_cols = set()
    pred_cols = set()

    # pick some targets to use
    possible_targets = [c for c in training_data.columns if c.startswith("target_")]
    # randomly pick a handful of targets
    # this can be vastly improved
    targets = ["target", "target_nomi_60", "target_jerome_20"]

    # all the possible features to train on
    feature_cols = [c for c in training_data if c.startswith("feature_")]

    """ do cross val to get out of sample training preds"""
    cv = 3
    train_test_zip = get_time_series_cross_val_splits(training_data, cv=cv, embargo=12)
    # get out of sample training preds via embargoed time series cross validation
    # optionally downsample training data to speed up this section.
    print("entering time series cross validation loop")
    for split, train_test_split in enumerate(train_test_zip):
        gc.collect()
        print(f"doing split {split+1} out of {cv}")
        train_split, test_split = train_test_split
        train_split_index = training_data[ERA_COL].isin(train_split)
        test_split_index = training_data[ERA_COL].isin(test_split)
        downsampled_train_split_index = train_split_index[train_split_index].index[::downsample_cross_val]

        # getting the per era correlation of each feature vs the primary target across the training split
        print("getting feature correlations over time and identifying riskiest features")
        all_feature_corrs_split = training_data.loc[downsampled_train_split_index, :].groupby(ERA_COL).apply(
            lambda d: d[feature_cols].corrwith(d[TARGET_COL]))
        # find the riskiest features by comparing their correlation vs the target in half 1 and half 2 of training data
        # there are probably more clever ways to do this
        riskiest_features_split = get_biggest_change_features(all_feature_corrs_split, 50)

        print(f"entering model training loop for split {split+1}")
        for target in targets:
            model_name = f"model_{target}"
            print(f"model: {model_name}")

            # train a model on the training split (and save it for future use)
            split_model_name = f"model_{target}_split{split+1}cv{cv}downsample{downsample_cross_val}"
            split_model = load_model(split_model_name)
            if not split_model:
                print(f"training model: {model_name}")
                split_model = LGBMRegressor(**model_params)
                split_model.fit(training_data.loc[downsampled_train_split_index, feature_cols],
                                training_data.loc[downsampled_train_split_index,
                                                  [target]])
                save_model(split_model, split_model_name)
            # now we can predict on the test part of the split
            model_expected_features = split_model.booster_.feature_name()
            if set(model_expected_features) != set(feature_cols):
                print(f"New features are available! Might want to retrain model {split_model_name}.")
            print(f"predicting {model_name}")
            training_data.loc[test_split_index, f"preds_{model_name}"] = \
                split_model.predict(training_data.loc[test_split_index, model_expected_features])

            # do neutralization
            print("doing neutralization to riskiest features")
            training_data.loc[test_split_index, f"preds_{model_name}_neutral_riskiest_50"] = neutralize(
                df=training_data.loc[test_split_index, :],
                columns=[f"preds_{model_name}"],
                neutralizers=riskiest_features_split,
                proportion=1.0,
                normalize=True,
                era_col=ERA_COL)[f"preds_{model_name}"]

            # remember that we made all of these different pred columns
            pred_cols.add(f"preds_{model_name}")
            pred_cols.add(f"preds_{model_name}_neutral_riskiest_50")

        print("creating ensembles")
        # ranking per era for all of our pred cols so we can combine safely on the same scales
        training_data[list(pred_cols)] = training_data.groupby(ERA_COL).apply(
            lambda d: d[list(pred_cols)].rank(pct=True))
        # do ensembles
        training_data["ensemble_neutral_riskiest_50"] = sum(
            [training_data[pred_col] for pred_col in pred_cols if pred_col.endswith("neutral_riskiest_50")]).rank(
            pct=True)
        training_data["ensemble_not_neutral"] = sum(
            [training_data[pred_col] for pred_col in pred_cols if "neutral" not in pred_col]).rank(pct=True)
        training_data["ensemble_all"] = sum([training_data[pred_col] for pred_col in pred_cols]).rank(pct=True)

        ensemble_cols.add("ensemble_neutral_riskiest_50")
        ensemble_cols.add("ensemble_not_neutral")
        ensemble_cols.add("ensemble_all")

    """ Now get some stats and pick our favorite model"""
    print("gathering validation metrics for out of sample training results")
    all_model_cols = list(pred_cols) + list(ensemble_cols)
    # use example_col preds_model_target as an estimates since no example preds provided for training
    # fast_mode=True so that we skip some of the stats that are slower to calculate
    training_stats = validation_metrics(training_data, all_model_cols, example_col="preds_model_target",
                                        fast_mode=True)
    print(training_stats[["mean", "sharpe"]].sort_values(by="sharpe", ascending=False).to_markdown())

    # pick the model that has the highest correlation sharpe
    best_pred_col = training_stats.sort_values(by="sharpe", ascending=False).head(1).index[0]
    print(f"selecting model {best_pred_col} as our highest sharpe model in validation")

    """ Now do a full train"""
    print("entering full training section")
    # getting the per era correlation of each feature vs the target across all of training data
    print("getting feature correlations with target and identifying riskiest features")
    all_feature_corrs = training_data.groupby(ERA_COL).apply(
        lambda d: d[feature_cols].corrwith(d[TARGET_COL]))
    # find the riskiest features by comparing their correlation vs the target in half 1 and half 2 of training data
    riskiest_features = get_biggest_change_features(all_feature_corrs, 50)

    for target in targets:
        gc.collect()
        model_name = f"model_{target}_downsample{downsample_full_train}"
        model = load_model(model_name)
        if not model:
            print(f"training {model_name}")
            model = LGBMRegressor(**model_params)
            # train on all of train, predict on val, predict on tournament
            model.fit(training_data.iloc[::downsample_full_train].loc[:, feature_cols],
                      training_data.iloc[::downsample_full_train][target])
            save_model(model, model_name)
        gc.collect()

    model_config["feature_cols"] = feature_cols
    model_config["targets"] = targets
    model_config["best_pred_col"] = best_pred_col
    model_config["riskiest_features"] = riskiest_features
    print(f"saving model config for {model_config_name}")
    save_model_config(model_config, model_config_name)
else:
    # load model config from previous model selection loop
    print(f"loading model config for {model_config_name}")
    model_config = load_model_config(model_config_name)
    feature_cols = model_config["feature_cols"]
    targets = model_config["targets"]
    best_pred_col = model_config["best_pred_col"]
    riskiest_features = model_config["riskiest_features"]


""" Things that we always do even if we've already trained """
gc.collect()
print("downloading tournament_data")
download_data(napi, 'numerai_tournament_data.parquet', f'numerai_tournament_data_{current_round}.parquet')
print("downloading validation_data")
download_data(napi, 'numerai_validation_data.parquet', 'numerai_validation_data.parquet')
print("downloading example_predictions")
download_data(napi, 'example_predictions.parquet', f'example_predictions_{current_round}.parquet')
print("downloading example_validation_predictions")
download_data(napi, 'example_validation_predictions.parquet', f'example_validation_predictions.parquet')

print("reading tournament_data")
tournament_data = pd.read_parquet(f'numerai_tournament_data_{current_round}.parquet')
print("reading validation_data")
validation_data = pd.read_parquet('numerai_validation_data.parquet')
print("reading example_predictions")
example_preds = pd.read_parquet(f'example_predictions_{current_round}.parquet')
print("reading example_validaton_predictions")
validation_example_preds = pd.read_parquet('example_validation_predictions.parquet')
# set the example predictions
validation_data[EXAMPLE_PREDS_COL] = validation_example_preds["prediction"]

# check for nans and fill nans
print("checking for nans in the tournament data")
if tournament_data.loc[tournament_data["data_type"] == "live", feature_cols].isna().sum().sum():
    cols_w_nan = tournament_data.loc[tournament_data["data_type"] == "live", feature_cols].isna().sum()
    total_rows = tournament_data[tournament_data["data_type"] == "live"]
    print(f"Number of nans per column this week: {cols_w_nan[cols_w_nan > 0]}")
    print(f"out of {total_rows} total rows")
    print(f"filling nans with 0.5")
    tournament_data.loc[:, feature_cols].fillna(0.5, inplace=True)
else:
    print("No nans in the features this week!")


pred_cols = set()
ensemble_cols = set()
for target in targets:
    gc.collect()
    model_name = f"model_{target}_downsample{downsample_full_train}"
    print(f"loading {model_name}")
    model = load_model(model_name)
    if not model:
        raise ValueError(f"{model_name} is not trained yet!")

    model_expected_features = model.booster_.feature_name()
    if set(model_expected_features) != set(feature_cols):
        print(f"New features are available! Might want to retrain model {model_name}.")
    print(f"predicting tournament and validation for {model_name}")
    validation_data.loc[:, f"preds_{model_name}"] = model.predict(validation_data.loc[:, model_expected_features])
    tournament_data.loc[:, f"preds_{model_name}"] = model.predict(tournament_data.loc[:, model_expected_features])

    # do different neutralizations
    # neutralize our predictions to the riskiest features only
    print("neutralizing to riskiest_50 for validation and tournament")
    validation_data[f"preds_{model_name}_neutral_riskiest_50"] = neutralize(df=validation_data,
                                                                            columns=[f"preds_{model_name}"],
                                                                            neutralizers=riskiest_features,
                                                                            proportion=1.0,
                                                                            normalize=True,
                                                                            era_col=ERA_COL)[f"preds_{model_name}"]
    tournament_data[f"preds_{model_name}_neutral_riskiest_50"] = neutralize(df=tournament_data,
                                                                            columns=[f"preds_{model_name}"],
                                                                            neutralizers=riskiest_features,
                                                                            proportion=1.0,
                                                                            normalize=True,
                                                                            era_col=ERA_COL)[f"preds_{model_name}"]

    pred_cols.add(f"preds_{model_name}")
    pred_cols.add(f"preds_{model_name}_neutral_riskiest_50")


# rank per era for each prediction column so that we can combine safely
validation_data[list(pred_cols)] = validation_data.groupby(ERA_COL).apply(lambda d: d[list(pred_cols)].rank(pct=True))
tournament_data[list(pred_cols)] = tournament_data.groupby(ERA_COL).apply(lambda d: d[list(pred_cols)].rank(pct=True))
# make ensembles for val and tournament
print('creating ensembles for tournament and validation')
validation_data["ensemble_neutral_riskiest_50"] = sum(
    [validation_data[pred_col] for pred_col in pred_cols if pred_col.endswith("neutral_riskiest_50")]).rank(
    pct=True)
tournament_data["ensemble_neutral_riskiest_50"] = sum(
    [tournament_data[pred_col] for pred_col in pred_cols if pred_col.endswith("neutral_riskiest_50")]).rank(
    pct=True)
ensemble_cols.add("ensemble_neutral_riskiest_50")

validation_data["ensemble_not_neutral"] = sum(
    [validation_data[pred_col] for pred_col in pred_cols if "neutral" not in pred_col]).rank(pct=True)
tournament_data["ensemble_not_neutral"] = sum(
    [tournament_data[pred_col] for pred_col in pred_cols if "neutral" not in pred_col]).rank(pct=True)
ensemble_cols.add("ensemble_not_neutral")

validation_data["ensemble_all"] = sum([validation_data[pred_col] for pred_col in pred_cols]).rank(pct=True)
tournament_data["ensemble_all"] = sum([tournament_data[pred_col] for pred_col in pred_cols]).rank(pct=True)

ensemble_cols.add("ensemble_neutral_riskiest_50")
ensemble_cols.add("ensemble_not_neutral")
ensemble_cols.add("ensemble_all")

gc.collect()
print("getting final validation stats")
# get our final validation stats for our chosen model
validation_stats = validation_metrics(validation_data, [best_pred_col], example_col=EXAMPLE_PREDS_COL,
                                      fast_mode=False)
print(validation_stats.to_markdown())

# rename best model to prediction and rank from 0 to 1 to meet diagnostic/submission file requirements
validation_data["prediction"] = validation_data[best_pred_col].rank(pct=True)
tournament_data["prediction"] = tournament_data[best_pred_col].rank(pct=True)
validation_data["prediction"].to_csv(f"prediction_files/validation_predictions_{current_round}.csv", index=True)
tournament_data["prediction"].to_csv(f"prediction_files/tournament_predictions_{current_round}.csv", index=True)
