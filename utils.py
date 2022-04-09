import numpy as np
import pandas as pd
import scipy
from halo import Halo
from pathlib import Path
import json
from scipy.stats import skew

ERA_COL = "era"
TARGET_COL = "target_nomi_v4_20"
DATA_TYPE_COL = "data_type"
EXAMPLE_PREDS_COL = "example_preds"

spinner = Halo(text='', spinner='dots')

MODEL_FOLDER = "models"
MODEL_CONFIGS_FOLDER = "model_configs"
PREDICTION_FILES_FOLDER = "prediction_files"


def save_prediction(df, name):
    try:
        Path(PREDICTION_FILES_FOLDER).mkdir(exist_ok=True, parents=True)
    except Exception as ex:
        pass
    df.to_csv(f"{PREDICTION_FILES_FOLDER}/{name}.csv", index=True)


def save_model(model, name):
    try:
        Path(MODEL_FOLDER).mkdir(exist_ok=True, parents=True)
    except Exception as ex:
        pass
    pd.to_pickle(model, f"{MODEL_FOLDER}/{name}.pkl")


def load_model(name):
    path = Path(f"{MODEL_FOLDER}/{name}.pkl")
    if path.is_file():
        model = pd.read_pickle(f"{MODEL_FOLDER}/{name}.pkl")
    else:
        model = False
    return model


def save_model_config(model_config, model_name):
    try:
        Path(MODEL_CONFIGS_FOLDER).mkdir(exist_ok=True, parents=True)
    except Exception as ex:
        pass
    with open(f"{MODEL_CONFIGS_FOLDER}/{model_name}.json", 'w') as fp:
        json.dump(model_config, fp)


def load_model_config(model_name):
    path_str = f"{MODEL_CONFIGS_FOLDER}/{model_name}.json"
    path = Path(path_str)
    if path.is_file():
        with open(path_str, 'r') as fp:
            model_config = json.load(fp)
    else:
        model_config = False
    return model_config


def get_biggest_change_features(corrs, n):
    all_eras = corrs.index.sort_values()
    h1_eras = all_eras[:len(all_eras) // 2]
    h2_eras = all_eras[len(all_eras) // 2:]

    h1_corr_means = corrs.loc[h1_eras, :].mean()
    h2_corr_means = corrs.loc[h2_eras, :].mean()

    corr_diffs = h2_corr_means - h1_corr_means
    worst_n = corr_diffs.abs().sort_values(ascending=False).head(n).index.tolist()
    return worst_n


def get_time_series_cross_val_splits(data, cv=3, embargo=12):
    all_train_eras = data[ERA_COL].unique()
    len_split = len(all_train_eras) // cv
    test_splits = [all_train_eras[i * len_split:(i + 1) * len_split] for i in range(cv)]
    # fix the last test split to have all the last eras, in case the number of eras wasn't divisible by cv
    remainder = len(all_train_eras) % cv
    if remainder != 0:
        test_splits[-1] = np.append(test_splits[-1], all_train_eras[-remainder:])

    train_splits = []
    for test_split in test_splits:
        test_split_max = int(np.max(test_split))
        test_split_min = int(np.min(test_split))
        # get all of the eras that aren't in the test split
        train_split_not_embargoed = [e for e in all_train_eras if not (test_split_min <= int(e) <= test_split_max)]
        # embargo the train split so we have no leakage.
        # one era is length 5, so we need to embargo by target_length/5 eras.
        # To be consistent for all targets, let's embargo everything by 60/5 == 12 eras.
        train_split = [e for e in train_split_not_embargoed if
                       abs(int(e) - test_split_max) > embargo and abs(int(e) - test_split_min) > embargo]
        train_splits.append(train_split)

    # convenient way to iterate over train and test splits
    train_test_zip = zip(train_splits, test_splits)
    return train_test_zip


def neutralize(df,
               columns,
               neutralizers=None,
               proportion=1.0,
               normalize=True,
               era_col="era"):
    if neutralizers is None:
        neutralizers = []
    unique_eras = df[era_col].unique()
    computed = []
    for u in unique_eras:
        df_era = df[df[era_col] == u]
        scores = df_era[columns].values
        if normalize:
            scores2 = []
            for x in scores.T:
                x = (scipy.stats.rankdata(x, method='ordinal') - .5) / len(x)
                x = scipy.stats.norm.ppf(x)
                scores2.append(x)
            scores = np.array(scores2).T
        exposures = df_era[neutralizers].values

        scores -= proportion * exposures.dot(
            np.linalg.pinv(exposures.astype(np.float32), rcond=1e-6).dot(scores.astype(np.float32)))

        scores /= scores.std(ddof=0)

        computed.append(scores)

    return pd.DataFrame(np.concatenate(computed),
                        columns=columns,
                        index=df.index)


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


def get_feature_neutral_mean(df, prediction_col, target_col, features_for_neutralization=None):
    if features_for_neutralization is None:
        features_for_neutralization = [c for c in df.columns if c.startswith("feature")]
    df.loc[:, "neutral_sub"] = neutralize(df, [prediction_col],
                                          features_for_neutralization)[prediction_col]
    scores = df.groupby("era").apply(
        lambda x: (unif(x["neutral_sub"]).corr(x[target_col]))).mean()
    return np.mean(scores)

def get_feature_neutral_mean_tb_era(df, prediction_col, target_col, tb, features_for_neutralization=None):
    if features_for_neutralization is None:
        features_for_neutralization = [c for c in df.columns if c.startswith("feature")]
    temp_df = df.reset_index(drop=True).copy()        # Reset index due to use of argsort later
    temp_df.loc[:, "neutral_sub"] = neutralize(temp_df, [prediction_col],
                                          features_for_neutralization)[prediction_col]
    temp_df_argsort = temp_df.loc[:, 'neutral_sub'].argsort()
    temp_df_tb_idx = pd.concat([temp_df_argsort.iloc[:tb],
                           temp_df_argsort.iloc[-tb:]])
    temp_df_tb = temp_df.loc[temp_df_tb_idx]
    tb_fnc = unif(temp_df_tb['neutral_sub']).corr(temp_df_tb[target_col])
    return tb_fnc


def fast_score_by_date(df, columns, target, tb=None, era_col="era"):
    unique_eras = df[era_col].unique()
    computed = []
    for u in unique_eras:
        df_era = df[df[era_col] == u]
        era_pred = np.float64(df_era[columns].values.T)
        era_target = np.float64(df_era[target].values.T)

        if tb is None:
            ccs = np.corrcoef(era_target, era_pred)[0, 1:]
        else:
            tbidx = np.argsort(era_pred, axis=1)
            tbidx = np.concatenate([tbidx[:, :tb], tbidx[:, -tb:]], axis=1)
            ccs = [np.corrcoef(era_target[tmpidx], tmppred[tmpidx])[0, 1] for tmpidx, tmppred in zip(tbidx, era_pred)]
            ccs = np.array(ccs)

        computed.append(ccs)

    return pd.DataFrame(np.array(computed), columns=columns, index=df[era_col].unique())

def exposure_dissimilarity_per_era(df, prediction_col, example_col, feature_cols=None):
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c.startswith("feature")]
    u = df.loc[:, feature_cols].corrwith(df[prediction_col])
    e = df.loc[:, feature_cols].corrwith(df[example_col])
    return (1 - (np.dot(u,e)/np.dot(e,e)))

def validation_metrics(validation_data, pred_cols, example_col, fast_mode=False,
                        target_col=TARGET_COL, features_for_neutralization=None):
    validation_stats = pd.DataFrame()
    feature_cols = [c for c in validation_data if c.startswith("feature_")]
    for pred_col in pred_cols:
        # Check the per-era correlations on the validation set (out of sample)
        validation_correlations = validation_data.groupby(ERA_COL).apply(
            lambda d: unif(d[pred_col]).corr(d[target_col]))

        mean = validation_correlations.mean()
        std = validation_correlations.std(ddof=0)
        sharpe = mean / std

        validation_stats.loc["mean", pred_col] = mean
        validation_stats.loc["std", pred_col] = std
        validation_stats.loc["sharpe", pred_col] = sharpe

        rolling_max = (validation_correlations + 1).cumprod().rolling(window=9000,  # arbitrarily large
                                                                      min_periods=1).max()
        daily_value = (validation_correlations + 1).cumprod()
        max_drawdown = -((rolling_max - daily_value) / rolling_max).max()
        validation_stats.loc["max_drawdown", pred_col] = max_drawdown

        payout_scores = validation_correlations.clip(-0.25, 0.25)
        payout_daily_value = (payout_scores + 1).cumprod()

        apy = (
                      (
                              (payout_daily_value.dropna().iloc[-1])
                              ** (1 / len(payout_scores))
                      )
                      ** 49  # 52 weeks of compounding minus 3 for stake compounding lag
                      - 1
              ) * 100

        validation_stats.loc["apy", pred_col] = apy

        if not fast_mode:
            # Check the feature exposure of your validation predictions
            max_per_era = validation_data.groupby(ERA_COL).apply(
                lambda d: d[feature_cols].corrwith(d[pred_col]).abs().max())
            max_feature_exposure = max_per_era.mean()
            validation_stats.loc["max_feature_exposure", pred_col] = max_feature_exposure

            # Check feature neutral mean
            feature_neutral_mean = get_feature_neutral_mean(validation_data, pred_col,
                                                                target_col, features_for_neutralization)
            validation_stats.loc["feature_neutral_mean", pred_col] = feature_neutral_mean

            # Check TB200 feature neutral mean
            tb200_feature_neutral_mean_era = validation_data.groupby(ERA_COL).apply(lambda df: \
                                            get_feature_neutral_mean_tb_era(df, pred_col,
                                                                            target_col, 200,
                                                                            features_for_neutralization))
            validation_stats.loc["tb200_feature_neutral_mean", pred_col] = tb200_feature_neutral_mean_era.mean()

            # Check top and bottom 200 metrics (TB200)
            tb200_validation_correlations = fast_score_by_date(
                validation_data,
                [pred_col],
                target_col,
                tb=200,
                era_col=ERA_COL
            )

            tb200_mean = tb200_validation_correlations.mean()[pred_col]
            tb200_std = tb200_validation_correlations.std(ddof=0)[pred_col]
            tb200_sharpe = tb200_mean / tb200_std

            validation_stats.loc["tb200_mean", pred_col] = tb200_mean
            validation_stats.loc["tb200_std", pred_col] = tb200_std
            validation_stats.loc["tb200_sharpe", pred_col] = tb200_sharpe

        # MMC over validation
        mmc_scores = []
        corr_scores = []
        for _, x in validation_data.groupby(ERA_COL):
            series = neutralize_series(unif(x[pred_col]), (x[example_col]))
            mmc_scores.append(np.cov(series, x[target_col])[0, 1] / (0.29 ** 2))
            corr_scores.append(unif(x[pred_col]).corr(x[target_col]))

        val_mmc_mean = np.mean(mmc_scores)
        val_mmc_std = np.std(mmc_scores)
        corr_plus_mmcs = [c + m for c, m in zip(corr_scores, mmc_scores)]
        corr_plus_mmc_sharpe = np.mean(corr_plus_mmcs) / np.std(corr_plus_mmcs)

        validation_stats.loc["mmc_mean", pred_col] = val_mmc_mean
        validation_stats.loc["corr_plus_mmc_sharpe", pred_col] = corr_plus_mmc_sharpe

        # Check correlation with example predictions
        per_era_corrs = validation_data.groupby(ERA_COL).apply(lambda d: unif(d[pred_col]).corr(unif(d[example_col])))
        corr_with_example_preds = per_era_corrs.mean()
        validation_stats.loc["corr_with_example_preds", pred_col] = corr_with_example_preds

        #Check exposure dissimilarity per era
        tdf = validation_data.groupby(ERA_COL).apply(lambda df: \
                                                exposure_dissimilarity_per_era(df, pred_col,
                                                example_col, feature_cols))
        validation_stats.loc["exposure_dissimilarity_mean", pred_col] = tdf.mean()

    # .transpose so that stats are columns and the model_name is the row
    return validation_stats.transpose()
