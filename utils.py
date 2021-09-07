import os
import logging
import requests
import numpy as np
import pandas as pd
import scipy
from halo import Halo
from pathlib import Path

ERA_COL = "era"
TARGET_COL = "target"


def save_model(model, name):
    pd.to_pickle(model, f"{name}.pkl")


def load_model(name):
    path = Path(f"{name}.pkl")
    if path.is_file():
        model = pd.read_pickle(f"{name}.pkl")
    else:
        model = False
    return model


def get_biggest_change_features(corrs, n):
    all_eras = corrs.index.sort_values()
    h1_eras = all_eras[:len(all_eras) // 2]
    h2_eras = all_eras[len(all_eras) // 2:]

    h1_corr_means = corrs.loc[h1_eras, :].mean()
    h2_corr_means = corrs.loc[h2_eras, :].mean()

    corr_diffs = h2_corr_means - h1_corr_means
    worst_n = corr_diffs.abs().sort_values(ascending=False).head(n).index.tolist()
    return worst_n


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
            np.linalg.pinv(exposures.astype(np.float32)).dot(scores.astype(np.float32)))

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


def get_feature_neutral_mean(df, prediction_col):
    feature_cols = [c for c in df.columns if c.startswith("feature")]
    df.loc[:, "neutral_sub"] = neutralize(df, [prediction_col],
                                          feature_cols)[prediction_col]
    scores = df.groupby("era").apply(
        lambda x: (unif(x["neutral_sub"]).corr(x[TARGET_COL]))).mean()
    return np.mean(scores)


def validation_metrics(validation_data, pred_cols, example_col):
    validation_stats = pd.DataFrame()
    feature_cols = [c for c in validation_data if c.startswith("feature_")]
    for pred_col in pred_cols:
        print(f"Doing validation metrics on: {pred_col}")
        # Check the per-era correlations on the validation set (out of sample)
        with Halo(text='Calculating correlations', spinner='dots'):
            validation_correlations = validation_data.groupby(ERA_COL).apply(
                lambda d: unif(d[pred_col]).corr(d[TARGET_COL]))
            mean = validation_correlations.mean()
            std = validation_correlations.std(ddof=0)
            sharpe = mean / std
            validation_stats.loc["mean", pred_col] = mean
            validation_stats.loc["std", pred_col] = std
            validation_stats.loc["sharpe", pred_col] = sharpe

        with Halo(text='Calculating max drawdown', spinner='dots'):
            rolling_max = (validation_correlations + 1).cumprod().rolling(window=9000,  # arbitrarily large
                                                                          min_periods=1).max()
            daily_value = (validation_correlations + 1).cumprod()
            max_drawdown = -((rolling_max - daily_value) / rolling_max).max()
            validation_stats.loc["max_drawdown", pred_col] = max_drawdown

        # Check the feature exposure of your validation predictions
        with Halo(text='Calculating feature exposure', spinner='dots'):
            max_per_era = validation_data.groupby(ERA_COL).apply(
                lambda d: d[feature_cols].corrwith(d[pred_col]).abs().max())
            max_feature_exposure = max_per_era.mean()
            validation_stats.loc["max_feature_exposure", pred_col] = max_feature_exposure

        # Check feature neutral mean
        with Halo(text='Calculating feature neutral mean', spinner='dots'):
            feature_neutral_mean = get_feature_neutral_mean(validation_data, pred_col)
            validation_stats.loc["feature_neutral_mean", pred_col] = feature_neutral_mean

        with Halo(text='Calculating MMC stats', spinner='dots'):
            print("calculating MMC stats...")
            # MMC over validation
            mmc_scores = []
            corr_scores = []
            for _, x in validation_data.groupby(ERA_COL):
                series = neutralize_series(unif(x[pred_col]), (x[example_col]))
                mmc_scores.append(np.cov(series, x[TARGET_COL])[0, 1] / (0.29 ** 2))
                corr_scores.append(unif(x[pred_col]).corr(x[TARGET_COL]))

            val_mmc_mean = np.mean(mmc_scores)
            val_mmc_std = np.std(mmc_scores)
            corr_plus_mmcs = [c + m for c, m in zip(corr_scores, mmc_scores)]
            corr_plus_mmc_sharpe = np.mean(corr_plus_mmcs) / np.std(corr_plus_mmcs)

            validation_stats.loc["mmc_mean", pred_col] = val_mmc_mean
            validation_stats.loc["corr_plus_mmc_sharpe", pred_col] = corr_plus_mmc_sharpe

        with Halo(text='Calculating correlation with example predictions', spinner='dots'):
            # Check correlation with example predictions
            per_era_corrs = validation_data.groupby(ERA_COL).apply(lambda d: unif(d[pred_col]).corr(unif(d["example_preds"])))
            corr_with_example_preds = per_era_corrs.mean()
            validation_stats.loc["corr_with_example_preds", pred_col] = corr_with_example_preds

    return validation_stats


def download_file(url: str, dest_path: str, show_progress_bars: bool = True):
    req = requests.get(url, stream=True)
    req.raise_for_status()

    # Total size in bytes.
    total_size = int(req.headers.get('content-length', 0))

    if os.path.exists(dest_path):
        logging.info("target file already exists")
        file_size = os.stat(dest_path).st_size  # File size in bytes
        if file_size < total_size:
            # Download incomplete
            logging.info("resuming download")
            resume_header = {'Range': 'bytes=%d-' % file_size}
            req = requests.get(url, headers=resume_header, stream=True,
                               verify=False, allow_redirects=True)
        elif file_size == total_size:
            # Download complete
            return
        else:
            # Error, delete file and restart download
            logging.error("deleting file and restarting")
            os.remove(dest_path)
            file_size = 0
    else:
        # File does not exist, starting download
        logging.info("starting download")

    with open(dest_path, "ab") as dest_file:
        for chunk in req.iter_content(1024):
            dest_file.write(chunk)


def download_data(napi, filename, dest_path, round=None):
    query = """
            query ($filename: String!) {
                dataset(filename: $filename)
            }
            """
    params = {
        'filename': filename
    }
    if round:
        query = """
                    query ($filename: String!, $round: Int) {
                        dataset(filename: $filename, round: $round)
                    }
                    """
        params['round'] = round
    dataset_url = napi.raw_query(query, params)['data']['dataset']
    download_file(dataset_url, dest_path, show_progress_bars=True)
    return dataset_url
