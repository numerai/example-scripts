import datetime
import gc
import logging
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numerapi
import numpy as np
import pandas as pd
import quandl
import requests
from dateutil.relativedelta import FR, relativedelta
from sklearn.ensemble import GradientBoostingRegressor

logging.basicConfig()

quandl_log = logging.getLogger("quandl")
quandl_log.setLevel(logging.DEBUG)

API_KEY = "<Quandl API KEY>"
quandl.ApiConfig.api_key = API_KEY

# -----Tickers and mapping-----
napi = numerapi.SignalsAPI()
eligible_tickers = pd.Series(napi.ticker_universe(), name="bloomberg_ticker")

ticker_map = pd.read_csv(
    "https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/signals_ticker_map_w_bbg.csv"
)
ticker_map = ticker_map[ticker_map.bloomberg_ticker.isin(eligible_tickers)]

numerai_tickers = ticker_map["bloomberg_ticker"]
yfinance_tickers = ticker_map["yahoo"]

eod_tickers = pd.read_csv(
    "https://s3.amazonaws.com/quandl-production-static/end_of_day_us_stocks/ticker_list.csv"
)
print(f"Number of eligible tickers : {len(eligible_tickers)}")

common_tickers = np.intersect1d(
    yfinance_tickers.values.astype(str), eod_tickers["Ticker"].values.astype(str)
)
print(f"Number of tickers common between EOD and Bloomberg: {len(common_tickers)}")

# -----Helper functions for feature extraction-----
def RSI(prices, interval=14):
    """Computes Relative Strength Index given a price series and lookback interval
    Modified from https://stackoverflow.com/questions/20526414/relative-strength-index-in-python-pandas
    See more here https://www.investopedia.com/terms/r/rsi.asp"""
    delta = prices.diff()

    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0

    RolUp = dUp.rolling(interval).mean()
    RolDown = dDown.rolling(interval).mean().abs()

    RS = RolUp / RolDown
    RSI = 100.0 - (100.0 / (1.0 + RS))
    return RSI


# -----Data loading function-----
def download_full_and_load(f_name: str = "full_EOD.zip") -> pd.DataFrame:
    """Downloads a zip of entire dataset and load csv from it.
    Much faster!
    """

    url = f"https://www.quandl.com/api/v3/databases/EOD/data?api_key={API_KEY}"

    if os.path.exists(f_name):
        print(f"Using downloaded file {f_name}")
    else:
        print("Downloading data...")
        with requests.get(url, stream=True) as r:
            with open(f_name, "wb") as fin_data:
                for chunk in r.iter_content(chunk_size=1024):
                    fin_data.write(chunk)
            print(f"Saved as: {f_name}")

    # column names in the csv file without headers
    cols = [
        "ticker", "date", "Open", "High", "Low", "Close", "Volume", "Dividend",
        "Split", "Adj_Open", "Adj_High", "Adj_Low", "Adj_Close", "Adj_Volume",
    ]

    # usecols refers to the column in the csv.
    # using only [ticker, date, adj_open, adj_close]
    # Loading only needed columns as FP32
    print("loading from csv...")
    full_data = pd.read_csv(
        f_name,
        usecols=[0, 1, 9, 12],
        compression="zip",
        dtype={0: str, 1: str, 9: np.float32, 12: np.float32},
        header=None,
    )

    # renaming the columns
    filter_columns = ["ticker", "date", "Adj_Open", "Adj_Close"]
    full_data.columns = filter_columns
    full_data.set_index("date", inplace=True)
    full_data.index = pd.to_datetime(full_data.index)

    full_data.rename(
        columns={
            "Adj_Open": "open",
            "Adj_Close": "close",
        },
        inplace=True,
    )

    full_data[["open", "close"]] = full_data[["open", "close"]].astype(np.float32)
    full_data = full_data[full_data.ticker.isin(common_tickers)]
    full_data["bloomberg_ticker"] = full_data.ticker.map(
        dict(zip(ticker_map["yahoo"], numerai_tickers))
    )
    full_data.sort_index(ascending=True, inplace=True)
    gc.collect()

    return full_data


def main():
    # downloads the whole dataset as zip and read data (takes around 1.5min)
    full_data = download_full_and_load(f_name="full_EOD.zip")

    # Building a custom feature
    full_data["day_chg"] = full_data["close"] / full_data["open"] - 1
    gc.collect()

    # -----Feature engineering-----
    ticker_groups = full_data.groupby("bloomberg_ticker")

    # RSI
    full_data["close_RSI_14"] = ticker_groups["close"].transform(lambda x: RSI(x, 14))
    full_data["close_RSI_21"] = ticker_groups["close"].transform(lambda x: RSI(x, 21))
    full_data["day_chg_RSI_14"] = ticker_groups["day_chg"].transform(
        lambda x: RSI(x, 14)
    )
    full_data["day_chg_RSI_21"] = ticker_groups["day_chg"].transform(
        lambda x: RSI(x, 21)
    )

    # SMA
    full_data["close_SMA_14"] = ticker_groups["close"].transform(
        lambda x: x.rolling(14).mean()
    )
    full_data["close_SMA_21"] = ticker_groups["close"].transform(
        lambda x: x.rolling(21).mean()
    )

    indicators = ["close_RSI_14", "close_RSI_21", "day_chg_RSI_14",
                  "close_SMA_14", "close_SMA_21", "day_chg_RSI_21"]

    full_data.dropna(axis=0, inplace=True)
    del ticker_groups

    # -----Feature engineering: Quintile-----
    date_groups = full_data.groupby(full_data.index)
    print("Quintiling...")
    for indicator in indicators:
        full_data[f"{indicator}_quintile"] = (
            date_groups[indicator]
            .transform(
                lambda group: pd.qcut(group, 100, labels=False, duplicates="drop")
            )
            .astype(np.float16)
        )
        gc.collect()

    del date_groups
    gc.collect()

    # -----Feature engineering: Quintile lag-----
    ticker_groups = full_data.groupby("ticker")
    # create lagged features, lag 0 is that day's value, lag 1 is yesterday's value, etc
    print("Calculating lag...")
    for indicator in indicators:
        num_days = 5
        for day in range(num_days + 1):
            full_data[f"{indicator}_quintile_lag_{day}"] = ticker_groups[
                f"{indicator}_quintile"
            ].transform(lambda group: group.shift(day))

        gc.collect()

    full_data.dropna(axis=0, inplace=True)

    del ticker_groups
    gc.collect()
    print("Calculating changes in lag...")
    # create difference of the lagged features (change in RSI quintile by day)
    for indicator in indicators:
        for day in range(0, num_days):
            full_data[f"{indicator}_diff_{day}"] = (
                full_data[f"{indicator}_quintile_lag_{day}"]
                - full_data[f"{indicator}_quintile_lag_{day + 1}"]
            ).astype(np.float16)
            gc.collect()

    # create difference of the lagged features (change in RSI quintile by day)
    for indicator in indicators:
        full_data[f"{indicator}_abs_diff_{day}"] = np.abs(
            full_data[f"{indicator}_quintile_lag_{day}"]
            - full_data[f"{indicator}_quintile_lag_{day + 1}"]
        ).astype(np.float16)
        gc.collect()

    TARGET_NAME = "target"
    PREDICTION_NAME = "signal"

    # read in Signals targets
    numerai_targets = "https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/signals_train_val_bbg.csv"
    targets = pd.read_csv(numerai_targets)
    targets["date"] = pd.to_datetime(targets["friday_date"], format="%Y%m%d")

    # merge our feature data with Numerai targets
    ML_data = pd.merge(
        full_data.reset_index(), targets, on=["date", "bloomberg_ticker"]
    ).set_index("date")
    print(f"Number of eras in data: {len(ML_data.index.unique())}")

    # for training and testing we want clean, complete data only
    ML_data.dropna(inplace=True)
    ML_data = ML_data[ML_data.index.weekday == 4]  # ensure we have only fridays
    ML_data = ML_data[
        ML_data.index.value_counts() > 200
    ]  # drop eras with under 200 observations per era
    feature_names = [f for f in ML_data.columns for y in ["lag", "diff"] if y in f]
    print(f"Using {len(feature_names)} features")

    last_friday = datetime.now() + relativedelta(weekday=FR(-1))
    date_string = last_friday.strftime("%Y-%m-%d")

    try:
        live_data = full_data.loc[date_string].copy()
    except KeyError as e:
        print(f"No ticker on {e}")
        live_data = full_data.iloc[:0].copy()
    live_data.dropna(subset=feature_names, inplace=True)

    # get data from the day before, for markets that were closed
    # on the most recent friday
    last_thursday = last_friday - timedelta(days=1)
    thursday_date_string = last_thursday.strftime("%Y-%m-%d")
    thursday_data = full_data.loc[thursday_date_string]
    # Only select tickers than aren't already present in live_data
    thursday_data = thursday_data[
        ~thursday_data.ticker.isin(live_data.ticker.values)
    ].copy()
    thursday_data.dropna(subset=feature_names, inplace=True)

    live_data = pd.concat([live_data, thursday_data])

    # train test split
    train_data = ML_data[ML_data["data_type"] == "train"].copy()
    test_data = ML_data[ML_data["data_type"] == "validation"].copy()

    train_data[feature_names] /= 100.0
    test_data[feature_names] /= 100.0
    live_data[feature_names] /= 100.0

    del ML_data
    gc.collect()

    # train model
    print("Training model...")
    model = GradientBoostingRegressor(n_estimators=50)
    model.fit(train_data[feature_names], train_data[TARGET_NAME])
    print("Model trained.")

    # predict test data
    train_data[PREDICTION_NAME] = model.predict(train_data[feature_names])
    test_data[PREDICTION_NAME] = model.predict(test_data[feature_names])

    print(f"Number of live tickers to submit: {len(live_data)}")
    live_data[PREDICTION_NAME] = model.predict(live_data[feature_names])

    # prepare and writeout example file
    diagnostic_df = pd.concat([test_data, live_data])
    diagnostic_df["friday_date"] = diagnostic_df.friday_date.fillna(
        last_friday.strftime("%Y%m%d")
    ).astype(int)
    diagnostic_df["data_type"] = diagnostic_df.data_type.fillna("live")
    diagnostic_df[
        ["bloomberg_ticker", "friday_date", "data_type", "signal"]
    ].reset_index(drop=True).to_csv("example_quandl_signal_upload.csv", index=False)
    print(
        "Example submission completed. Upload to signals.numer.ai for scores and live submission"
    )


if __name__ == "__main__":
    main()
