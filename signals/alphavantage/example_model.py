import concurrent.futures as _futures
import gc
import itertools
import json
import os
import random
import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numerapi
import numpy as np
import pandas as pd
import requests
from dateutil.relativedelta import FR, relativedelta
from sklearn.ensemble import GradientBoostingRegressor
from tqdm.auto import tqdm

#https://www.alphavantage.co/premium/

key = "<ALPHAVANTAGE API KEY>"  # $50 USD, 75 calls per minute subscription will work
BASE_URL = "https://www.alphavantage.co/query"

TARGET_NAME = "target"
PREDICTION_NAME = "signal"


def get_daily_ts_adj(ticker, output_size="full", data_type="csv", backoff=0) -> pd.DataFrame:
    """Loads one ticker in csv format"""
    time.sleep(backoff)

    # WEEKLY
    function = "TIME_SERIES_WEEKLY_ADJUSTED"
    url = (BASE_URL+ f"?function={function}&symbol={ticker}&apikey={key}&datatype={data_type}")

    data = pd.read_csv(url)
    data["ticker"] = ticker

    if "targeting a higher API call" in data.loc[0][0]:
        # retrying...
        return get_daily_ts_adj(
            ticker, output_size, data_type,
            min(60, backoff + random.choice(range(1, 10))),
        )

    return data


def get_tickers_sequential(tickers) -> pd.DataFrame:
    """Loads a list of tickers sequentially"""

    dfs = []
    for ticker in tqdm(tickers):
        response = get_daily_ts_adj(ticker)
        response["ticker"] = ticker
        dfs.append(response)

    return pd.concat(dfs)


def get_tickers_parallel(tickers) -> pd.DataFrame:
    n = 70  # Setting n<75 for extra safety
    chunks = [tickers[i : i + n] for i in range(0, len(tickers), n)]

    res = []
    pbar = tqdm(total=len(tickers))
    with _futures.ThreadPoolExecutor() as executor:
        for chunk in chunks:
            futures = []
            for i, ticker in enumerate(chunk):
                futures.append(executor.submit(get_daily_ts_adj, ticker=ticker))
            for future in _futures.as_completed(futures):
                try:
                    response = future.result()
                    if len(response) < 5:
                        pbar.update(1)
                        continue
                    else:
                        res.append(response)
                        pbar.update(1)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    pbar.update(1)
                    continue

    return pd.concat(res)


def load_data(tickers, f_path="full_data.csv", threads=False) -> pd.DataFrame:
    if os.path.exists(f_path):
        data = pd.read_csv(f_path)
        data = data.loc[data.ticker.isin(tickers)]
    else:
        if threads:
            data = get_tickers_parallel(tickers)
        else:
            data = get_tickers_sequential(tickers)
        data.to_csv(f_path)

    data.set_index("timestamp", inplace=True)
    data.index.rename("date", inplace=True)
    data.index = pd.to_datetime(data.index)

    return data

def generate_featues(full_data: pd.DataFrame) -> pd.DataFrame:
    ticker_groups = full_data.groupby("bloomberg_ticker")
    indicators = []
    sma_periods = [2, 5, 21, 50, 200]
    ema_periods = [2, 5, 21, 50, 200]

    for period in sma_periods:
        full_data[f"close_SMA_{period}"] = ticker_groups["adjusted close"].transform(
            lambda x: x.rolling(period).mean()
        )
        indicators.append(f"close_SMA_{period}")

    for period in ema_periods:
        full_data[f"close_EMA_{period}"] = ticker_groups["adjusted close"].transform(
            lambda x: x.ewm(span=period).mean()
        )
        indicators.append(f"close_EMA_{period}")

    full_data.dropna(inplace=True, axis=0)

    date_groups = full_data.groupby(full_data.index)
    for indicator in tqdm(indicators):
        gc.collect()
        full_data.loc[:, f"{indicator}_quintile"] = (
            date_groups[indicator]
            .apply(lambda group: pd.qcut(group, 5, labels=False, duplicates="drop"))
            .astype(np.float16)
        )
        gc.collect()
    del date_groups
    gc.collect()

    return full_data


def main():
    napi = numerapi.SignalsAPI()

    # Numerai Universe
    eligible_tickers = pd.Series(napi.ticker_universe(), name="bloomberg_ticker")
    print(f"Number of eligible tickers : {len(eligible_tickers)}")

    ticker_map = pd.read_csv(
        "https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/signals_ticker_map_w_bbg.csv"
    )

    # ----- Yahoo <-> Bloomberg mapping -----
    yfinance_tickers = eligible_tickers.map(
        dict(zip(ticker_map["bloomberg_ticker"], ticker_map["yahoo"]))
    ).dropna()
    bloomberg_tickers = ticker_map["bloomberg_ticker"]
    print(f"Number of eligible, mapped tickers: {len(yfinance_tickers)}")

    us_ticker_map = ticker_map[ticker_map.bloomberg_ticker.str[-2:] == "US"]
    #tickers = us_ticker_map.yahoo.dropna().values #for US tickers
    tickers = ticker_map.yahoo.dropna().values #For possible tickers

    # ----- Raw data loading and formatting -----
    print(f"using tickers: {len(tickers)}")
    full_data = load_data(tickers, "full_data.csv", threads=True)

    full_data["bloomberg_ticker"] = full_data.ticker.map(
        dict(zip(ticker_map["yahoo"], bloomberg_tickers))
    )

    full_data = full_data[
        ["bloomberg_ticker", "open", "high", "low", "close", "adjusted close"]
    ].sort_index(ascending=True)
    full_data.dropna(inplace=True, axis=0)

    # ----- Merging targets -----
    url = "https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/signals_train_val_bbg.csv"
    targets = pd.read_csv(url)

    targets["target"] = targets["target"].astype(np.float16)
    targets["date"] = pd.to_datetime(targets["friday_date"], format="%Y%m%d")
    gc.collect()

    # ----- Generate and select features -----
    full_data = generate_featues(full_data)
    feature_names = [f for f in full_data.columns if "quintile" in f]

    ML_data = pd.merge(
        full_data.reset_index(), targets, on=["date", "bloomberg_ticker"],
    ).set_index("date")
    print(f"Number of eras in data: {len(ML_data.index.unique())}")

    ML_data = ML_data[ML_data.index.weekday == 4]
    ML_data = ML_data[ML_data.index.value_counts() > 200]

    # ----- Train test split -----
    train_data = ML_data[ML_data["data_type"] == "train"]
    test_data = ML_data[ML_data["data_type"] == "validation"]

    corrs = train_data.groupby(train_data.index).apply(
        lambda x: x[feature_names+[TARGET_NAME]].corr()[TARGET_NAME]
    )
    mean_corr = corrs[feature_names].mean(0)
    print(mean_corr)

    last_friday = datetime.now() + relativedelta(weekday=FR(-1))
    print(last_friday)
    date_string = last_friday.strftime("%Y-%m-%d")

    try:
        live_data = full_data.loc[date_string].copy()
    except KeyError as e:
        print(f"No ticker on {e}")
        live_data = full_data.iloc[:0].copy()
    live_data.dropna(subset=feature_names, inplace=True)
    print(len(live_data))
    # ----- Train model -----
    print("Training model...")
    model = GradientBoostingRegressor()
    model.fit(train_data[feature_names], train_data[TARGET_NAME])
    print("Model trained.")

    # ----- Predict test data -----
    train_data[PREDICTION_NAME] = model.predict(train_data[feature_names])
    test_data[PREDICTION_NAME] = model.predict(test_data[feature_names])
    live_data[PREDICTION_NAME] = model.predict(live_data[feature_names])

    diagnostic_df = pd.concat([test_data, live_data])
    diagnostic_df["friday_date"] = diagnostic_df.friday_date.fillna(
        last_friday.strftime("%Y%m%d")
    ).astype(int)
    diagnostic_df["data_type"] = diagnostic_df.data_type.fillna("live")
    diagnostic_df[
        ["bloomberg_ticker", "friday_date", "data_type", "signal"]
    ].reset_index(drop=True).to_csv("example_signal_alphavantage.csv", index=False)
    print(
        "Submission saved to example_signal_alphavantage.csv. Upload to signals.numer.ai for scores and diagnostics"
    )


if __name__ == "__main__":
    main()
