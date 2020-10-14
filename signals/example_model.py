import numerapi
# a fork of yfinance that implements retries nicely
# pip install -e git+http://github.com/leonhma/yfinance.git@master#egg=yfinance
import yfinance
import simplejson

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests as re
from datetime import datetime
from dateutil.relativedelta import relativedelta, FR
from sklearn.linear_model import LinearRegression


def RSI(prices, interval=10):
    '''Computes Relative Strength Index given a price series and lookback interval
  Modified from https://stackoverflow.com/questions/20526414/relative-strength-index-in-python-pandas
  See more here https://www.investopedia.com/terms/r/rsi.asp'''
    delta = prices.diff()

    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0

    RolUp = dUp.rolling(interval).mean()
    RolDown = dDown.rolling(interval).mean().abs()

    RS = RolUp / RolDown
    RSI = 100.0 - (100.0 / (1.0 + RS))
    return RSI


def main():
    '''Creates example_signal_upload.csv to upload for validation and live data submission'''
    napi = numerapi.SignalsAPI()

    # read in list of active Signals tickers which can change slightly era to era
    eligible_tickers = pd.Series(napi.ticker_universe(), name='bloomberg_ticker')
    print(f"Number of eligible tickers: {len(eligible_tickers)}")

    # read in yahoo to bloomberg ticker map, still a work in progress, h/t wsouza
    ticker_map = pd.read_csv(
        'https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/signals_ticker_map_w_bbg.csv'
    )
    print(f"Number of tickers in map: {len(ticker_map)}")

    # map eligible numerai tickers to yahoo finance tickers
    yfinance_tickers = eligible_tickers.map(
        dict(zip(ticker_map['bloomberg_ticker'], ticker_map['yahoo']))).dropna()
    bloomberg_tickers = ticker_map['bloomberg_ticker']
    print(f'Number of eligible, mapped tickers: {len(yfinance_tickers)}')

    # download data
    n = 1000  # chunk row size
    chunk_df = [
        yfinance_tickers.iloc[i:i + n]
        for i in range(0, len(yfinance_tickers), n)
    ]

    concat_dfs = []
    print("Downloading data...")
    for df in chunk_df:
        try:
            # set threads = True for faster performance, but tickers will fail, scipt may hang
            # set threads = False for slower performance, but more tickers will succeed
            temp_df = yfinance.download(df.str.cat(sep=' '),
                                        start='2005-12-01',
                                        threads=False)
            temp_df = temp_df['Adj Close'].stack().reset_index()
            concat_dfs.append(temp_df)
        except:  # simplejson.errors.JSONDecodeError:
            pass

    full_data = pd.concat(concat_dfs)

    # properly position and clean raw data, after taking adjusted close only
    full_data.columns = ['date', 'ticker', 'price']
    full_data.set_index('date', inplace=True)
    # convert yahoo finance tickers back to numerai tickers
    full_data['bloomberg_ticker'] = full_data.ticker.map(
        dict(zip(ticker_map['yahoo'], bloomberg_tickers)))
    print('Data downloaded.')
    print(f"Number of tickers with data: {len(full_data.bloomberg_ticker.unique())}")

    ticker_groups = full_data.groupby('ticker')
    full_data['RSI'] = ticker_groups['price'].transform(lambda x: RSI(x))

    # group by era (date) and create quintile labels within each era, useful for learning relative ranking
    date_groups = full_data.groupby(full_data.index)
    full_data['RSI_quintile'] = date_groups['RSI'].transform(
        lambda group: pd.qcut(group, 5, labels=False, duplicates='drop'))
    full_data.dropna(inplace=True)

    # create lagged features grouped by ticker
    ticker_groups = full_data.groupby('ticker')
    num_days = 5
    # lag 0 is that day's value, lag 1 is yesterday's value, etc
    for day in range(num_days + 1):
        full_data[f'RSI_quintile_lag_{day}'] = ticker_groups[
            'RSI_quintile'].transform(lambda group: group.shift(day))

    # create difference of the lagged features and absolute difference of the lagged features (change in RSI quintile by day)
    for day in range(num_days):
        full_data[f'RSI_diff_{day}'] = full_data[
                                           f'RSI_quintile_lag_{day}'] - full_data[
                                           f'RSI_quintile_lag_{day + 1}']
        full_data[f'RSI_abs_diff_{day}'] = np.abs(
            full_data[f'RSI_quintile_lag_{day}'] -
            full_data[f'RSI_quintile_lag_{day + 1}'])

    # define column names of features, target, and prediction
    feature_names = [f'RSI_quintile_lag_{num}' for num in range(num_days)] + [
        f'RSI_diff_{num}' for num in range(num_days)
    ] + [f'RSI_abs_diff_{num}' for num in range(num_days)]
    print(f'Features for training:\n {feature_names}')

    TARGET_NAME = 'target'
    PREDICTION_NAME = 'signal'

    # read in Signals targets
    targets = pd.read_csv('historical_targets.csv')
    targets['date'] = pd.to_datetime(targets['friday_date'], format='%Y%m%d')

    # merge our feature data with Numerai targets
    ML_data = pd.merge(full_data.reset_index(), targets,
                       on=['date', 'bloomberg_ticker']).set_index('date')
    # print(f'Number of eras in data: {len(ML_data.index.unique())}')

    # for training and testing we want clean, complete data only
    ML_data.dropna(inplace=True)
    ML_data = ML_data[ML_data.index.weekday ==
                      4]  # ensure we have only fridays
    ML_data = ML_data[ML_data.index.value_counts() >
                      50]  # drop eras with under 50 observations per era

    # train test split
    train_data = ML_data[ML_data['data_type'] == 'train']
    test_data = ML_data[ML_data['data_type'] == 'validation']

    # train model
    print("Training model...")
    model = LinearRegression()
    model.fit(train_data[feature_names], train_data[TARGET_NAME])
    print("Model trained.")

    # predict test data
    test_data[PREDICTION_NAME] = model.predict(test_data[feature_names])

    # predict live data
    # choose data as of most recent friday
    last_friday = datetime.now() + relativedelta(weekday=FR(-1))
    date_string = last_friday.strftime('%Y-%m-%d')

    live_data = full_data.loc[date_string].copy()
    live_data.dropna(subset=feature_names, inplace=True)
    print(f"Number of live tickers to submit: {len(live_data)}")
    live_data[PREDICTION_NAME] = model.predict(live_data[feature_names])

    # prepare and writeout example file
    diagnostic_df = pd.concat([test_data, live_data])
    diagnostic_df['friday_date'] = diagnostic_df.friday_date.fillna(
        last_friday.strftime('%Y%m%d')).astype(int)
    diagnostic_df['data_type'] = diagnostic_df.data_type.fillna('live')
    diagnostic_df[['bloomberg_ticker', 'friday_date', 'data_type',
                   'signal']].reset_index(drop=True).to_csv(
        'example_signal_upload.csv', index=False)
    print(
        'Example submission completed. Upload to signals.numer.ai for scores and live submission'
    )


if __name__ == '__main__':
    main()
