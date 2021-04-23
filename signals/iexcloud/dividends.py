import os
import time

import numerapi
import numpy as np
import pandas as pd
from iexfinance import stocks
from iexfinance.utils.exceptions import IEXQueryError
from tqdm.auto import tqdm


SANDBOX = True
os.environ['IEX_TOKEN'] = 'XXXXXXXXX' if SANDBOX else 'XXXXXXXXX'
os.environ['IEX_API_VERSION'] = 'iexcloud-sandbox' if SANDBOX else 'stable'
os.environ['NUMERAI_PUBLIC_ID'] = 'XXXXXXXXX'
os.environ['NUMERAI_SECRET_KEY'] = 'XXXXXXXXX'

# replace with your own Numerai Signals model name
MODEL_NAME = 'gosuto_test'

napi = numerapi.SignalsAPI()

universe = pd.DataFrame({'bloomberg_ticker': napi.ticker_universe()})
universe[['ticker', 'region']] = universe['bloomberg_ticker'].str.split(' ', n=2, expand=True)
universe = universe[universe['region'] == 'US']

def get_stock_dividends(symbol, range):
    try:
        dividends = stocks.Stock(symbol).get_dividends(range=range)
    except ConnectionError as e:
        print(e)
        print('retrying after 30 seconds...')
        time.sleep(31)
        get_stock_dividends(symbol, range)
    return dividends


print("retrieving all known stocks with dividends data in region US...")
data = []
not_found = []
for _, symbol, bloomberg_ticker in tqdm(list(universe[['ticker', 'bloomberg_ticker']].itertuples())):
    try:
        # use a range of up to 1y for extended backtesting of the validation set
        dividends = get_stock_dividends(symbol, range='1y')
        if not dividends.empty:
            dividends.index = pd.to_datetime(dividends.index)
            # NOTE: Stocks that used to payout dividends but do not anymore are not filtered out
            # how often per year are dividends payed out?
            dividends['frequency_int'] = dividends['frequency'].str.strip().map({
                'annual': 1,
                'semi-annual': 2,
                'quarterly': 4,
                'monthly': 12,
                'weekly': 52,
                'daily': 365,
            })
            prices = stocks.get_historical_data(symbols=symbol, start=dividends.index.min(), close_only=True)
            # resample both prices and dividends to daily time series
            # last known price/dividends is always assumed best truth, hence ffill
            joined = prices.resample('D').ffill().join(dividends.resample('D').ffill()).ffill()
            # calculate the dividend apy of the stock
            joined['div_apy'] = joined['amount'] * joined['frequency_int'] / joined['close']

            joined['bloomberg_ticker'] = bloomberg_ticker
            data.append(joined[['bloomberg_ticker', 'div_apy']])
    except IEXQueryError:
        not_found.append(symbol)
full_data = pd.concat(data).sort_index()

if len(not_found) > 0:
    print(f'could not retrieve the following symbols: {not_found}')

print(f"retrieved usable dividend data for {full_data['bloomberg_ticker'].nunique()} symbols")

full_data['signal'] = (full_data
    .groupby(full_data.index)['div_apy']
    .transform(lambda date: pd.qcut(date, 5, labels=False, duplicates='drop')) / 4
)

# read in numerai signals' targets
try:
    targets = pd.read_csv('numerai_signals_historical.csv')
except FileNotFoundError:
    napi.download_validation_data()
    targets = pd.read_csv('numerai_signals_historical.csv')
targets['friday_date'] = pd.to_datetime(targets['friday_date'], format='%Y%m%d')
targets = targets.rename({'friday_date': 'date'}, axis=1).set_index('date')

# covert to time series consisting of only Fridays
data_for_merge = (full_data
    .groupby('bloomberg_ticker')
    .apply(lambda x: x.asfreq('W-FRI'))
    .droplevel(0)
    .sort_index()
    .reset_index()
    .rename({'index': 'date'}, axis=1)
    .drop('div_apy', axis=1)
    .dropna()
)

# merge with targets
print("merging with numerai's targets...")
preds = pd.merge(data_for_merge, targets, how='left', on=['date', 'bloomberg_ticker']).set_index('date')
# define live targets (last known Friday)
preds.loc[preds.index[-1], 'data_type'] = 'live'
# drop everything else that could not be merged
preds = preds[~preds['data_type'].isna()]
print(f"successfully merged {preds['bloomberg_ticker'].unique().size} symbols")

# a quick look at the mean correlation of the signal with the validation target
print(f"mean correlation with target: {preds[preds['data_type'] == 'validation'][['signal', 'target']].corr()['target'][0]}")

# build submission, write to file and upload
submission = (preds
    .reset_index()
    .rename({'date': 'friday_date'}, axis=1)
    [['bloomberg_ticker', 'friday_date', 'data_type', 'signal']]
).to_csv('submission.csv', index=False)
napi.upload_predictions('submission.csv', model_id=napi.get_models()[MODEL_NAME])
