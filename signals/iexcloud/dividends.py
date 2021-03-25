import os

from iexfinance import stocks
from iexfinance.utils.exceptions import IEXQueryError
import numerapi
import pandas as pd
from tqdm import tqdm


SANDBOX = True

if SANDBOX:
    os.environ['IEX_TOKEN'] = os.environ['IEX_TOKEN_SANDBOX']
    os.environ['IEX_API_VERSION'] = 'iexcloud-sandbox'

try:
    napi = numerapi.SignalsAPI(
        public_id=os.environ['NUMERAI_PUBLIC_ID'],
        secret_key=os.environ['NUMERAI_SECRET_KEY']
    )
except:
    napi = numerapi.SignalsAPI()

map = pd.read_parquet('signals/iexcloud/extended_map_us.parquet')

# TODO: use proper INFO logging instead of print (in line with numerapi's logging)
print('retrieving dividends data from iex cloud for all mapped symbols...')

data = []
not_found = []
for _, symbol, bloomberg_ticker in tqdm(list(map[['symbol', 'bloomberg_ticker']].itertuples())):
    try:
        dividends = stocks.Stock(symbol).get_dividends(range='5y')
        if not dividends.empty:
            # only continue if dividends data was retrieved
            dividends.index = pd.to_datetime(dividends.index)
            # how often per year are dividends payed out?
            dividends['frequency'] = dividends['frequency'].map({
                'annual': 1,
                'quarterly': 4,
                'monthly': 12,
                'weekly': 52,
                'daily': 365,
            })
            prices = stocks.get_historical_data(symbols=symbol, start=dividends.index.min(), close_only=True)
            # resample both prices and dividends to daily time series
            # last known price/dividends is always assumed best truth, hence ffill
            joined = prices.resample('D').ffill().join(dividends.resample('D').ffill()).ffill()
            # calculate the daily dividends' apy of the stock
            joined['div_apy'] = joined['amount'] * joined['frequency'] / joined['close']

            joined['bloomberg_ticker'] = bloomberg_ticker
            data.append(joined[['bloomberg_ticker', 'div_apy']])
    except IEXQueryError:
        not_found.append(symbol)
full_data = pd.concat(data).sort_index()

if len(not_found) > 0:
    print(f'could not retrieve following symbols: {not_found}')

# transform the daily apy to a quintile (so either 0, .25, .5, .75 or 1)
print(f"converting to quintile signal for {full_data['bloomberg_ticker'].unique().size} symbols...")
full_data['signal'] = (full_data
    .groupby(full_data.index)['div_apy']
    .transform(lambda date: pd.qcut(date, 5, labels=False, duplicates='drop')) / 4
)

# read in numerai signals' targets
try:
    targets = pd.read_csv('historical_targets.csv')
except FileNotFoundError:
    napi.download_validation_data(dest_filename='historical_targets.csv')
    targets = pd.read_csv('historical_targets.csv')
targets['friday_date'] = pd.to_datetime(targets['friday_date'], format='%Y%m%d')
targets = targets.rename({'friday_date': 'date'}, axis=1)

# covert to time series consisting of only Fridays
# TODO: compare Friday's signal (asfreq) with other weekly aggregates
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
preds = pd.merge(data_for_merge, targets, how='left', on=['date', 'bloomberg_ticker'])
# define live targets (last known Friday)
preds.loc[preds.index[-1], 'data_type'] = 'live'
# drop everything else that could not be merged
preds[~preds['data_type'].isna()]
print(f"successfully merged {preds['bloomberg_ticker'].unique().size} symbols")

# build submission, write to file and upload
submission = (preds
    .rename({'date': 'friday_date'}, axis=1)
    [['bloomberg_ticker', 'friday_date', 'data_type', 'signal']]
).to_csv('submission.csv', index=False)
napi.upload_predictions('submission.csv', model_id=napi.get_models()['gosuto_test'])

# TODO:
# what is corr of signal with target?
# is there really a signal or is it neutralised to 0?
# can same diagostics be performed offline as is done online?
