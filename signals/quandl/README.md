This script uses Quandl's [end of day US stock prices](https://www.quandl.com/data/EOD-End-of-Day-US-Stock-Prices/) data. 

*Before running, install quandl's python package and numerapi*

```
pip install quandl
pip install numerapi
```

**`API_KEY = "<Quandl API KEY>"` This needs to be set before running the script.**

To get an API key, create a paid account on Quandl.

This example downloads the whole 'Time-series' data in a .zip file and loads Adj_Open and Adj_Close columns from it. However, specific tickers for specific time span can also be loaded iteratively using API(much slower). [Getting started with the API](https://www.quandl.com/data/EOD-End-of-Day-US-Stock-Prices/usage/quickstart/api).

While the feature extraction and modeling part are very similar to the main `example_model.py`, the focus here is to make the data loading flexible so different data sources can be easily 'plugged'.

**Steps to re-arrange the data as in Signals' main `example_script.py`**

1. Find common tickers between [EOD data source ticker list](https://s3.amazonaws.com/quandl-production-static/end_of_day_us_stocks/ticker_list.csv) and Numerai Signals Universe's yahoo tickers. 
2. Specify the columns in `download_full_and_load` with common tickers and rename columns as required by feature extraction setup.

```
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
```

3. Map ticker names to Bloomberg tickers using [Numerai's Bloomberg ticker map](https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/signals_ticker_map_w_bbg.csv).

```
    full_data = full_data[full_data.ticker.isin(common_tickers)]
    full_data["bloomberg_ticker"] = full_data.ticker.map(
        dict(zip(ticker_map["yahoo"], ticker_map["bloomberg_ticker"]))
     )
```

After creating a `day_chg` column and applying RSI and SMA on them, features are quintiled and lags are calculated as in main `example_model.py`.
