This script uses data from [AlphaVantage](https://www.alphavantage.co/) using raw queries which do not require any extra library.
```
pip install numerapi tqdm
```
`numerapi` and `tqdm` (to check progress of downloads) are needed.

`key = "<ALPHAVANTAGE API KEY>"` **this needs to be set before running the script.**

You can generate a premium key by subscribing to [premium membership](https://www.alphavantage.co/premium/). This script was created with the lowest 75 calls/minute plan so it would work. This takes around 13 mins to load all US stocks.

Alpha Vantage offers various typed of data including Stocks, cryptocurrencies, and technical indicators. [Documentation](https://www.alphavantage.co/documentation/).

### Data Loading

Since the API has some rate limits of fixed calls/minute, this script provides two functions for loading the data.

- Sequential 
    + This may take over an hour to load all available tickers.
    + Safe and loading function can be stopped.
    + `full_data = load_data(tickers, "full_data.csv", threads=False)`

- Parallel
    + uses threads from Python's `concurrent` module.
    + much faster loading (~13 mins for all US tckers)
    + execution is not stoppable.
    + `full_data = load_data(tickers, "full_data.csv", threads=True)`
    + Thanks to **[Jordi Villar](https://twitter.com/jrdi)** for supercharging the parallel execution and bringing it to ~13 mins.

Curently available tickers from the universe using yahoo tickers are in `tickers.csv.txt` file. 

`tickers = pd.read_csv("tickers.csv.txt")`

### Feature Generation

- Since Alpha vantage has differnet types of data resolutions, this script uses **weekly data** (OHLC) to generate features.
- Some features are `simple moveing average` and `exponential moving average` of periods `[2, 5, 21, 50, 200]`.


