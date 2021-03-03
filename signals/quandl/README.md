This script uses Quandl's [End of Day US Stock Prices](https://www.quandl.com/data/EOD-End-of-Day-US-Stock-Prices/) data. (It's a premium data source)

Two ways to download the data

1. Load every ticker and its daily features sequentially. (slower)
2. Load entire dataset once and load data from it. (much faster)

This script provides functions for both ways of loading the data in same format.

*Need to install quandl's python package and numerapi*

```
pip install quandl
pip install numerapi
```

**`API_KEY = "<Quandl API KEY>"` This need to be set before running the script.**
