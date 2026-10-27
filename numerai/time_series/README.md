# Time-series features from Unity

This data building example implements the prompt:

> For every feature in the "medium" feature set and every ticker, using only that ticker's own past up to each date:
> 1. rank today's value within the ticker's last 52 weeks
> 2. take the 13-week average minus the 52-week average
> 3. count the weeks since the ticker's 52-week high
> 4. count the weeks since the value last changed

It downloads the latest available Classic Unity release and produces four numeric columns per numeric
medium feature, keyed by `numerai_ticker` and `date`. Like the [Kev earnings-release example](../../signals/edgar_kev/),
the output is a separate feature file that you can join to tournament data.
`feature_country` is categorical metadata and is excluded, as in the Unity model notebook.

## Run

From this folder:

```bash
pip install -r requirements.txt
python time_series.py --tickers 'AAPL US' 'MSFT US'  # start with two tickers
python time_series.py                              # every ticker
python time_series.py --data-version v6.0           # pin a release for reproducibility
```

Train, validation, and live observations are combined **before** calculating features, so the start of
validation and live retains each ticker's earlier history. Targets, data-type filters, and cross-sectional
ranks are not used. Historical downloads are cached under `data/<version>/`; live is refreshed each run.
The full run loads the medium features and builds an output four times as wide in memory, so it requires
substantial RAM. Use `--tickers` for a small example; it filters rows when reading the downloaded files.

The default output is `data/time_series_features.parquet`. Join it to a Unity file from the same release:

```python
import pandas as pd

features = pd.read_parquet("data/time_series_features.parquet")
live = pd.read_parquet("data/v6.0/live.parquet")  # use the version printed by the builder
live["date"] = pd.to_datetime(live["date"])
live = live.merge(features, on=["numerai_ticker", "date"], how="left", validate="one_to_one")
```

## Definitions

For a source column `feature_x`, the output columns are:

| Suffix | Calculation |
| --- | --- |
| `_rank_52w` | Today's percentile rank among this ticker's nonmissing observations in the last 52 weeks. Ties get their average rank; the range is `(0, 1]`. |
| `_mean_13w_minus_52w` | Mean over the last 13 weeks minus mean over the last 52 weeks. |
| `_weeks_since_high_52w` | Elapsed weeks since the most recent occurrence of the highest value in the last 52 weeks; zero for a high today. |
| `_weeks_since_change` | Elapsed weeks since the most recent observed change; zero on the date a new value is observed. |

Windows include today and exclude the date exactly 13 or 52 weeks earlier: `(date - window, date]`.
On regularly weekly data these contain 13 or 52 observations. Gaps do not stretch a window; ages use
elapsed calendar days divided by seven and may be fractional. A change across a gap is dated to the
first observation of the new value.

Rank, means, and high age use available history for new tickers. Change age is NaN until an actual
change is observed: the first value does not reveal how long it has been unchanged. A missing current
value produces four NaNs and breaks the known unchanged run; missing historical values are excluded
from rolling calculations. Conflicting duplicate ticker/date observations raise an error.

These are raw numeric features, not bins or predictions. This example makes no claim about their
predictive performance. For live inference, retain historical observations; a live file alone cannot
provide a ticker's trailing history.

Run the synthetic checks with `python -m unittest discover -s . -p 'test_*.py'`.
