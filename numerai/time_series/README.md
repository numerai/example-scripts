# Time-series features from Unity

This data building example implements the prompt:

> Let's build new features from the Unity data.
>
> For every feature in the "medium" feature set and every ticker, using only that ticker's own past up to each date:
> 1. rank today's value within the ticker's last 52 weeks
> 2. take the 13-week average minus the 52-week average
> 3. count the weeks since the ticker's 52-week high
> 4. count the weeks since the value last changed
>
> Compute all dates at once, rank and bin each new feature 0–4 per date like the Unity features, and join them on numerai_ticker and date.

It downloads the latest available Classic Unity release and produces four binned columns per numeric
medium feature, keyed by `numerai_ticker` and `date`. Like the [Kev earnings-release example](../../signals/edgar_kev/),
the output is a separate feature file that you can join to tournament data.
`feature_country` is categorical metadata and is excluded, as in the Unity model notebook.

## Run

From this folder:

```bash
pip install -r requirements.txt
python time_series.py --tickers 'AAPL US' 'MSFT US'  # smoke test on two tickers
python time_series.py                              # every ticker, for per-date bins
python time_series.py --data-version v6.0           # pin a release for reproducibility
```

Train, validation, and live observations are combined **before** calculating features, so the start of
validation and live retains each ticker's earlier history. The four transforms use only that ticker's
past through the current date; they are then ranked and binned across tickers within each date. Targets
and data-type filters are not used. Historical downloads are cached under `data/<version>/`; live is refreshed each run.
The full run loads the medium features and builds an output four times as wide in memory, so it requires
substantial RAM. Use `--tickers` for a small example; it filters rows when reading the downloaded files.
Bins depend on the tickers included in the run, so use the full universe when building training features.

The default output is `data/time_series_features.parquet`. Join it to a Unity file from the same release:

```python
import pandas as pd

features = pd.read_parquet("data/time_series_features.parquet")
live = pd.read_parquet("data/v6.0/live.parquet")  # use the version printed by the builder
live["date"] = pd.to_datetime(live["date"])
live = live.merge(features, on=["numerai_ticker", "date"], how="left", validate="one_to_one")
```

## Definitions

For a source column `feature_x`, the following raw transforms are computed before binning:

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

Rank, mean difference, and high age remain NaN until at least 52 calendar weeks have elapsed since the
feature's first nonmissing observation and the trailing 52-week window contains at least 52 nonmissing
observations. The mean difference also requires 13 nonmissing observations in the 13-week window.
This deliberately leaves new listings and incomplete weekly histories missing.
Change age is NaN until an actual
change is observed: the first value does not reveal how long it has been unchanged. A missing current
value produces four NaNs and breaks the known unchanged run; missing historical values are excluded
from rolling calculations. Conflicting duplicate ticker/date observations raise an error.

Each raw transform is ranked across tickers **within its date** using average ranks for ties, then
binned with `ceil(percentile_rank * 5) - 1`. The saved columns keep the suffixes above but contain
Unity-style values 0–4; missing values stay missing and equal values stay tied. The first rank is a
historical comparison within a ticker; this second rank puts the resulting feature on a common scale
across tickers. For live inference, retain historical observations; a live file alone cannot provide a
ticker's trailing history.

## Tips for the agent

- **Only look back.** Every window ends at the current date. Ranking against a stock's full history leaks the future.
- **Respect short histories.** A new listing has no 52-week past. Leave those values missing rather than inventing one.
- **Rank and bin per date.** Put new features on the same 0–4 scale as Unity.
- **Pick different questions, not more windows.** The same transform at 52 and 156 weeks is mostly a copy. A few distinct ideas on fixed horizons beat a grid of variants.
- **Score what pays.** Judge new features on NCORR and neutral BMC against the benchmark, not on how they look alone. Compare out-of-sample predictions with and without the new features on the same dates; use the target, neutralizers, and public benchmark from the same data release. The [Unity notebook](../v6_example_model.ipynb) shows NCORR and neutral contribution against the benchmark. Benchmark contribution is a proxy, not official NMMC against the Meta Model.
- **Transforms aren't all you need.** Transforms re-express what Unity already knows. They can make your model different from everyone else's, which is what NMMC rewards, but they cannot add new source information.
- **To raise NCORR, bring new information:** filings, news, alternative data, like the [EDGAR features](../../signals/edgar_kev/) from the Unity announcement. If your agents can find it and join it on `numerai_ticker` and `date`, using only information available at that date, it becomes a feature, and its history gives you four more. Evaluate whether it improves NCORR; improvement is not guaranteed.

The feature builder does not measure predictive performance; use the paired model comparison below.

Run the synthetic checks with `python -m unittest discover -s . -p 'test_*.py'`.

## Reproduce the model comparison

The [recorded results and provenance](reproduction_results/README.md) include both sampled and
full-training comparisons. Neither reproduces the supplied post's numbers to four decimals. With all
training rows, original-period neutral BMC increases from 0.00558 to 0.00771; retrained-period neutral
BMC changes from 0.00738 to 0.00733. NCORR decreases in both periods.

`reproduce.py` runs the baseline and augmented models for both validation periods in the supplied
*Unity Feature Transforms* post. Install its additional, pinned dependencies in a separate environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r reproduction_requirements.txt
python reproduce.py --config reproduction_config.json --results-dir reproduction_results/sampled
```

With these dependencies installed, run `python validate_reproduction.py` for the accelerated-kernel,
continuous-value, and cache-integrity checks. The lightweight feature-builder tests above do not
require the reproduction dependencies.

The configuration pins v6.0 round 1385, all 1,040 base features (including country as a categorical input), the
350 numeric medium features, `target_jupiter_60`, and the `lgbm_jupiter_60` benchmark. It specifies the
LightGBM settings, sampling seed, 200,000 training rows, date cutoffs, and a 60-business-day embargo.
The 22 continuous base factors, such as `feature_adv_20d_factor`, retain their original values; only the
new time-series features are ranked and binned by this runner.
It trains two models on the original training split and two after extending training through 2022.
Each pair uses the same training rows and is scored on the same eligible validation rows and dates.

The post does not supply its original model configuration or exact cutoffs. The checked-in configuration
uses the Unity notebook's model and sampling recipe as an explicit fallback. Matching the post's rounded
numbers is checked and reported; the claimed numbers are not used to tune the models.

For the confirmation using **every eligible training row**, reuse the first run's verified features:

```bash
python reproduce_full.py --config full_training_config.json --prepared-cache data/reproduction
```

`full_training_config.json` keeps the model settings and evaluation rules fixed, and uses all 4,329,443
original training rows and 5,707,284 rows through 2022. Its streaming LightGBM input is checked against
the sklearn wrapper on actual data before training. With no `--prepared-cache`, this command downloads
the pinned data and builds the features itself. Its results go under `reproduction_results/full/`;
the sampled attempt remains under `reproduction_results/sampled/`.

The runner uses a compiled implementation of the example's transforms and compares its raw values and
bins with `build_features` and `rank_and_bin` on synthetic edge cases and sampled actual stock histories.
It builds a disk-backed matrix so the full feature set can be processed without holding all float32
columns in memory. Use `--cache-dir` and `--data-dir` to select cache and download locations. Large
downloads, matrices, models, and predictions stay in the cache; results include the full configuration,
data and code hashes, library versions, date coverage, per-date NCORR and neutral BMC, aggregate scores,
and cumulative plots. The optional `--source-post` argument also records the supplied HTML's hash.
Changed inputs, source code, dependencies, or configuration require a fresh cache directory. Compare
the recorded input hashes with the checked-in results when verifying a rerun; the version name alone
does not identify a data snapshot.
To render either per-date CSV with yearly date labels, run
`python plot_reproduction.py reproduction_results/sampled/per_date_scores.csv` (or the `full` path).
