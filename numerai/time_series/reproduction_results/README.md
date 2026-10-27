# Unity transform reproduction

This records an independent attempt to reproduce the supplied *Unity Feature Transforms* post with
the [prompt and tips](../README.md). The source reports 1,040 Unity features versus 2,440 features
after adding four transforms to each of the 350 numeric medium features. Its original model settings,
training sample, data round, exact date cutoffs, and short-history implementation were not supplied.
The configurations here state those choices explicitly.

Scores below are means of weekly NCORR and neutral benchmark contribution against
`lgbm_jupiter_60`. This benchmark contribution is the offline proxy used in the example, not official
NMMC against Numerai's Meta Model.

| Period | Run | Unity NCORR | With transforms NCORR | Unity neutral BMC | With transforms neutral BMC |
| --- | --- | ---: | ---: | ---: | ---: |
| Original | Supplied post | 0.0383 | 0.0371 | 0.0041 | 0.0081 |
| Retrained through 2022 | Supplied post | — | — | 0.0033 | 0.0079 |
| Original | 200,000 training rows | 0.0196853 | 0.0159607 | 0.00148836 | 0.00276978 |
| Retrained through 2022 | 200,000 training rows | 0.0231149 | 0.0183240 | 0.00579593 | 0.00552834 |
| Original | All training rows | 0.0346069 | 0.0320885 | 0.00558218 | 0.00770695 |
| Retrained through 2022 | All training rows | 0.0349951 | 0.0308860 | 0.00738309 | 0.00733436 |

**Neither attempt reproduces the reported numbers to four decimals.** With all training rows,
original-period neutral BMC increases by about 38%, while retrained-period neutral BMC is slightly
lower. NCORR decreases in both periods. The full-training result supports an original-period
contribution gain under this configuration, but does not reproduce the claimed doubling or the
retrained-period gain. The omitted original configuration and snapshot prevent identifying a unique
cause for the differences; the published quantitative result remains unverified by these runs.

## Data and feature definitions

- Classic Unity **v6.0, round 1385**, downloaded on 2026-10-27. The input file hashes, sizes, schemas,
  and row counts are recorded in each run's `summary.json`.
- The train file contains 4,329,443 rows, from 2003-01-03 through 2018-12-28. The validation file
  contains 2,724,886 rows, including reserved test rows and dates without mature targets.
- All 1,040 source features are retained. Country is categorical; the 22 continuous factor columns
  preserve their original float32 values. Only the new features are ranked and binned by the runner.
- The 350 numeric medium features receive the four fixed transforms. All windows end at the current
  date and use only that ticker's observations. The three 52-week transforms require 52 calendar weeks
  since the first valid observation and 52 nonmissing trailing observations; the short mean additionally
  requires 13 nonmissing observations. Change age remains unknown until an actual change is observed.
- The resulting transforms use average-tie ranks within each date and `ceil(percentile_rank * 5) - 1`
  bins, preserving missing values. All observed tickers on that date participate in binning. The
  [example README](../README.md#definitions) specifies window boundaries, gaps, and missing-value rules.
- Train, validation, and test feature observations provide history. Targets do not enter feature
  generation. Modeling and scoring use mature `target_jupiter_60` values; this column was verified
  equal to the `target` alias in the downloaded train and validation files.

## Training and evaluation

Both configurations use 300 LightGBM trees, learning rate 0.05, maximum depth 5, 31 leaves, feature
fraction 0.1, seed 0, and eight model threads. They use the same feature definitions and evaluation
rules. The sampled configuration follows the Unity notebook's 200,000-row sampling recipe with seed 7
in the original Parquet row order. The full configuration uses every eligible row: 4,329,443 for
original training and 5,707,284 after extending training through 2022. There is no early stopping or
hyperparameter search against the reported numbers.

For each period, the baseline and augmented model use exactly the same training rows. A
60-business-day embargo follows the last training date. Evaluation excludes reserved test rows and
rows without mature targets, while retaining them as historical feature observations.

| Period | Last training date | First scoring date | Last scoring date | Weekly dates | Scored stock/date rows per model |
| --- | --- | --- | --- | ---: | ---: |
| Original | 2018-12-28 | 2019-03-29 | 2026-07-24 | 383 | 2,558,422 |
| Retrained through 2022 | 2022-12-30 | 2023-03-31 | 2026-07-24 | 174 | 1,173,934 |

The official `numerai-tools==0.7.2` functions score each pair on identical ticker/date keys, with all
300 neutralizer columns and the public `lgbm_jupiter_60` benchmark. Neither run lost selected
rows in the benchmark or neutralizer joins. Each per-date CSV records row counts and ticker-key hashes
so the paired coverage can be checked independently. Summaries also report neutral BMC on the last
200 available dates and prediction correlation with the benchmark.

## Repeat the runs

From `numerai/time_series/`, use Python 3.13 and the checked-in dependency lock:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r reproduction_requirements.txt
python -m unittest discover -s . -p 'test_*.py'
python validate_reproduction.py
python reproduce.py --config reproduction_config.json --results-dir reproduction_results/sampled
python reproduce_full.py --config full_training_config.json --prepared-cache data/reproduction
```

The full driver can also run without `--prepared-cache`; it then downloads the pinned input files and
prepares its own features. Reusing the first run's matrix avoids recalculating identical transforms.
Changed code, inputs, configuration, or recorded environment requires a fresh cache. Compare the input
hashes with the checked-in summaries when verifying a rerun.

The full driver streams training data with LightGBM's `Sequence` interface. Before training, it checks
native-versus-sklearn predictions on 10,000 actual training rows, all 2,440 columns, and 300 trees; the
measured maximum absolute difference was zero. The accelerated transforms are also checked against the
public example on synthetic cases and sampled actual stock histories. Nine feature-builder tests and
four additional reproduction checks passed.

The measured environment is Python 3.13.9 on an Apple M4 Max with 64 GiB RAM, macOS on arm64.
[`environment.txt`](environment.txt) contains the full installed versions and NumPy/Apple Accelerate
configuration. `blas_threads=1` records the requested library limit; `threadpoolctl` cannot enumerate
Apple Accelerate's pool. The sampled run took 571.5 seconds, including 350.5 seconds for feature
preparation, excluding the initial downloads and discarded preparation attempts. Inputs occupy about
8.26 GB; the ordinal/transformed matrix occupies 17.2 GB, plus separate continuous-factor columns.
The full-training confirmation took 1,216.5 seconds (20.3 minutes), reusing the prepared features,
with a measured peak process RSS of 30.12 GiB. Its training-row files were independently checked:
every eligible row appears exactly once, in source order, and the earlier samples are subsets.

## Artifacts

The supplied HTML is identified by SHA256
`af61a346b700ab56113c2013d2e639365e7e1de29a8ffd908b2dc7e09bb939b7`.
The optional `--source-post` argument records that hash on a rerun. Large model, prediction, matrix,
and training-row files remain in the local cache; compact provenance and measured scores are committed.

- [`sampled/summary.json`](sampled/summary.json): configuration, input/code hashes, coverage, measured
  metrics, differences from the reported figures, and four-decimal match checks.
- [`sampled/per_date_scores.csv`](sampled/per_date_scores.csv): paired weekly scores and coverage.
- [`sampled/output_manifest.json`](sampled/output_manifest.json): hashes of models, predictions, and
  sampled training-row indices, generated after the signed run.
- [`sampled/cumulative_scores.svg`](sampled/cumulative_scores.svg): cumulative weekly NCORR and neutral BMC.
- [`full/summary.json`](full/summary.json): full-training metrics, canonical model parameters,
  input/code/environment signatures, coverage, parity checks, timing, and memory measurements.
- [`full/per_date_scores.csv`](full/per_date_scores.csv): full-training paired weekly scores and coverage.
- [`full/output_manifest.json`](full/output_manifest.json): full model, prediction, and training-row hashes.
- [`full/cumulative_scores.svg`](full/cumulative_scores.svg): full-training cumulative NCORR and neutral BMC.

Charts use the recorded per-date scores. Regenerate the readable yearly date labels with
`python plot_reproduction.py reproduction_results/sampled/per_date_scores.csv`, or the corresponding
`full` path.
