# RetailCast India — 28-Day Demand Forecast

Forecasts `d_1914`..`d_1941` for 60 product-store series, per the RetailCast India challenge brief.

## Setup

Dependencies: `pandas`, `numpy`, `scikit-learn` (only). No `lightgbm`, no internet access
required at run time — `sklearn.ensemble.HistGradientBoostingRegressor` is used for the
gradient-boosted component so the pipeline runs anywhere scikit-learn is already installed.

```bash
pip install -r requirements.txt   # at Lilly, resolve this via Artifactory, not public PyPI
```

This repo does not bundle the raw data (per the starter kit convention, it lives in a sibling
`../data/` directory: `sales_train.csv`, `calendar.csv`, `sell_prices.csv`, `market_signal.csv`,
`vendor_signal.csv`, `sample_submission.csv`). Point `--data-dir` at wherever your copy lives.

## Reproduce the submission

```bash
python3 src/generate_submission.py --data-dir ../data --out output/submission.csv
python3 validate_format.py --submission output/submission.csv --sample ../data/sample_submission.csv
```

(adjust `../data` to wherever the challenge data folder sits relative to this repo)

## Reproduce the data audit (every claim in `approach_summary.md`)

```bash
cd src && python3 audit.py
```

This re-derives, from the raw CSVs, every data-quality finding referenced in the decision log:
the `market_signal.csv` leakage/scale relationship, the two late-launch product lines, the
isolated price glitch and its overlap with the forecast horizon, the festival lift table, and
`vendor_signal.csv`'s per-series correlation with actual sales.

## Reproduce the model-selection backtest

```bash
cd src && python3 backtest.py                        # last 28 days of history as holdout
python3 backtest.py --train_end_day 1857              # earlier fold
python3 backtest.py --train_end_day 1829              # earlier fold
```

Each run trains all three candidate model families (seasonal baseline, pooled GBM, ensemble)
on data before the holdout and scores them against the true held-out days with mean RMSSE and
WAPE, mirroring the competition's own scoring. See `approach_summary.md` Q5 for the results and
why the ensemble was selected.

## Repo layout

```
src/
  data_loader.py        loading + launch-date detection (auto, per series)
  features.py           calendar/event/price feature engineering + price-glitch guard
  models.py             Model A (seasonal baseline), B (pooled GBM), C (ensemble)
  metrics.py            RMSSE / WAPE implementations
  backtest.py           head-to-head model comparison on held-out history
  audit.py              reproduces every data-quality claim from raw data
  generate_submission.py   trains the winning model on full history, writes submission.csv
output/
  submission.csv         final forecast
approach_summary.md       technical decision log (the 7 required questions)
validate_format.py        organizer-supplied format validator
```

## Key modelling decisions (see `approach_summary.md` for full reasoning)

- `market_signal.csv` is never used as a feature: it is a noised ~10.5x rescaling of the
  actual target (leakage) and has zero coverage of the forecast horizon.
- `vendor_signal.csv` is used only as a weak auxiliary feature — its daily correlation with
  actual sales is near zero per series, despite superficially higher correlation when pooled.
- `HOMECARE_1_DETERGENT` and `HOMECARE_2_AGARBATTI` are auto-detected as late-launch products
  (all 10 stores, consistent cutover dates) and trained only on their post-launch history.
- A single-week price glitch in `GROCERY_3_PICKLE`/`MH_2` (isolated ~72% drop, no sales
  response) is neutralised by a general isolated-price-jump guard, not a one-off patch —
  this matters because one occurrence of the same glitch falls inside the forecast horizon.
