# RetailCast India — Approach Summary / Technical Decision Log

All claims below are reproducible via `src/audit.py` (data findings) or `src/backtest.py`
(model comparison); code line references are to files in this repo.

## Q1. Audit method

I worked outward from structure to substance. First, shape and coverage: row/column counts,
day-index ranges, date ranges, and confirming `sales_train` ids matched `sample_submission`
exactly. Second, per-series distributions: mean, std, max, and zero-fraction for all 60
series, sorted, to spot outliers and intermittency levels. Third, regime stability: I split
each series' history into quarters and looked at ratio of the last quarter's mean to the
first's — several series showed 100x+ ratios, which sent me looking for launch dates via a
rolling-mean threshold crossing. Fourth, feed provenance: for `market_signal` and
`vendor_signal` I checked (a) day-index coverage relative to the `d_1914` cutoff and (b)
per-series correlation with actual sales, not just pooled correlation, since pooled
correlation across differently-scaled series is a well-known trap. Fifth, price sanity: I
scanned `sell_prices` for weeks that moved >40% relative to a store-item's own median and
cross-referenced sales that week. I considered the audit "done" once every provided file had
been checked against the single question "will I have this at prediction time, and does it
mean what it claims to mean" — at that point every remaining anomaly was either explained or
explicitly logged as unresolved (see Q3).

## Q2. Data verdicts

**What:** `market_signal.csv`, all 60 series, `d_1`–`d_1913`.
**Evidence:** Per-series correlation with `actual` sales is 0.87–0.96 (`audit.py` section 1).
Regressing `mkt_signal` on `actual` gives a consistent slope of ~10.5–11x across series of
wildly different item/store combinations (checked directly on `ELECTRONICS_1_CHARGER_MH_1`:
10.66x, and confirmed on two more series: 10.72x and 10.13x), with residual std about half of
`mkt_signal`'s total std — i.e. `mkt_signal ≈ 10.5 × actual + noise`. The feed also stops
dead at `d_1913`, exactly the last day we have ground truth for.
**Action:** Excluded entirely from every model. It is never joined into `features.py` or
`models.py`.
**Reading rejected:** "It's a genuinely predictive external market index that happens to be
well-correlated with our own sales" — a real independent demand index could correlate with
sales, but wouldn't reproduce a near-constant per-series multiplicative scale factor across
six unrelated product categories, and a genuinely externally-sourced index wouldn't
conveniently truncate at the exact last day of known ground truth. Both facts together are
much better explained by "this was constructed from the target," which also means it cannot
exist for the horizon — using it would have been actively counterproductive even if the
horizon coverage problem didn't exist.

**What:** `vendor_signal.csv`, all 60 series, full range including the horizon (`d_1`–`d_1941`).
**Evidence:** Pooled correlation with actual sales looks strong (0.78), but per-series
correlation is mostly 0.0–0.3 (`audit.py` section 5; only one series exceeds 0.5). The pooled
number is a scale artifact — some series average ~130 units/day and others ~0.2, so pooling
mixes "getting the general size of the number right" with "tracking day-to-day demand," and
the latter is what a forecast feature actually needs.
**Action:** Included only as one of many weak features in the pooled GBM (Model B), never as
a primary driver, and never in the seasonal baseline (Model A) at all.
**Reading rejected:** "A vendor forecast with 0.78 correlation is strong and should be
weighted heavily" — this reading only survives if you don't check per-series; once you do,
the feature has almost no daily skill and inconsistent per-series bias (0.15x–5.8x of actual
recent levels), so leaning on it would inject noise rather than signal.

**What:** `HOMECARE_1_DETERGENT` (all 10 stores) and `HOMECARE_2_AGARBATTI` (all 10 stores).
**Evidence:** Rolling-mean launch detection (`data_loader.detect_launch_day`) finds a hard
cutover from all-zero to real sales at day ~462–474 for DETERGENT and day ~1284–1327 for
AGARBATTI, consistently across every store for each item. `sell_prices.csv` has prices for
both items from day 1, so this isn't a missing-price artifact — the SKUs existed in the price
system before being stocked.
**Action:** Both models are trained per-series only on data from each series' detected launch
day forward (`start_day` logic in `_make_supervised_frame`; `level_window`/`launch_idx` logic
in `forecast_seasonal_baseline`), not the full 1,913-day history.
**Reading rejected:** "It's organic demand ramp-up, so more history helps the trend estimate"
— an organic ramp is gradual and store-idiosyncratic; a same-week cutover across all 10 stores
for one item is a listing/distribution event, and including the ~1,300+ zero days for
AGARBATTI would have crushed its estimated level and (for a difference-based scale like
RMSSE's denominator) badly distorted its error scaling too.

**What:** `GROCERY_3_PICKLE` / `MH_2`, `wm_yr_wk 2040` and `wm_yr_wk 2315`.
**Evidence:** Price drops from a stable ~₹4.34 to exactly ₹1.20 for one isolated week, with
sales for that week (`d_997`–`d_1003`: 0,2,0,0,1,3,0) not elevated versus the series' overall
mean — no promo-shaped response. `wm_yr_wk 2315` maps to `d_1921`–`d_1927`, inside the 28-day
forecast horizon.
**Action:** `features.clean_prices` clamps any single-week price move >40% from a store-item's
own median back to the median, but only when both neighboring weeks are near-median (so real,
sustained repricing is left alone). This is a general rule, not a one-off patch for this row.
**Reading rejected:** "It's a real flash promotion" — a genuine promo at that price point
would be expected to move volume; it didn't, and the same anomaly recurring at exactly the
horizon week strongly suggests it's a recorded/joined-in-error price row rather than a
merchandising event.

## Q3. What I left alone

Several very-low-volume series (e.g. `HOMECARE_2_AGARBATTI_KA_3`, mean 0.23/day, 88% zero
days) look "broken" if judged only by variance — sales are dominated by long zero-runs
punctuated by small spikes. I left this pattern untouched rather than smoothing or imputing
it, because it's the expected shape of genuine intermittent low-volume retail demand, not an
artifact: the same items sell reliably in other stores, prices are stable, and there's no
launch discontinuity for this item at this store. Smoothing it would understate real
volatility and produce an artificially confident (and wrong) forecast; restraint here just
means letting the RMSSE scaling and a wide day-of-week/level estimate do their job.

## Q4. Modelling choices

I built and backtested three families: (A) a per-series seasonal baseline — an
exponentially-weighted recent level (56-day window, 21-day half-life) times an empirical
day-of-week factor times a pooled festival-lift multiplier; (B) a single pooled
`HistGradientBoostingRegressor` (Poisson loss) trained across all 60 series with
lag/rolling/calendar/price/vendor features, forecast recursively day-by-day; and (C) their
arithmetic-mean ensemble. I considered per-series gradient boosting (rejected — 60 series is
far too little data per model to justify it) and an ARIMA/ETS family (rejected in favor of
the simpler weighted-level baseline, since with launch discontinuities and hard zero-runs,
classical decomposition assumptions are shaky and a transparent weighted mean was easier to
reason about and just as effective in backtesting). Q2's verdicts shape every model directly:
`market_signal` never enters any feature set; `vendor_signal` enters only Model B, with low
weight implied by its actual predictive value; both late-launch products are trained on
post-launch-only windows in both A and B; and cleaned (not raw) prices feed Model B.

## Q5. Validation you trust

I backtested by holding out the last 28 days of history as a stand-in horizon, training only
on data before it, and re-running the exact same launch-detection and feature logic on the
truncated training data only (no peeking at the holdout to decide launch dates or price
cleaning). I did this for three independent origins (`train_end_day` = 1885, 1857, 1829) to
reduce the chance of a single lucky/unlucky 28-day window driving the model choice. Averaged
mean RMSSE: A=0.700, B=0.716, C=0.694; averaged WAPE: A=0.414, B=0.444, C=0.410. The ensemble
won on both metrics in 2 of 3 folds and was essentially tied in the third, so I selected it.
What could make this look better than reality: using the same launch-detection or price-clean
parameters tuned on the holdout itself (I didn't — those are fixed, data-driven rules run
identically on train-only slices), or picking the fold that flatters a given model (I
averaged three, not the best one). Expected real-horizon mean RMSSE: roughly 0.65–0.75.

## Q6. Least-sure call

The choice to weight the ensemble 50/50 (rather than something like 70/30 toward the
consistently-stronger seasonal baseline) is my least confident call — it was the simplest
option and it won in backtesting, but the backtest is only 3 folds on 60 fairly noisy series,
so the "50/50 beats other splits" claim is thin. Evidence that would change my mind: a wider
sweep of blend weights across more backtest origins showing a different weight consistently
dominating. Hedge in the meantime: the pooled GBM component (B) already has middling
individual performance, so even if the true optimal blend is closer to 70/30 toward A, the
50/50 ensemble's error is not far from either component alone — it fails gracefully rather
than catastrophically.

## Q7. Reproduce and stress

```
python3 src/generate_submission.py --data-dir data --out output/submission.csv
```
If next month's data arrived with a new problem from the same family as something I found —
e.g. a third leaky vendor feed, another mid-panel product launch, or a price glitch landing on
a different week — the launch-detection and price-jump-guard logic in `data_loader.py` /
`features.py` are general rules (rolling-mean threshold, isolated-deviation-from-median) that
would catch a repeat of the *launch* or *price-glitch* pattern automatically. A new leaky
feed, however, would not be caught automatically — the market_signal leakage was found by
manual per-series correlation and scale-factor inspection (`audit.py` section 1), and nothing
in the pipeline currently runs that check on arbitrary new files. I would catch it, the
pipeline would not.
