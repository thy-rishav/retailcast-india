"""
Backtest: hold out the LAST 28 days of the (post-launch) training history as a
proxy horizon, since we cannot see the real d_1914-1941 ground truth. Train each
of the three candidate model families on data before the holdout, forecast 28
days forward, score with mean RMSSE (primary, matches competition weighting)
and global WAPE (secondary).

This mimics the real submission task as closely as possible: same 28-day
horizon length, same launch-detection logic re-run on train-only data (no
leakage), same feature availability rules (price/calendar/vendor known ahead,
market_signal never used).
"""
from __future__ import annotations
import json
import sys
import numpy as np
import pandas as pd

from data_loader import load_sales, load_calendar, load_prices, load_vendor_signal, detect_launch_day, N_TRAIN_DAYS, HORIZON
from models import forecast_seasonal_baseline, forecast_pooled_gbm, ensemble_forecast
from metrics import score_submission


def run_backtest(base: str | None = None, horizon: int = HORIZON, train_end_day: int | None = None):
    sales = load_sales(base)
    cal = load_calendar(base)
    prices = load_prices(base)
    vendor = load_vendor_signal(base)

    if train_end_day is None:
        train_end_day = N_TRAIN_DAYS - horizon        # last day used for training
    holdout_start = train_end_day + 1                 # first held-out day

    dcols_all = [c for c in sales.columns if c.startswith("d_")]
    vals_all = sales[dcols_all].values.astype(float)
    launch_lookup = {
        sid: detect_launch_day(vals_all[i][:train_end_day])
        for i, sid in enumerate(sales["id"].values)
    }

    print(f"[backtest] training through d_{train_end_day}, evaluating d_{holdout_start}..d_{holdout_start + horizon - 1}",
          file=sys.stderr)

    print("[backtest] fitting Model A: seasonal baseline...", file=sys.stderr)
    fc_a = forecast_seasonal_baseline(sales, cal, train_end_day, horizon)

    print("[backtest] fitting Model B: pooled gradient boosted model...", file=sys.stderr)
    fc_b = forecast_pooled_gbm(sales, cal, prices, vendor, train_end_day, horizon)

    print("[backtest] building Model C: ensemble...", file=sys.stderr)
    fc_c = ensemble_forecast(fc_a, fc_b)

    scores = {}
    for name, fc in [("A_seasonal_baseline", fc_a), ("B_pooled_gbm", fc_b), ("C_ensemble", fc_c)]:
        res = score_submission(sales, holdout_start, horizon, fc, launch_lookup)
        scores[name] = {"mean_rmsse": res["mean_rmsse"], "wape": res["wape"]}
        print(f"[backtest] {name}: mean_RMSSE={res['mean_rmsse']:.4f}  WAPE={res['wape']:.4f}", file=sys.stderr)

    return scores, {"A_seasonal_baseline": fc_a, "B_pooled_gbm": fc_b, "C_ensemble": fc_c}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_end_day", type=int, default=None)
    args = ap.parse_args()
    scores, _ = run_backtest(train_end_day=args.train_end_day)
    print(json.dumps(scores, indent=2))
