"""
Generate the final submission.csv for RetailCast India.

Model choice: ensemble of (A) seasonal baseline and (B) pooled gradient boosted
model. Selected because it had the best mean RMSSE AND best WAPE, averaged
across three independent 28-day backtest folds (see backtest.py / README),
beating both components individually and beating model B alone by a wide
margin. See approach_summary.md Q4-Q5 for the full reasoning.

Usage:
    python3 generate_submission.py [--data-dir ../data] [--out ../output/submission.csv]
"""
from __future__ import annotations
import argparse
import os
import sys

from data_loader import load_sales, load_calendar, load_prices, load_vendor_signal, N_TRAIN_DAYS, HORIZON
from models import forecast_seasonal_baseline, forecast_pooled_gbm, ensemble_forecast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None, help="Directory containing the input CSVs")
    ap.add_argument("--out", default=None, help="Output path for submission.csv")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_path = args.out or os.path.join(here, "..", "output", "submission.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("[generate] loading data...", file=sys.stderr)
    sales = load_sales(args.data_dir)
    cal = load_calendar(args.data_dir)
    prices = load_prices(args.data_dir)
    vendor = load_vendor_signal(args.data_dir)

    train_end_day = N_TRAIN_DAYS  # use ALL available history for the real submission

    print("[generate] fitting seasonal baseline (Model A)...", file=sys.stderr)
    fc_a = forecast_seasonal_baseline(sales, cal, train_end_day, HORIZON)

    print("[generate] fitting pooled GBM (Model B)...", file=sys.stderr)
    fc_b = forecast_pooled_gbm(sales, cal, prices, vendor, train_end_day, HORIZON)

    print("[generate] blending into final ensemble (Model C)...", file=sys.stderr)
    fc_c = ensemble_forecast(fc_a, fc_b)

    fc_c.to_csv(out_path, index=False)
    print(f"[generate] wrote {out_path}  ({len(fc_c)} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
