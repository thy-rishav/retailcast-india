"""
Standalone audit script. Reproduces every data-quality claim made in
approach_summary.md directly from the raw files, so each claim is verifiable
by running this script rather than having to trust the chat transcript alone.

Usage: python3 audit.py
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

from data_loader import load_sales, load_calendar, load_prices, load_vendor_signal, \
    get_day_cols, detect_launch_day, leakage_check, N_TRAIN_DAYS, data_dir
from features import clean_prices, event_lift_table, build_calendar_features


def audit_launch_dates(sales: pd.DataFrame) -> pd.DataFrame:
    dcols = get_day_cols(sales)
    vals = sales[dcols].values.astype(float)
    rows = []
    for i, sid in enumerate(sales["id"].values):
        idx = detect_launch_day(vals[i])
        rows.append({"id": sid, "item_id": sales["item_id"].iloc[i],
                     "store_id": sales["store_id"].iloc[i], "launch_day_idx": idx,
                     "usable_days": len(dcols) - idx})
    df = pd.DataFrame(rows)
    late = df[df["launch_day_idx"] > 60].sort_values("launch_day_idx", ascending=False)
    return df, late


def audit_price_glitches(prices: pd.DataFrame, threshold: float = 0.40):
    prices = prices.sort_values(["item_id", "store_id", "wm_yr_wk"]).reset_index(drop=True)
    flags = []
    for (item, store), idx in prices.groupby(["item_id", "store_id"]).groups.items():
        g = prices.loc[idx, "sell_price"]
        med = g.median()
        if med == 0 or pd.isna(med):
            continue
        rel = (g - med).abs() / med
        bad = rel > threshold
        prev_ok = g.shift(1).sub(med).abs().div(med) < threshold
        next_ok = g.shift(-1).sub(med).abs().div(med) < threshold
        isolated = bad & prev_ok.fillna(True) & next_ok.fillna(True)
        for wk, price in zip(prices.loc[g.index[isolated], "wm_yr_wk"], g[isolated]):
            flags.append({"item_id": item, "store_id": store, "wm_yr_wk": int(wk),
                           "price": float(price), "series_median": float(med)})
    return pd.DataFrame(flags)


def audit_festival_lift(sales: pd.DataFrame, cal: pd.DataFrame) -> pd.Series:
    dcols = get_day_cols(sales)
    long_sales = sales.melt(id_vars="id", value_vars=dcols, var_name="d", value_name="actual")
    cal_f = build_calendar_features(cal)
    return event_lift_table(long_sales, cal_f).sort_values(ascending=False)


def audit_vendor_signal_skill(sales: pd.DataFrame, vendor: pd.DataFrame) -> pd.Series:
    dcols = get_day_cols(sales)
    long_sales = sales.melt(id_vars="id", value_vars=dcols, var_name="d", value_name="actual")
    vendor2 = vendor.rename(columns={"d": "d"})
    merged = vendor2.merge(long_sales, on=["id", "d"], how="inner")
    return merged.groupby("id").apply(lambda g: g["vendor_forecast"].corr(g["actual"])).sort_values()


def main():
    sales = load_sales()
    cal = load_calendar()
    prices = load_prices()
    vendor = load_vendor_signal()

    print("=" * 70)
    print("1) market_signal.csv leakage check")
    print("=" * 70)
    print(json.dumps(leakage_check(), indent=2))

    print("\n" + "=" * 70)
    print("2) Launch-date discontinuities (series with launch_day_idx > 60)")
    print("=" * 70)
    _, late = audit_launch_dates(sales)
    print(late.to_string(index=False))

    print("\n" + "=" * 70)
    print("3) Isolated single-week price glitches (>40% move, isolated)")
    print("=" * 70)
    glitches = audit_price_glitches(prices)
    print(glitches.to_string(index=False) if len(glitches) else "none found")
    horizon_weeks = set(cal[cal["d_num"].between(N_TRAIN_DAYS + 1, N_TRAIN_DAYS + 28)]["wm_yr_wk"])
    if len(glitches):
        in_horizon = glitches[glitches["wm_yr_wk"].isin(horizon_weeks)]
        print(f"\n-> {len(in_horizon)} glitch(es) fall inside the forecast horizon weeks {sorted(horizon_weeks)}:")
        print(in_horizon.to_string(index=False))

    print("\n" + "=" * 70)
    print("4) Festival lift multipliers (pooled across series, vs. each series' own mean)")
    print("=" * 70)
    print(audit_festival_lift(sales, cal).to_string())

    print("\n" + "=" * 70)
    print("5) vendor_signal.csv per-series correlation with actual sales")
    print("=" * 70)
    corr = audit_vendor_signal_skill(sales, vendor)
    print(f"mean={corr.mean():.3f}  median={corr.median():.3f}  max={corr.max():.3f}  min={corr.min():.3f}")
    print(f"vendor_signal max day: d_{vendor['d_num'].max()} (covers the {28}-day forecast horizon: "
          f"{vendor['d_num'].max() >= N_TRAIN_DAYS + 28})")


if __name__ == "__main__":
    main()
