"""
Feature engineering that is safe to use at prediction time.

Everything here is either:
  (a) known in advance (calendar dates, festivals, day-of-week, snap flags, prices -
      all provided through the horizon in calendar.csv / sell_prices.csv), or
  (b) derived purely from each series' own past (lags, rolling means) which is
      legitimately available when we stand at forecast time.

market_signal.csv is deliberately excluded (see data_loader.py docstring / the
Q2 write-up) - it is target leakage and has no horizon coverage anyway.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# One isolated, non-repeating price glitch was found in the raw price feed:
# GROCERY_3_PICKLE / MH_2 drops to 1.20 (vs a stable ~4.34) for exactly one week
# at a time, with no accompanying sales lift (rules out a real promo). One of the
# two occurrences (wm_yr_wk 2315) falls inside the forecast horizon itself. We do
# not hand-fix the price table; instead we neutralise its effect by winsorizing
# any single-week price move of more than 40% relative to a store-item's trailing
# median back to that median. This is a general, reusable rule (not a one-off
# patch) that would also catch the same failure mode if it recurred elsewhere.
PRICE_JUMP_THRESHOLD = 0.40


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_values(["item_id", "store_id", "wm_yr_wk"]).reset_index(drop=True).copy()
    cleaned_price = prices["sell_price"].copy()

    for _, idx in prices.groupby(["item_id", "store_id"]).groups.items():
        g = prices.loc[idx, "sell_price"]
        med = g.median()
        if med == 0 or pd.isna(med):
            continue
        rel = (g - med).abs() / med
        bad = rel > PRICE_JUMP_THRESHOLD
        prev_ok = g.shift(1).sub(med).abs().div(med) < PRICE_JUMP_THRESHOLD
        next_ok = g.shift(-1).sub(med).abs().div(med) < PRICE_JUMP_THRESHOLD
        isolated = bad & prev_ok.fillna(True) & next_ok.fillna(True)
        cleaned_price.loc[g.index[isolated]] = med

    out = prices.copy()
    out["sell_price"] = cleaned_price
    return out


EVENT_LIFT_ORDER = [
    "Diwali", "New_Year", "Eid_al_Fitr", "Dussehra", "Christmas", "Republic_Day",
    "Ganesh_Chaturthi", "Pongal", "Holi", "IPL_Final", "Onam", "Independence_Day",
    "Ram_Navami", "Gandhi_Jayanti", "Raksha_Bandhan",
]


def build_calendar_features(cal: pd.DataFrame) -> pd.DataFrame:
    cal = cal.copy()
    cal["date"] = pd.to_datetime(cal["date"])
    cal["dow"] = cal["wday"]  # 1=Sat...7=Fri, already provided
    cal["is_event"] = cal["event_name_1"].notna().astype(int)
    for name in EVENT_LIFT_ORDER:
        cal[f"event_{name}"] = (cal["event_name_1"] == name).astype(int)
    return cal


def event_lift_table(sales_long: pd.DataFrame, cal: pd.DataFrame) -> pd.Series:
    """Empirical per-event lift multiplier vs. each series' own mean, pooled
    across all series (small sample per single series-event so we pool)."""
    m = sales_long.merge(cal[["d", "event_name_1"]], on="d", how="left")
    series_mean = m.groupby("id")["actual"].transform("mean").replace(0, np.nan)
    m["rel"] = m["actual"] / series_mean
    lift = m[m.event_name_1.notna()].groupby("event_name_1")["rel"].mean()
    return lift.clip(lower=0.8, upper=2.0)  # keep sane bounds, avoid overfitting to n=1-2 events


def dow_factor_table(sales_long: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    """Per-series day-of-week multiplicative factor relative to that series' own mean."""
    m = sales_long.merge(cal[["d", "wday"]], on="d", how="left")
    series_mean = m.groupby("id")["actual"].transform("mean").replace(0, np.nan)
    m["rel"] = m["actual"] / series_mean
    tab = m.groupby(["id", "wday"])["rel"].mean().unstack("wday")
    return tab.clip(lower=0.4, upper=2.0)
