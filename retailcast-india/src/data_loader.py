"""
Data loading utilities for RetailCast India.

Design decisions baked in here (see approach_summary.md for full reasoning):
  - market_signal.csv is NOT loaded as a model feature. Investigation showed it is a
    noised linear rescaling of the actual target (~10-11x actual sales, per-series
    correlation 0.87-0.96) and it has zero coverage of the forecast horizon (d_1914+).
    It is target leakage, not an independent market indicator. We keep a small
    diagnostic function (`leakage_check`) so the claim is reproducible from code,
    but it is never joined into the modelling frame.
  - vendor_signal.csv IS loaded (it covers the full horizon), but is used only as a
    weak auxiliary/reference signal given near-zero per-series daily correlation
    with actual sales (see approach_summary.md Q2).
  - Each series' usable history starts at its detected "launch day" (see
    `detect_launch_day`), not necessarily d_1. Two product lines
    (HOMECARE_1_DETERGENT, HOMECARE_2_AGARBATTI) are launched mid-panel.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

HORIZON = 28
N_TRAIN_DAYS = 1913


def data_dir(base: str | None = None) -> str:
    if base:
        return base
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "data")


def load_sales(base: str | None = None) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir(base), "sales_train.csv"))


def load_calendar(base: str | None = None) -> pd.DataFrame:
    cal = pd.read_csv(os.path.join(data_dir(base), "calendar.csv"))
    cal["d_num"] = cal["d"].str.replace("d_", "", regex=False).astype(int)
    return cal


def load_prices(base: str | None = None) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir(base), "sell_prices.csv"))


def load_vendor_signal(base: str | None = None) -> pd.DataFrame:
    vs = pd.read_csv(os.path.join(data_dir(base), "vendor_signal.csv"))
    vs["d_num"] = vs["d"].str.replace("d_", "", regex=False).astype(int)
    return vs


def get_day_cols(sales: pd.DataFrame) -> list[str]:
    return [c for c in sales.columns if c.startswith("d_")]


def detect_launch_day(series: np.ndarray, window: int = 30, frac_of_mean: float = 0.3,
                       min_level: float = 0.5) -> int:
    """
    Detect the first day index (0-based) from which a series has "really" launched,
    i.e. the point after which a trailing rolling mean first exceeds a threshold and
    stays populated. Products sold since day 1 will trigger this near index 0
    (bounded below by `window - 1`, the rolling-window warm-up) which is harmless:
    it just means "use full history".

    Returns the 0-based index of the first usable day.
    """
    s = pd.Series(series, dtype=float)
    roll = s.rolling(window).mean()
    thresh = max(min_level, np.nanmean(series) * frac_of_mean)
    hit = np.where(roll.values > thresh)[0]
    if len(hit) == 0:
        return 0
    idx = int(hit[0])
    # Back off to the first non-zero day at or after (idx - window) so we don't
    # discard genuine early low-volume sales that preceded the ramp.
    start = max(0, idx - window)
    nz = np.where(series[start:idx + 1] > 0)[0]
    if len(nz) > 0:
        return start + int(nz[0])
    return idx


def build_launch_table(sales: pd.DataFrame) -> pd.DataFrame:
    dcols = get_day_cols(sales)
    vals = sales[dcols].values.astype(float)
    launch_idx = [detect_launch_day(vals[i]) for i in range(len(sales))]
    return pd.DataFrame({
        "id": sales["id"].values,
        "item_id": sales["item_id"].values,
        "store_id": sales["store_id"].values,
        "launch_day_idx": launch_idx,       # 0-based index into dcols
        "n_usable_days": [len(dcols) - li for li in launch_idx],
    })


def leakage_check(base: str | None = None, sample_id: str = "ELECTRONICS_1_CHARGER_MH_1_validation"):
    """
    Reproduces the market_signal leakage finding from the investigation chat.
    Not used in the modelling pipeline -- kept for auditability.
    """
    sales = load_sales(base)
    ms = pd.read_csv(os.path.join(data_dir(base), "market_signal.csv"))
    dcols = get_day_cols(sales)
    long_sales = sales.melt(id_vars="id", value_vars=dcols, var_name="d", value_name="actual")
    merged = ms.merge(long_sales, on=["id", "d"], how="inner")
    corrs = merged.groupby("id").apply(lambda g: g["mkt_signal"].corr(g["actual"]))
    sub = merged[merged.id == sample_id]
    nz = sub[sub.actual > 0]
    ratio = (nz["mkt_signal"] / nz["actual"])
    return {
        "per_series_corr_mean": float(corrs.mean()),
        "per_series_corr_min": float(corrs.min()),
        "per_series_corr_max": float(corrs.max()),
        "sample_series": sample_id,
        "sample_scale_factor_mean": float(ratio.mean()),
        "sample_scale_factor_std": float(ratio.std()),
        "market_signal_max_day": int(ms["d"].str.replace("d_", "", regex=False).astype(int).max()),
        "sales_train_max_day": N_TRAIN_DAYS,
    }
