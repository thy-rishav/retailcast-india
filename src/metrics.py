from __future__ import annotations
import numpy as np
import pandas as pd


def rmsse(actual_train: np.ndarray, actual_test: np.ndarray, forecast: np.ndarray) -> float:
    """M5-style RMSSE. actual_train is the in-sample history (post-launch, i.e.
    the scale should reflect the period the series was actually active) used
    only to compute the naive-forecast scaling denominator."""
    actual_train = np.asarray(actual_train, dtype=float)
    if len(actual_train) < 2:
        denom = 1.0
    else:
        diffs = np.diff(actual_train)
        denom = np.mean(diffs ** 2)
        if denom <= 1e-9:
            denom = max(np.mean(actual_train ** 2), 1e-6)
    num = np.mean((np.asarray(actual_test, dtype=float) - np.asarray(forecast, dtype=float)) ** 2)
    return float(np.sqrt(num / denom))


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    denom = np.sum(np.abs(actual))
    if denom <= 1e-9:
        return float(np.sum(np.abs(forecast)))
    return float(np.sum(np.abs(actual - forecast)) / denom)


def score_submission(sales: pd.DataFrame, holdout_start_day: int, horizon: int,
                      forecast_df: pd.DataFrame, launch_lookup: dict) -> dict:
    """
    sales: full sales_train-style frame (id + d_1..d_N), N >= holdout_start_day+horizon-1
    holdout_start_day: first 1-indexed day of the held-out actuals (== train_end_day+1)
    launch_lookup: id -> 0-based launch index (computed on TRAIN data only, no leakage)
    """
    dcols = [c for c in sales.columns if c.startswith("d_")]
    vals = sales.set_index("id")[dcols]
    per_series_rmsse = {}
    all_actual, all_forecast = [], []
    for _, r in forecast_df.iterrows():
        sid = r["id"]
        fcols = [c for c in forecast_df.columns if c.startswith("F")]
        fc = r[fcols].values.astype(float)
        test_days = [f"d_{holdout_start_day + k}" for k in range(horizon)]
        actual_test = vals.loc[sid, test_days].values.astype(float)
        launch_idx = launch_lookup.get(sid, 0)
        train_days = [f"d_{i}" for i in range(launch_idx + 1, holdout_start_day)]
        actual_train = vals.loc[sid, train_days].values.astype(float) if train_days else np.array([0.0])
        per_series_rmsse[sid] = rmsse(actual_train, actual_test, fc)
        all_actual.append(actual_test)
        all_forecast.append(fc)
    mean_rmsse = float(np.mean(list(per_series_rmsse.values())))
    global_wape = wape(np.concatenate(all_actual), np.concatenate(all_forecast))
    return {"mean_rmsse": mean_rmsse, "wape": global_wape, "per_series_rmsse": per_series_rmsse}
