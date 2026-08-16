"""
Three candidate model families, evaluated head-to-head in backtest.py:

  A) seasonal_baseline   - per-series weighted recent level x day-of-week factor
                            x festival lift. Simple, robust, fully explainable.
  B) pooled_gbm          - a single HistGradientBoostingRegressor fit across all
                            60 series (pooled), with lag/rolling/calendar/price
                            features, forecast recursively day-by-day.
  C) ensemble            - arithmetic mean of A and B.

All three only ever use each series' own post-launch history (see
data_loader.detect_launch_day) and only features that are genuinely known ahead
of time (calendar, price, vendor_signal) - never market_signal.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from data_loader import get_day_cols, detect_launch_day, HORIZON
from features import build_calendar_features, event_lift_table, dow_factor_table, clean_prices


# ---------------------------------------------------------------------------
# Model A: seasonal baseline
# ---------------------------------------------------------------------------

def forecast_seasonal_baseline(sales: pd.DataFrame, cal: pd.DataFrame,
                                train_end_day: int, horizon: int = HORIZON,
                                level_window: int = 56, halflife: int = 21) -> pd.DataFrame:
    """
    train_end_day: last 1-indexed day number (d_<train_end_day>) included in training.
    Returns a DataFrame [id, F1..Fhorizon].
    """
    dcols_all = get_day_cols(sales)
    train_dcols = [f"d_{i}" for i in range(1, train_end_day + 1)]
    train_dcols = [c for c in train_dcols if c in dcols_all]

    cal_f = build_calendar_features(cal)
    cal_map = cal_f.set_index("d_num")

    # build long-format actuals restricted to training window for factor tables
    sub_sales = sales[["id"] + train_dcols]
    long_sales = sub_sales.melt(id_vars="id", value_vars=train_dcols, var_name="d", value_name="actual")
    dow_tab = dow_factor_table(long_sales, cal_f)
    lift_tab = event_lift_table(long_sales, cal_f)

    horizon_days = list(range(train_end_day + 1, train_end_day + 1 + horizon))
    rows = []
    vals_all = sales[dcols_all].values.astype(float)
    for i, sid in enumerate(sales["id"].values):
        series = vals_all[i]
        launch_idx = detect_launch_day(series[:train_end_day])
        usable = series[launch_idx:train_end_day]
        if len(usable) == 0:
            level = 0.0
        else:
            w_window = usable[-level_window:] if len(usable) >= 1 else usable
            n = len(w_window)
            weights = 0.5 ** (np.arange(n)[::-1] / halflife)
            level = float(np.average(w_window, weights=weights))

        dow_series = dow_tab.loc[sid] if sid in dow_tab.index else None
        f = []
        for d in horizon_days:
            wday = int(cal_map.loc[d, "wday"]) if d in cal_map.index else None
            factor = 1.0
            if dow_series is not None and wday in dow_series.index and not pd.isna(dow_series.loc[wday]):
                factor = float(dow_series.loc[wday])
            event = cal_map.loc[d, "event_name_1"] if d in cal_map.index else None
            if isinstance(event, str) and event in lift_tab.index:
                factor *= float(lift_tab.loc[event])
            f.append(max(0.0, level * factor))
        rows.append([sid] + f)

    cols = ["id"] + [f"F{k}" for k in range(1, horizon + 1)]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Model B: pooled gradient boosted model, recursive forecasting
# ---------------------------------------------------------------------------

LAGS = [1, 7, 14, 28]
ROLL_WINDOWS = [7, 14, 28, 56]


def _make_supervised_frame(sales: pd.DataFrame, cal: pd.DataFrame, prices_clean: pd.DataFrame,
                            vendor: pd.DataFrame, train_end_day: int) -> pd.DataFrame:
    """Long panel of (id, day) rows up to train_end_day with lag/rolling/calendar/price
    features and the target `actual`, restricted to each series' post-launch window."""
    dcols_all = get_day_cols(sales)
    vals_all = sales[dcols_all].values.astype(float)
    from features import EVENT_LIFT_ORDER
    cal_f = build_calendar_features(cal)
    event_cols = [f"event_{n}" for n in EVENT_LIFT_ORDER]
    cal_small = cal_f[["d_num", "wday", "month", "snap_MH", "snap_KA", "snap_TN", "is_event"] + event_cols]

    prices_idx = prices_clean.set_index(["item_id", "store_id", "wm_yr_wk"])["sell_price"]
    wk_map = cal_f.set_index("d_num")["wm_yr_wk"]
    vendor_idx = vendor.set_index(["id", "d_num"])["vendor_forecast"]

    max_lag = max(LAGS + ROLL_WINDOWS)
    frames = []
    for i, row in sales.iterrows():
        sid, item_id, store_id, state_id = row["id"], row["item_id"], row["store_id"], row["state_id"]
        series = vals_all[i]
        launch_idx = detect_launch_day(series[:train_end_day])  # 0-based
        start_day = launch_idx + 1 + max_lag  # 1-indexed day, leave room for lag features
        if start_day > train_end_day:
            continue
        days = np.arange(start_day, train_end_day + 1)
        df = pd.DataFrame({"d_num": days})
        df["id"] = sid
        df["item_id"] = item_id
        df["store_id"] = store_id
        df["state_id"] = state_id
        df["actual"] = series[days - 1]
        for lag in LAGS:
            df[f"lag_{lag}"] = series[days - 1 - lag]
        for w in ROLL_WINDOWS:
            # rolling mean of the w days strictly before each day
            roll_vals = np.array([series[max(0, d - 1 - w):d - 1].mean() if d - 1 - w >= 0
                                   else series[0:max(1, d - 1)].mean() for d in days])
            df[f"roll_{w}"] = roll_vals
        df["days_since_launch"] = days - (launch_idx + 1)
        df = df.merge(cal_small, on="d_num", how="left")
        df["wm_yr_wk"] = df["d_num"].map(wk_map)
        df["sell_price"] = df.apply(
            lambda r: prices_idx.get((item_id, store_id, r["wm_yr_wk"]), np.nan), axis=1)
        snap_col = {"MH": "snap_MH", "KA": "snap_KA", "TN": "snap_TN"}[state_id]
        df["snap_state"] = df[snap_col]
        df["vendor_forecast"] = [vendor_idx.get((sid, d), np.nan) for d in days]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["sell_price"] = out.groupby(["item_id", "store_id"])["sell_price"].transform(
        lambda s: s.ffill().bfill())
    out["vendor_forecast"] = out["vendor_forecast"].fillna(out["roll_28"])
    return out


FEATURE_COLS = (
    [f"lag_{l}" for l in LAGS] + [f"roll_{w}" for w in ROLL_WINDOWS] +
    ["days_since_launch", "wday", "month", "snap_state", "is_event", "sell_price",
     "vendor_forecast", "item_code", "store_code"]
)


def forecast_pooled_gbm(sales: pd.DataFrame, cal: pd.DataFrame, prices: pd.DataFrame,
                         vendor: pd.DataFrame, train_end_day: int, horizon: int = HORIZON,
                         random_state: int = 0) -> pd.DataFrame:
    prices_clean = clean_prices(prices)
    train_df = _make_supervised_frame(sales, cal, prices_clean, vendor, train_end_day)

    item_codes = {v: k for k, v in enumerate(sorted(sales["item_id"].unique()))}
    store_codes = {v: k for k, v in enumerate(sorted(sales["store_id"].unique()))}
    train_df["item_code"] = train_df["item_id"].map(item_codes)
    train_df["store_code"] = train_df["store_id"].map(store_codes)

    from features import EVENT_LIFT_ORDER
    event_cols = [f"event_{n}" for n in EVENT_LIFT_ORDER]
    feat_cols = FEATURE_COLS + event_cols

    X = train_df[feat_cols].astype(float).values
    y = train_df["actual"].astype(float).values
    model = HistGradientBoostingRegressor(
        loss="poisson", max_depth=6, learning_rate=0.06, max_iter=300,
        min_samples_leaf=20, l2_regularization=1.0, random_state=random_state,
    )
    model.fit(X, y)

    # recursive multi-step forecast
    dcols_all = get_day_cols(sales)
    vals_all = sales[dcols_all].values.astype(float)
    cal_f = build_calendar_features(cal)
    cal_map = cal_f.set_index("d_num")
    prices_idx = prices_clean.set_index(["item_id", "store_id", "wm_yr_wk"])["sell_price"]
    wk_map = cal_f.set_index("d_num")["wm_yr_wk"]
    vendor_idx = vendor.set_index(["id", "d_num"])["vendor_forecast"]

    results = {sid: [] for sid in sales["id"]}
    max_lag = max(LAGS + ROLL_WINDOWS)
    for i, row in sales.iterrows():
        sid, item_id, store_id, state_id = row["id"], row["item_id"], row["store_id"], row["state_id"]
        history = list(vals_all[i][:train_end_day])  # index 0 = day 1
        launch_idx = detect_launch_day(np.array(history))
        snap_col = {"MH": "snap_MH", "KA": "snap_KA", "TN": "snap_TN"}[state_id]

        for h in range(1, horizon + 1):
            d = train_end_day + h
            feat = {}
            for lag in LAGS:
                idx = d - 1 - lag
                feat[f"lag_{lag}"] = history[idx] if 0 <= idx < len(history) else 0.0
            for w in ROLL_WINDOWS:
                lo = max(0, d - 1 - w)
                hi = d - 1
                window_vals = history[lo:hi] if hi > lo else history[-1:]
                feat[f"roll_{w}"] = float(np.mean(window_vals)) if len(window_vals) else 0.0
            feat["days_since_launch"] = d - (launch_idx + 1)
            crow = cal_map.loc[d] if d in cal_map.index else None
            feat["wday"] = float(crow["wday"]) if crow is not None else 0.0
            feat["month"] = float(crow["month"]) if crow is not None else 0.0
            feat["snap_state"] = float(crow[snap_col]) if crow is not None else 0.0
            feat["is_event"] = float(crow["is_event"]) if crow is not None else 0.0
            for ec in event_cols:
                feat[ec] = float(crow[ec]) if crow is not None else 0.0
            wk = wk_map.get(d, None)
            feat["sell_price"] = float(prices_idx.get((item_id, store_id, wk), np.nan))
            feat["vendor_forecast"] = float(vendor_idx.get((sid, d), feat["roll_28"]))
            feat["item_code"] = item_codes[item_id]
            feat["store_code"] = store_codes[store_id]

            xrow = np.array([[feat[c] for c in feat_cols]], dtype=float)
            xrow = np.nan_to_num(xrow, nan=0.0)
            pred = max(0.0, float(model.predict(xrow)[0]))
            results[sid].append(pred)
            history.append(pred)

    cols = ["id"] + [f"F{k}" for k in range(1, horizon + 1)]
    rows = [[sid] + results[sid] for sid in sales["id"]]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Model C: ensemble
# ---------------------------------------------------------------------------

def ensemble_forecast(fc_a: pd.DataFrame, fc_b: pd.DataFrame) -> pd.DataFrame:
    fcols = [c for c in fc_a.columns if c != "id"]
    merged = fc_a.merge(fc_b, on="id", suffixes=("_a", "_b"))
    out = merged[["id"]].copy()
    for c in fcols:
        out[c] = (merged[f"{c}_a"] + merged[f"{c}_b"]) / 2.0
    return out
