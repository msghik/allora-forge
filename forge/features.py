"""Feature engineering -- the single source of truth.

Canonical ``add_features`` used by training AND inference. Pure pandas/numpy (no
pandas_ta) so the cloudpickled ``predict.pkl`` is portable, and registered for
pickle-by-value at export so the feature code travels inside the artifact.

All features are scale-free (returns, ratios, oscillators, z-scores, cyclical
time) so the model generalizes across BTC price regimes. Requires a DatetimeIndex.
"""
from __future__ import annotations

FEATURE_COLS = [
    # returns / momentum
    "log_return", "ret_12", "ret_24", "ret_48",
    "return_lag_1", "return_lag_2", "return_lag_3",
    # volatility
    "volatility_12", "volatility_24", "volatility_48", "parkinson_24", "vol_regime",
    # trend / moving-average distance
    "ema_dist_20", "ema_dist_50", "ema_dist_100", "macd", "macd_hist", "adx_14",
    # oscillators
    "rsi", "stoch_k", "williams_r", "cci", "mfi",
    # bands
    "bb_width", "bb_pct",
    # volume / order flow
    "vol_z", "flow_imbalance",
    # candle shape (microstructure proxies)
    "body", "upper_wick", "lower_wick", "close_pos",
    # cyclical time
    "hour_sin", "hour_cos", "minute_sin", "minute_cos", "dow_sin", "dow_cos",
]

EPS = 1e-12


def add_features(df):
    """Engineer the canonical feature set from raw OHLCV candles.

    Imports are inside the function so the captured code is self-sufficient at
    inference time. Adds no forward-looking columns -> safe on live data.
    """
    import numpy as np
    import pandas as pd  # noqa: F401

    data = df.copy()
    o, h, l, c, v = (data["open"], data["high"], data["low"],
                     data["close"], data["volume"])

    def wilder(s, n):
        return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()

    # --- returns / momentum ---
    data["log_return"] = np.log(c / c.shift(1))
    for k in (12, 24, 48):
        data[f"ret_{k}"] = np.log(c / c.shift(k))
    for lag in (1, 2, 3):
        data[f"return_lag_{lag}"] = data["log_return"].shift(lag)

    # --- volatility ---
    for k in (12, 24, 48):
        data[f"volatility_{k}"] = data["log_return"].rolling(k).std()
    data["parkinson_24"] = np.sqrt(
        (np.log(h / l) ** 2).rolling(24).mean() / (4.0 * np.log(2.0)))
    data["vol_regime"] = data["volatility_12"] / (
        data["volatility_12"].rolling(288).median() + EPS)

    # --- trend / MA distance ---
    for n in (20, 50, 100):
        data[f"ema_dist_{n}"] = c / c.ewm(span=n, adjust=False).mean() - 1.0
    macd = (c.ewm(span=12, adjust=False).mean()
            - c.ewm(span=26, adjust=False).mean()) / c
    data["macd"] = macd
    data["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()

    # ADX(14)
    up_move, down_move = h.diff(), -l.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
                        index=data.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
                         index=data.index)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = wilder(tr, 14)
    plus_di = 100 * wilder(plus_dm, 14) / (atr + EPS)
    minus_di = 100 * wilder(minus_dm, 14) / (atr + EPS)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    data["adx_14"] = wilder(dx, 14)

    # --- oscillators ---
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    rs = wilder(gain, 14) / (wilder(loss, 14) + EPS)
    data["rsi"] = 100.0 - 100.0 / (1.0 + rs)

    ll14, hh14 = l.rolling(14).min(), h.rolling(14).max()
    data["stoch_k"] = 100 * (c - ll14) / (hh14 - ll14 + EPS)
    data["williams_r"] = -100 * (hh14 - c) / (hh14 - ll14 + EPS)

    tp = (h + l + c) / 3.0
    sma_tp = tp.rolling(20).mean()
    mad = (tp - sma_tp).abs().rolling(20).mean()
    data["cci"] = (tp - sma_tp) / (0.015 * mad + EPS)

    mf = tp * v
    pos_mf = mf.where(tp > tp.shift(1), 0.0).rolling(14).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0.0).rolling(14).sum()
    data["mfi"] = 100.0 - 100.0 / (1.0 + pos_mf / (neg_mf + EPS))

    # --- Bollinger bands (20, 2) ---
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    upper, lower = mid + 2 * sd, mid - 2 * sd
    data["bb_width"] = 100.0 * (upper - lower) / (mid + EPS)
    data["bb_pct"] = (c - lower) / (upper - lower + EPS)

    # --- volume / order flow ---
    data["vol_z"] = (v - v.rolling(48).mean()) / (v.rolling(48).std(ddof=0) + EPS)
    data["flow_imbalance"] = (np.sign(data["log_return"]) * v).rolling(12).sum() / (
        v.rolling(12).sum() + EPS)

    # --- candle shape (microstructure proxies) ---
    rng = (h - l) + EPS
    data["body"] = (c - o) / rng
    data["upper_wick"] = (h - np.maximum(o, c)) / rng
    data["lower_wick"] = (np.minimum(o, c) - l) / rng
    data["close_pos"] = (c - l) / rng

    # --- cyclical time ---
    idx = data.index
    data["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    data["minute_sin"] = np.sin(2 * np.pi * idx.minute / 60)
    data["minute_cos"] = np.cos(2 * np.pi * idx.minute / 60)
    data["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    data["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)

    data.dropna(inplace=True)
    return data
