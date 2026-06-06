"""Feature engineering -- the single source of truth.

This is the canonical ``add_features`` used by training AND inference. It is
intentionally dependency-light (pure pandas/numpy, no pandas_ta) so the
cloudpickled ``predict.pkl`` is portable to any scoring environment. When the
model is exported, this module is registered for pickle-by-value, so the feature
code travels *inside* predict.pkl.
"""
from __future__ import annotations

# The exact 11-feature set the model is trained on.
FEATURE_COLS = [
    "log_return",
    "rsi",
    "bb_width",
    "bb_pct",
    "volatility_24h",
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_6",
    "hour_sin",
    "hour_cos",
]


def add_features(df):
    """Engineer the canonical Topic 69 feature set from raw OHLCV candles.

    RSI-14 (Wilder) and Bollinger 20/2 (bandwidth + %B) are computed in pure
    pandas/numpy. Requires a DatetimeIndex (the cyclical hour encoding uses
    ``index.hour``). Adds no forward-looking columns, so it is safe on live data.
    Imports are inside the function so the captured code is self-sufficient.
    """
    import numpy as np
    import pandas as pd  # noqa: F401  (kept for self-sufficiency)

    data = df.copy()

    # 1. Log returns.
    data["log_return"] = np.log(data["close"] / data["close"].shift(1))

    # 2a. RSI (Wilder's smoothing, length 14).
    rsi_len = 14
    delta = data["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / rsi_len, min_periods=rsi_len, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_len, min_periods=rsi_len, adjust=False).mean()
    rs = avg_gain / avg_loss
    data["rsi"] = 100.0 - 100.0 / (1.0 + rs)

    # 2b. Bollinger Bands (length 20, std 2): bandwidth and %B.
    bb_len, bb_std = 20, 2.0
    mid = data["close"].rolling(bb_len).mean()
    sd = data["close"].rolling(bb_len).std(ddof=0)
    upper = mid + bb_std * sd
    lower = mid - bb_std * sd
    data["bb_width"] = 100.0 * (upper - lower) / mid
    data["bb_pct"] = (data["close"] - lower) / (upper - lower)

    # 3. Rolling realized volatility (24h std of returns).
    data["volatility_24h"] = data["log_return"].rolling(window=24).std()

    # 4. Lagged returns (memory).
    for lag in (1, 2, 3, 6):
        data[f"return_lag_{lag}"] = data["log_return"].shift(lag)

    # 5. Cyclical hour-of-day encoding.
    data["hour_sin"] = np.sin(2 * np.pi * data.index.hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data.index.hour / 24)

    data.dropna(inplace=True)
    return data
