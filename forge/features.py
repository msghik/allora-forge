"""Feature engineering -- the single source of truth.

Canonical ``add_features`` used by training AND inference. Pure pandas/numpy (no
pandas_ta) so the cloudpickled ``predict.pkl`` is portable, and registered for
pickle-by-value at export so the feature code travels inside the artifact.

Feature blocks (all scale-free -> generalize across price regimes):
  * single-asset price/volume/microstructure/time   (``FEATURE_COLS``)
  * order flow from raw klines (taker-buy / CVD)     (``ORDERFLOW_COLS``)
  * cross-asset lead-lag vs a reference asset         (``CROSS_FEATURE_COLS``)
  * futures positioning (funding rate, open interest) (``FUTURES_COLS``)

Optional blocks are gated by config flags (so the train/inference feature list is
deterministic) and *degrade gracefully*: if a data source is missing the columns
are still produced, filled with neutral values, so the model never sees NaNs and
the worker keeps serving. Requires a DatetimeIndex.
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
    # volume / order flow (proxy)
    "vol_z", "flow_imbalance",
    # candle shape (microstructure proxies)
    "body", "upper_wick", "lower_wick", "close_pos",
    # cyclical time
    "hour_sin", "hour_cos", "minute_sin", "minute_cos", "dow_sin", "dow_cos",
]

# Real order-flow features (need taker-buy volume / trade count from raw klines).
ORDERFLOW_COLS = [
    "taker_buy_ratio_z", "ofi_12", "ofi_48", "cvd_slope_24",
    "trade_intensity_z", "avg_trade_size_z",
]

CROSS_FEATURE_COLS = [
    "{p}_log_return", "{p}_ret_12", "{p}_ret_24", "{p}_volatility_24",
    "{p}_return_lag_1", "{p}_return_lag_2", "{p}_return_lag_3",
    "ret_spread_1", "ret_spread_12", "x_corr_48", "x_beta_48",
]

# Futures positioning features (funding rate + open interest).
FUTURES_COLS = [
    "funding_rate", "funding_z", "funding_roll_24",
    "oi_change_12", "oi_change_48", "oi_price_div_12",
]

# On-chain features (Dune): stablecoin supply, CEX net-flow, DEX volume, activity.
ONCHAIN_COLS = [
    "stable_supply_chg24", "stable_supply_z",
    "cex_netflow_z", "cex_netflow_sum24_z",
    "dex_vol_z", "active_addr_z",
]

EPS = 1e-12


def cross_feature_cols(prefix: str) -> list:
    """Cross-asset feature column names for a given reference prefix (e.g. 'eth')."""
    return [c.format(p=prefix) for c in CROSS_FEATURE_COLS]


def active_feature_cols(cross_prefix=None, use_orderflow: bool = False,
                        use_futures: bool = False, use_onchain: bool = False) -> list:
    """Full ordered feature list the model trains/predicts on (config-determined)."""
    cols = list(FEATURE_COLS)
    if use_orderflow:
        cols += list(ORDERFLOW_COLS)
    if cross_prefix:
        cols += cross_feature_cols(cross_prefix)
    if use_futures:
        cols += list(FUTURES_COLS)
    if use_onchain:
        cols += list(ONCHAIN_COLS)
    return cols


def build_features(df, ref_df=None, cross_prefix=None, fut_df=None, onchain_df=None,
                   use_orderflow: bool = False, use_futures: bool = False,
                   use_onchain: bool = False):
    """Single source of truth used by both training and inference.

    Primary single-asset features (+ optional order flow), optionally joined with
    cross-asset features (``ref_df``), futures features (``fut_df``) and on-chain
    features (``onchain_df``). Optional blocks degrade to neutral when absent.
    """
    feats = add_features(df, use_orderflow=use_orderflow)
    if ref_df is not None and cross_prefix:
        feats = add_cross_features(feats, ref_df, prefix=cross_prefix)
    if use_futures:
        feats = add_futures_features(feats, fut_df)
    if use_onchain:
        feats = add_onchain_features(feats, onchain_df)
    return feats


def _dtindex(x):
    import pandas as pd
    if isinstance(x.index, pd.DatetimeIndex):
        return x
    for col in ("date", "timestamp"):
        if col in x.columns:
            unit = "ms" if col == "timestamp" else None
            return x.set_index(pd.to_datetime(x[col], unit=unit))
    return x


def add_cross_features(primary_feats, ref_df, prefix="eth"):
    """Add reference-asset (e.g. ETH) features + cross interactions to the primary
    (e.g. BTC) feature frame. Captures lead-lag, relative strength, rolling
    correlation and beta -- strong signals between correlated crypto majors."""
    import numpy as np
    import pandas as pd

    r = _dtindex(ref_df.copy())
    rc = r["close"]
    ref_lr = np.log(rc / rc.shift(1))

    ref = pd.DataFrame(index=r.index)
    ref[f"{prefix}_log_return"] = ref_lr
    ref[f"{prefix}_ret_12"] = np.log(rc / rc.shift(12))
    ref[f"{prefix}_ret_24"] = np.log(rc / rc.shift(24))
    ref[f"{prefix}_volatility_24"] = ref_lr.rolling(24).std()
    for lag in (1, 2, 3):
        ref[f"{prefix}_return_lag_{lag}"] = ref_lr.shift(lag)

    out = primary_feats.join(ref.reindex(primary_feats.index))

    # cross interactions (primary returns vs reference returns)
    b_lr, x_lr = out["log_return"], out[f"{prefix}_log_return"]
    out["ret_spread_1"] = b_lr - x_lr
    out["ret_spread_12"] = out["ret_12"] - out[f"{prefix}_ret_12"]
    out["x_corr_48"] = b_lr.rolling(48).corr(x_lr)
    out["x_beta_48"] = b_lr.rolling(48).cov(x_lr) / (x_lr.rolling(48).var() + EPS)

    # Drop only on the cross columns -- never blanket-dropna, or all-NaN raw
    # passthrough columns (e.g. taker-buy when the exchange omits it) wipe every row.
    out.dropna(subset=cross_feature_cols(prefix), inplace=True)
    return out


def add_futures_features(primary_feats, fut_df):
    """Add perpetual-futures positioning features (funding rate + open interest).

    ``fut_df`` is a time-indexed frame with ``funding_rate`` and ``open_interest``
    columns at any cadence; it is reindexed/forward-filled onto the feature index.
    Missing data degrades to neutral (no NaNs propagated, no rows dropped) so the
    worker keeps serving when futures data is briefly unavailable.
    """
    import numpy as np
    import pandas as pd

    out = primary_feats.copy()
    idx = out.index

    if fut_df is not None and not fut_df.empty:
        f = _dtindex(fut_df.copy())
        f = f[~f.index.duplicated(keep="last")].sort_index()
        f = f.reindex(idx.union(f.index)).sort_index().ffill().reindex(idx)
        funding = f["funding_rate"] if "funding_rate" in f else pd.Series(np.nan, index=idx)
        oi = f["open_interest"] if "open_interest" in f else pd.Series(np.nan, index=idx)
    else:
        funding = pd.Series(np.nan, index=idx)
        oi = pd.Series(np.nan, index=idx)

    # --- funding rate: level, 3-day z-score, 1-day mean (carry pressure) ---
    out["funding_rate"] = funding
    fmean = funding.rolling(288 * 3, min_periods=12).mean()
    fstd = funding.rolling(288 * 3, min_periods=12).std()
    out["funding_z"] = (funding - fmean) / (fstd + EPS)
    out["funding_roll_24"] = funding.rolling(288, min_periods=12).mean()

    # --- open interest: 1h / 4h log change + divergence vs price ---
    log_oi = np.log(oi.where(oi > 0))
    out["oi_change_12"] = log_oi - log_oi.shift(12)
    out["oi_change_48"] = log_oi - log_oi.shift(48)
    ret_12 = out["ret_12"] if "ret_12" in out else pd.Series(0.0, index=idx)
    out["oi_price_div_12"] = out["oi_change_12"] * np.sign(ret_12)

    out[FUTURES_COLS] = out[FUTURES_COLS].fillna(0.0)
    return out


def add_onchain_features(primary_feats, onchain_df):
    """Add on-chain features (stablecoin supply, CEX net-flow, DEX volume, network
    activity) from a Dune-sourced frame, reindexed/forward-filled onto the feature
    index. Each metric is optional and degrades to neutral (no NaNs, no row drops)
    so the worker keeps serving when on-chain data is unavailable."""
    import numpy as np
    import pandas as pd

    out = primary_feats.copy()
    idx = out.index
    src = _dtindex(onchain_df.copy()) if (onchain_df is not None and not onchain_df.empty) \
        else pd.DataFrame()

    def metric(name):
        if name in getattr(src, "columns", []):
            s = src[name]
            s = s[~s.index.duplicated(keep="last")].sort_index()
            return s.reindex(idx.union(s.index)).sort_index().ffill().reindex(idx)
        return pd.Series(np.nan, index=idx)

    def z(s, n=288 * 7):
        return (s - s.rolling(n, min_periods=24).mean()) / (s.rolling(n, min_periods=24).std() + EPS)

    # stablecoin supply: 24h log change + z-score of its change (mint/burn pressure)
    ss = metric("stable_supply")
    log_ss = np.log(ss.where(ss > 0))
    out["stable_supply_chg24"] = log_ss - log_ss.shift(288)
    out["stable_supply_z"] = z(log_ss.diff())

    # CEX net-flow (into exchanges = sell pressure): instantaneous z + 24h-sum z
    nf = metric("cex_netflow")
    out["cex_netflow_z"] = z(nf)
    out["cex_netflow_sum24_z"] = z(nf.rolling(288, min_periods=24).sum())

    # DEX volume + active addresses: log-z (risk-on / activity regime)
    dv = metric("dex_volume")
    out["dex_vol_z"] = z(np.log(dv.where(dv > 0)))
    aa = metric("active_addr")
    out["active_addr_z"] = z(np.log(aa.where(aa > 0)))

    out[ONCHAIN_COLS] = out[ONCHAIN_COLS].fillna(0.0)
    return out


def add_features(df, use_orderflow: bool = False):
    """Engineer the canonical feature set from raw OHLCV(+order-flow) candles.

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

    def zscore(s, n):
        return (s - s.rolling(n).mean()) / (s.rolling(n).std(ddof=0) + EPS)

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

    # --- volume / order flow (sign-of-return proxy; always available) ---
    data["vol_z"] = zscore(v, 48)
    data["flow_imbalance"] = (np.sign(data["log_return"]) * v).rolling(12).sum() / (
        v.rolling(12).sum() + EPS)

    # --- candle shape (microstructure proxies) ---
    rng = (h - l) + EPS
    data["body"] = (c - o) / rng
    data["upper_wick"] = (h - np.maximum(o, c)) / rng
    data["lower_wick"] = (np.minimum(o, c) - l) / rng
    data["close_pos"] = (c - l) / rng

    # --- real order flow (taker-buy volume / CVD), if requested ---
    if use_orderflow:
        if "taker_buy_base" in data.columns and data["taker_buy_base"].notna().any():
            tb = data["taker_buy_base"].astype("float64")
            delta = 2.0 * tb - v                      # buy volume - sell volume
            data["taker_buy_ratio_z"] = zscore(tb / (v + EPS), 96)
            data["ofi_12"] = delta.rolling(12).sum() / (v.rolling(12).sum() + EPS)
            data["ofi_48"] = delta.rolling(48).sum() / (v.rolling(48).sum() + EPS)
            cvd = delta.cumsum()
            data["cvd_slope_24"] = (cvd - cvd.shift(24)) / (v.rolling(24).sum() + EPS)
            trades = data["trades"].astype("float64") if "trades" in data.columns \
                else pd.Series(np.nan, index=data.index)
            data["trade_intensity_z"] = zscore(trades, 96)
            qv = data["quote_volume"].astype("float64") if "quote_volume" in data.columns \
                else (c * v)
            data["avg_trade_size_z"] = zscore(qv / (trades + EPS), 96)
        else:  # source unavailable -> neutral so the feature list stays stable
            for col in ORDERFLOW_COLS:
                data[col] = 0.0

    # --- cyclical time ---
    idx = data.index
    data["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    data["minute_sin"] = np.sin(2 * np.pi * idx.minute / 60)
    data["minute_cos"] = np.cos(2 * np.pi * idx.minute / 60)
    data["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    data["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)

    base_and_flow = list(FEATURE_COLS) + (list(ORDERFLOW_COLS) if use_orderflow else [])
    data.dropna(subset=base_and_flow, inplace=True)
    return data
