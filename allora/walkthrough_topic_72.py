"""
walkthrough_topic_72.py -- train + export a predict.pkl for Allora Forge TOPIC 72
(1h BTC/USD log-return prediction, updated every 5 minutes).

WHY THIS VERSION: the Builder Kit's default features are raw windowed OHLCV bars
(48 bars x 5 = 240 non-stationary level features), which give a tree ~0.50 DA
(grade F). This script instead engineers the stationary, scale-free features we
validated (returns / volatility / RSI / MACD / EMA-distance / candle shape /
cross-asset ETH), which lift DA to ~0.53 and pass a clear majority of the kit's
metrics. It uses the kit ONLY for deployment (wallet / faucet / submission); the
model + features are self-contained, so predict.pkl runs anywhere with
numpy/pandas/ccxt/lightgbm.

HOW TO USE (inside the cloned allora-forge-builder-kit, its venv active)
    pip install ccxt                      # this script fetches OHLCV via ccxt
    cp /path/to/allora-forge/allora/walkthrough_topic_72.py .
    python walkthrough_topic_72.py        # -> writes predict.pkl  (DA/grade printed)
    TOPIC_ID=72 python deploy_worker.py
    python -m allora_forge_builder_kit.web_dashboard      # monitor :8787

Data source: ccxt 'binanceus' by default (set ALLORA_EXCHANGE=binance outside the US).
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

import cloudpickle
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

EXCHANGE = os.environ.get("ALLORA_EXCHANGE", "binanceus")
SYMBOL = os.environ.get("ALLORA_SYMBOL", "BTC/USDT")
CROSS = os.environ.get("ALLORA_CROSS_SYMBOL", "ETH/USDT")
TIMEFRAME = "5m"
HORIZON = 12              # 12 * 5m = 1 hour ahead  (topic-72 target)
DAYS = 365
EPS = 1e-12

FEATURE_COLS = [
    "log_return", "ret_12", "ret_24", "ret_48", "return_lag_1", "return_lag_2", "return_lag_3",
    "volatility_12", "volatility_24", "volatility_48", "rsi", "macd", "macd_hist",
    "ema_dist_20", "ema_dist_50", "ema_dist_100", "bb_pct", "bb_width", "vol_z",
    "body", "upper_wick", "lower_wick", "close_pos",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "eth_log_return", "eth_ret_12", "ret_spread_1", "ret_spread_12", "x_corr_48",
]


# ----------------- data (ccxt) -----------------
def fetch_ohlcv(symbol, since_ms=None, limit=500):
    """Recent `limit` candles (since_ms=None) or paginate forward from since_ms."""
    import ccxt
    ex = getattr(ccxt, EXCHANGE)({"enableRateLimit": True})
    if since_ms is None:
        rows = ex.fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
    else:
        rows, since = [], since_ms
        while True:
            batch = ex.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=1000)
            if not batch:
                break
            rows += batch
            since = batch[-1][0] + 1
            if batch[-1][0] >= ex.milliseconds() - 60_000:
                break
            time.sleep(ex.rateLimit / 1000.0)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms")
    return df.set_index("date")[["open", "high", "low", "close", "volume"]].astype("float64")


# ----------------- features (stationary, scale-free) -----------------
def add_features(df, ref_df):
    d = df.copy()
    o, h, l, c, v = d["open"], d["high"], d["low"], d["close"], d["volume"]

    def wilder(s, n):
        return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()

    d["log_return"] = np.log(c / c.shift(1))
    for k in (12, 24, 48):
        d[f"ret_{k}"] = np.log(c / c.shift(k))
        d[f"volatility_{k}"] = d["log_return"].rolling(k).std()
    for lag in (1, 2, 3):
        d[f"return_lag_{lag}"] = d["log_return"].shift(lag)
    delta = c.diff()
    rs = wilder(delta.clip(lower=0), 14) / (wilder(-delta.clip(upper=0), 14) + EPS)
    d["rsi"] = 100 - 100 / (1 + rs)
    macd = (c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()) / c
    d["macd"] = macd
    d["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    for n in (20, 50, 100):
        d[f"ema_dist_{n}"] = c / c.ewm(span=n, adjust=False).mean() - 1.0
    mid, sd = c.rolling(20).mean(), c.rolling(20).std(ddof=0)
    d["bb_pct"] = (c - (mid - 2 * sd)) / (4 * sd + EPS)
    d["bb_width"] = 100 * (4 * sd) / (mid + EPS)
    d["vol_z"] = (v - v.rolling(48).mean()) / (v.rolling(48).std(ddof=0) + EPS)
    rng = (h - l) + EPS
    d["body"] = (c - o) / rng
    d["upper_wick"] = (h - np.maximum(o, c)) / rng
    d["lower_wick"] = (np.minimum(o, c) - l) / rng
    d["close_pos"] = (c - l) / rng
    idx = d.index
    d["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    d["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    d["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)

    # cross-asset (ETH)
    rc = ref_df["close"].reindex(d.index).ffill()
    eth_lr = np.log(rc / rc.shift(1))
    d["eth_log_return"] = eth_lr
    d["eth_ret_12"] = np.log(rc / rc.shift(12))
    d["ret_spread_1"] = d["log_return"] - eth_lr
    d["ret_spread_12"] = d["ret_12"] - d["eth_ret_12"]
    d["x_corr_48"] = d["log_return"].rolling(48).corr(eth_lr)

    d.dropna(subset=FEATURE_COLS, inplace=True)
    return d


# ----------------- train -----------------
if __name__ == "__main__":
    print(f"[fetch] {DAYS}d of {TIMEFRAME} {SYMBOL}/{CROSS} via {EXCHANGE} ...")
    since = int((datetime.now(timezone.utc) - timedelta(days=DAYS)).timestamp() * 1000)
    btc = fetch_ohlcv(SYMBOL, since_ms=since)
    eth = fetch_ohlcv(CROSS, since_ms=since)
    feats = add_features(btc, eth)
    feats["target"] = np.log(feats["close"].shift(-HORIZON) / feats["close"])
    feats.dropna(subset=["target"], inplace=True)
    X, y = feats[FEATURE_COLS], feats["target"]
    print(f"[data] {len(X)} rows, {len(FEATURE_COLS)} features")

    BASE = dict(n_estimators=600, learning_rate=0.02, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0, random_state=42, verbose=-1)
    GRID = [dict(max_depth=d, num_leaves=nl, min_child_samples=mc)
            for d, nl in ((3, 15), (4, 31)) for mc in (200, 400)]
    n, k = len(X), 4
    vlen = n // (2 * k)
    folds = [(n - (k - i) * vlen, n - (k - 1 - i) * vlen) for i in range(k)]

    def evaluate(params):
        das, oof_p, oof_t = [], [], []
        for v0, v1 in folds:
            tr = max(1, v0 - HORIZON)
            m = LGBMRegressor(**{**BASE, **params}).fit(X.iloc[:tr], y.iloc[:tr])
            p, t = m.predict(X.iloc[v0:v1]), y.iloc[v0:v1].values
            das.append(np.mean(np.sign(p) == np.sign(t)))
            oof_p += list(p); oof_t += list(t)
        oof_p, oof_t = np.array(oof_p), np.array(oof_t)
        r = float(np.corrcoef(oof_p, oof_t)[0, 1]) if np.std(oof_p) > 0 else 0.0
        return float(np.mean(das)), float(np.std(das)), r, oof_p, oof_t

    print(f"[tune] {len(GRID)} configs x {k} folds ...")
    best = None
    for params in GRID:
        da_m, da_s, r, oof_p, oof_t = evaluate(params)
        print(f"   depth={params['max_depth']} min_child={params['min_child_samples']:>3} | "
              f"DA={da_m:.3f}+/-{da_s:.3f}  r={r:.3f}")
        if best is None or da_m > best[0]:
            best = (da_m, da_s, r, oof_p, oof_t, params)
    da_m, da_s, r, oof_p, oof_t, BEST = best
    print(f"[best] {BEST} -> DA={da_m:.3f}+/-{da_s:.3f}  r={r:.3f}")

    denom = float(np.dot(oof_p, oof_p))
    SCALE = float(np.clip(np.dot(oof_p, oof_t) / denom, 0.05, 1.0)) if denom > 0 else 1.0
    print(f"[scale] SCALE={SCALE:.3f}")
    try:
        from allora_forge_builder_kit import PerformanceEvaluator
        g = PerformanceEvaluator().evaluate(y_true=oof_t, y_pred=oof_p * SCALE)
        print(f"[grade] {g.get('grade')}  passed {g.get('num_passed')}/{g.get('num_primary_metrics')}  {g.get('passed')}")
    except Exception as exc:  # noqa: BLE001
        print(f"[grade] skipped ({exc})")

    final_model = LGBMRegressor(**{**BASE, **BEST}).fit(X, y)

    def predict(nonce: int = None) -> float:
        """Predicted 1h BTC/USD LOG-RETURN (topic 72's target). Self-fetching via ccxt."""
        b = fetch_ohlcv(SYMBOL, since_ms=None, limit=500)
        e = fetch_ohlcv(CROSS, since_ms=None, limit=500)
        f = add_features(b, e)
        if len(f) == 0:
            raise ValueError("not enough live candle history")
        return float(final_model.predict(f[FEATURE_COLS].tail(1))[0]) * SCALE

    try:
        print(f"[live] sample prediction = {predict():+.6f} log-return")
    except Exception as exc:  # noqa: BLE001
        print(f"[live] sample skipped ({exc}); will work once deployed")

    with open("predict.pkl", "wb") as fh:
        cloudpickle.dump(predict, fh)
    print("[done] wrote predict.pkl  ->  deploy:  TOPIC_ID=72 python deploy_worker.py")
