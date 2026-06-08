"""Incremental OHLCV ingestion + a persistent local store, per symbol.

Each daily run extends each symbol's store with only the candles new since the
last run (dedup + gap checks), so training always sees fresh + historical data.
The fetcher is injectable (``fetcher(config, symbol, since_ms)``) so the pipeline
and server can be tested offline with synthetic data.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

log = logging.getLogger("forge.data")

OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def _exchange(config):
    import ccxt
    return getattr(ccxt, config.exchange)({"enableRateLimit": True})


def fetch_ohlcv(config, symbol, since_ms=None, limit=None) -> pd.DataFrame:
    """Fetch candles for ``symbol``. With ``since_ms`` paginate forward to now;
    otherwise fetch the most recent ``limit`` candles."""
    ex = _exchange(config)
    rows = []
    if since_ms is None:
        rows = ex.fetch_ohlcv(symbol, config.timeframe, limit=limit or config.recent_candles)
    else:
        since = since_ms
        while True:
            batch = ex.fetch_ohlcv(symbol, config.timeframe, since=since,
                                   limit=config.fetch_page_limit)
            if not batch:
                break
            rows += batch
            since = batch[-1][0] + 1
            if batch[-1][0] >= ex.milliseconds() - 60_000:
                break
            time.sleep(ex.rateLimit / 1000.0)
    return _to_frame(rows)


def _to_frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["timestamp", *OHLCV_COLS])
    if df.empty:
        return df.set_index(pd.DatetimeIndex([], name="date"))[OHLCV_COLS]
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.set_index("date").drop(columns=["timestamp"])[OHLCV_COLS]


def load_store(config, symbol=None) -> pd.DataFrame:
    symbol = symbol or config.symbol
    path = config.data_path_for(symbol)
    import os
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index.name = "date"
        return df[OHLCV_COLS].astype("float64")
    return pd.DataFrame(columns=OHLCV_COLS,
                        index=pd.DatetimeIndex([], name="date")).astype("float64")


def save_store(config, symbol, df: pd.DataFrame) -> None:
    config.ensure_dirs()
    df.sort_index().to_csv(config.data_path_for(symbol))


def _integrity_check(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    before = len(df)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if before != len(df):
        log.warning("dropped %d duplicate candle(s)", before - len(df))
    if len(df) > 1:
        step = pd.Timedelta(timeframe)
        gaps = df.index.to_series().diff().dropna()
        n_gaps = int((gaps > step).sum())
        if n_gaps:
            log.warning("found %d gap(s) larger than one %s candle", n_gaps, timeframe)
    return df


def update_data(config, symbol=None, fetcher=None) -> pd.DataFrame:
    """Load ``symbol``'s store, fetch new candles, append, validate, persist,
    return the full history. ``fetcher(config, symbol, since_ms)`` overrides the
    live exchange fetch for tests."""
    symbol = symbol or config.symbol
    fetcher = fetcher or (lambda cfg, sym, since_ms: fetch_ohlcv(cfg, sym, since_ms=since_ms))
    store = load_store(config, symbol)

    if store.empty:
        start = datetime.now(timezone.utc) - timedelta(days=config.train_window_days + 5)
        since_ms = int(start.timestamp() * 1000)
        log.info("[%s] cold start: fetching ~%d days of history", symbol, config.train_window_days)
    else:
        since_ms = int(store.index[-1].timestamp() * 1000) + 1
        log.info("[%s] incremental fetch since %s", symbol, store.index[-1])

    fresh = fetcher(config, symbol, since_ms)
    if store.empty:
        combined = fresh
    elif fresh.empty:
        combined = store
    else:
        combined = pd.concat([store, fresh])
    combined = _integrity_check(combined, config.timeframe)
    if not combined.empty:
        combined[OHLCV_COLS] = combined[OHLCV_COLS].astype("float64")
    save_store(config, symbol, combined)
    log.info("[%s] store now holds %d candles (%s -> %s)",
             symbol, len(combined), combined.index.min(), combined.index.max())
    return combined


def get_recent_candles(config, symbol=None, limit=None, fetcher=None) -> pd.DataFrame:
    """Candles for a single live inference. Prefers a live fetch; falls back to
    the tail of the local store if the exchange is unreachable."""
    symbol = symbol or config.symbol
    limit = limit or config.recent_candles
    if fetcher is not None:
        return fetcher(config, symbol, None).tail(limit)
    try:
        df = fetch_ohlcv(config, symbol, since_ms=None, limit=limit)
        if not df.empty:
            return df
    except Exception as exc:  # network / exchange error -> fall back to store
        log.warning("[%s] live fetch failed (%s); using local store tail", symbol, exc)
    return load_store(config, symbol).tail(limit)


def window(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Keep only the last ``days`` of candles (rolling training window)."""
    if df.empty:
        return df
    cutoff = df.index.max() - pd.Timedelta(days=days)
    return df[df.index >= cutoff]
