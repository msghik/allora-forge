"""Incremental OHLCV ingestion + persistent local stores, per symbol.

Each daily run extends each symbol's store with only the candles new since the
last run (dedup + gap checks). On top of the core OHLCV it also captures **order
flow** (taker-buy volume / trade count / quote volume) from raw klines when the
exchange exposes them, and a separate **futures** store (funding rate + open
interest). Every fetcher is injectable so the pipeline and server can run offline
with synthetic data.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

log = logging.getLogger("forge.data")

OHLCV_COLS = ["open", "high", "low", "close", "volume"]
EXT_COLS = ["quote_volume", "trades", "taker_buy_base"]   # order-flow extras
STORE_COLS = OHLCV_COLS + EXT_COLS


def _exchange(config, futures: bool = False):
    import ccxt
    name = config.futures_exchange if futures else config.exchange
    opts = {"enableRateLimit": True}
    if futures:
        opts["options"] = {"defaultType": "swap"}
    return getattr(ccxt, name)(opts)


# ----------------------------------------------------------------------------
# OHLCV (+ order flow)
# ----------------------------------------------------------------------------
def _raw_klines(ex, symbol, timeframe, since, limit):
    """Binance-family raw klines including taker-buy volume; None if unsupported."""
    getter = getattr(ex, "publicGetKlines", None) or getattr(ex, "public_get_klines", None)
    if getter is None:
        return None
    if not getattr(ex, "markets", None):   # ex.market() needs markets loaded first
        ex.load_markets()
    market = ex.market(symbol)
    params = {"symbol": market["id"], "interval": timeframe, "limit": min(limit, 1000)}
    if since is not None:
        params["startTime"] = int(since)
    rows = getter(params)
    out = []
    for k in rows:
        # [openTime,o,h,l,c,vol,closeTime,quoteVol,trades,takerBuyBase,takerBuyQuote,ignore]
        out.append([int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]),
                    float(k[5]), float(k[7]), float(k[8]), float(k[9])])
    return out


def fetch_ohlcv(config, symbol, since_ms=None, limit=None) -> pd.DataFrame:
    """Fetch candles for ``symbol`` with order-flow columns when available.

    With ``since_ms`` paginate forward to now; otherwise fetch the most recent
    ``limit`` candles. Falls back to ccxt's core OHLCV (extras = NaN) on exchanges
    that don't expose raw klines.
    """
    ex = _exchange(config)
    extended = True
    rows = []
    if since_ms is None:
        n = limit or config.recent_candles
        raw = None
        try:
            raw = _raw_klines(ex, symbol, config.timeframe, None, n)
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] raw klines failed (%s); using core OHLCV", symbol, exc)
        if raw is not None:
            rows = raw[-n:]
        else:
            extended = False
            rows = ex.fetch_ohlcv(symbol, config.timeframe, limit=n)
    else:
        since = since_ms
        while True:
            batch = None
            if extended:
                try:
                    batch = _raw_klines(ex, symbol, config.timeframe, since,
                                        config.fetch_page_limit)
                except Exception as exc:  # noqa: BLE001
                    log.warning("[%s] raw klines failed (%s); using core OHLCV", symbol, exc)
            if batch is None:
                extended = False
                batch = ex.fetch_ohlcv(symbol, config.timeframe, since=since,
                                       limit=config.fetch_page_limit)
            if not batch:
                break
            rows += batch
            last_ts = batch[-1][0]
            since = last_ts + 1
            if last_ts >= ex.milliseconds() - 60_000:
                break
            time.sleep(ex.rateLimit / 1000.0)
    return _to_frame(rows, extended)


def _to_frame(rows, extended: bool) -> pd.DataFrame:
    cols = (["timestamp", *OHLCV_COLS, *EXT_COLS] if extended
            else ["timestamp", *OHLCV_COLS])
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        empty = pd.DataFrame(columns=STORE_COLS,
                             index=pd.DatetimeIndex([], name="date"))
        return empty.astype("float64")
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("date").drop(columns=["timestamp"])
    for col in EXT_COLS:
        if col not in df.columns:
            df[col] = float("nan")
    return df[STORE_COLS].astype("float64")


def load_store(config, symbol=None) -> pd.DataFrame:
    symbol = symbol or config.symbol
    path = config.data_path_for(symbol)
    import os
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index.name = "date"
        for col in STORE_COLS:
            if col not in df.columns:           # backward-compat with old OHLCV-only stores
                df[col] = float("nan")
        return df[STORE_COLS].astype("float64")
    return pd.DataFrame(columns=STORE_COLS,
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
    for col in STORE_COLS:                       # tolerate fetchers that omit extras
        if col not in fresh.columns:
            fresh[col] = float("nan")
    fresh = fresh[STORE_COLS]
    if store.empty:
        combined = fresh
    elif fresh.empty:
        combined = store
    else:
        combined = pd.concat([store, fresh])
    combined = _integrity_check(combined, config.timeframe)
    if not combined.empty:
        combined[STORE_COLS] = combined[STORE_COLS].astype("float64")
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


# ----------------------------------------------------------------------------
# Futures alt-data (funding rate + open interest)
# ----------------------------------------------------------------------------
def _paginate_funding(ex, symbol, since_ms):
    out, since = [], since_ms
    while True:
        batch = ex.fetch_funding_rate_history(symbol, since=since, limit=1000)
        if not batch:
            break
        out += batch
        since = batch[-1]["timestamp"] + 1
        if batch[-1]["timestamp"] >= ex.milliseconds() - 60_000 or len(batch) < 1000:
            break
        time.sleep(ex.rateLimit / 1000.0)
    return out


def _paginate_oi(ex, symbol, timeframe, since_ms):
    # Binance open-interest history is limited to ~30 days; older startTime is rejected.
    earliest = ex.milliseconds() - 29 * 24 * 3600 * 1000
    out, since = [], max(since_ms or 0, earliest)
    while True:
        batch = ex.fetch_open_interest_history(symbol, timeframe, since=since, limit=500)
        if not batch:
            break
        out += batch
        since = batch[-1]["timestamp"] + 1
        if batch[-1]["timestamp"] >= ex.milliseconds() - 60_000 or len(batch) < 500:
            break
        time.sleep(ex.rateLimit / 1000.0)
    return out


def fetch_futures(config, symbol=None, since_ms=None) -> pd.DataFrame:
    """Funding rate (full history, ~8h cadence) + open interest (recent, exchange
    limited to ~30d) as a time-indexed frame. Robust: returns empty on failure so
    futures features degrade to neutral instead of breaking the run."""
    symbol = symbol or config.futures_symbol
    if not symbol:
        return pd.DataFrame()
    if since_ms is None:
        start = datetime.now(timezone.utc) - timedelta(days=config.train_window_days + 5)
        since_ms = int(start.timestamp() * 1000)
    try:
        ex = _exchange(config, futures=True)
        ex.load_markets()
    except Exception as exc:  # noqa: BLE001
        log.warning("[futures] exchange init failed (%s); futures features off", exc)
        return pd.DataFrame()

    funding = pd.Series(dtype="float64")
    try:
        rows = _paginate_funding(ex, symbol, since_ms)
        if rows:
            funding = pd.Series(
                {pd.to_datetime(r["timestamp"], unit="ms"): float(r["fundingRate"])
                 for r in rows if r.get("fundingRate") is not None}).sort_index()
    except Exception as exc:  # noqa: BLE001
        log.warning("[futures] funding fetch failed (%s)", exc)

    oi = pd.Series(dtype="float64")
    try:
        rows = _paginate_oi(ex, symbol, config.timeframe, since_ms)
        vals = {}
        for r in rows:
            v = r.get("openInterestAmount")
            if v is None:
                info = r.get("info", {})
                v = info.get("sumOpenInterest") or info.get("openInterest")
            if v is not None:
                vals[pd.to_datetime(r["timestamp"], unit="ms")] = float(v)
        if vals:
            oi = pd.Series(vals).sort_index()
    except Exception as exc:  # noqa: BLE001
        log.warning("[futures] open-interest fetch failed (%s)", exc)

    if funding.empty and oi.empty:
        return pd.DataFrame()
    out = pd.DataFrame(index=funding.index.union(oi.index))
    out.index.name = "date"
    out["funding_rate"] = funding.reindex(out.index)
    out["open_interest"] = oi.reindex(out.index)
    return out.sort_index()


def update_futures(config, symbol=None, fetcher=None) -> pd.DataFrame:
    """Refresh and persist the futures store; return the full frame.
    ``fetcher(config, symbol, since_ms)`` overrides the live fetch for tests."""
    symbol = symbol or config.futures_symbol
    if not symbol:
        return pd.DataFrame()
    fetcher = fetcher or (lambda cfg, sym, since_ms: fetch_futures(cfg, sym, since_ms))
    fresh = fetcher(config, symbol, None)
    if fresh is None or fresh.empty:
        log.info("[futures %s] no data; features will be neutral", symbol)
        return pd.DataFrame()
    fresh = fresh[~fresh.index.duplicated(keep="last")].sort_index()
    config.ensure_dirs()
    fresh.to_csv(config.futures_path_for(symbol))
    log.info("[futures %s] store holds %d rows (%s -> %s)", symbol, len(fresh),
             fresh.index.min(), fresh.index.max())
    return fresh


def get_recent_futures(config, symbol=None, fetcher=None) -> pd.DataFrame:
    """Recent funding/OI for a live inference; falls back to the stored frame."""
    symbol = symbol or config.futures_symbol
    if not symbol:
        return pd.DataFrame()
    if fetcher is not None:
        return fetcher(config, symbol, None)
    try:
        df = fetch_futures(config, symbol)
        if not df.empty:
            return df
    except Exception as exc:  # noqa: BLE001
        log.warning("[futures %s] live fetch failed (%s); using store", symbol, exc)
    import os
    path = config.futures_path_for(symbol)
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index.name = "date"
        return df
    return pd.DataFrame()


def window(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Keep only the last ``days`` of candles (rolling training window)."""
    if df.empty:
        return df
    cutoff = df.index.max() - pd.Timedelta(days=days)
    return df[df.index >= cutoff]
