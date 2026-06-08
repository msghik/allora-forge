"""On-chain alt-data via Dune Analytics.

Each canonical metric (``stable_supply``, ``cex_netflow``, ``dex_volume``,
``active_addr``) maps to a Dune **saved query** that returns two columns,
``ts`` (timestamp) and ``value``. We fetch the latest cached results (cheap; set
``dune_execute`` to re-run and spend credits), assemble a time-indexed frame, and
persist it. Everything is best-effort: any failure leaves the metric out and the
feature layer fills it neutral, so the worker never breaks on a Dune hiccup.

Set ``ALLORA_DUNE_API_KEY`` + ``ALLORA_DUNE_QUERY_*`` to enable. The SQL for each
query is in ``docs/ONCHAIN.md``.
"""
from __future__ import annotations

import logging
import os
import time

import pandas as pd

log = logging.getLogger("forge.onchain")

BASE = "https://api.dune.com/api/v1"


def _results(api_key: str, query_id: str, execute: bool) -> list:
    import requests
    headers = {"X-Dune-API-Key": api_key}
    if execute:
        ex = requests.post(f"{BASE}/query/{query_id}/execute", headers=headers, timeout=30)
        ex.raise_for_status()
        eid = ex.json()["execution_id"]
        for _ in range(60):
            st = requests.get(f"{BASE}/execution/{eid}/status", headers=headers, timeout=30).json()
            state = st.get("state")
            if state == "QUERY_STATE_COMPLETED":
                break
            if state in ("QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"):
                raise RuntimeError(f"Dune execution {state}")
            time.sleep(5)
        res = requests.get(f"{BASE}/execution/{eid}/results", headers=headers, timeout=60)
    else:
        res = requests.get(f"{BASE}/query/{query_id}/results", headers=headers, timeout=60)
    res.raise_for_status()
    return res.json()["result"]["rows"]


def _rows_to_series(rows: list) -> pd.Series | None:
    if not rows:
        return None
    keys = list(rows[0].keys())
    tkey = "ts" if "ts" in keys else ("time" if "time" in keys else keys[0])
    vkey = "value" if "value" in keys else next((k for k in keys if k != tkey), None)
    if vkey is None:
        return None
    data = {}
    for r in rows:
        v = r.get(vkey)
        if v is None:
            continue
        t = pd.to_datetime(r[tkey], utc=True).tz_convert(None)
        data[t] = float(v)
    if not data:
        return None
    return pd.Series(data).sort_index()


def fetch_onchain(config) -> pd.DataFrame:
    """Fetch every configured Dune metric into one time-indexed frame (best-effort)."""
    if not config.use_onchain:
        return pd.DataFrame()
    frames = {}
    for metric, qid in config.dune_queries.items():
        try:
            s = _rows_to_series(_results(config.dune_api_key, qid, config.dune_execute))
        except Exception as exc:  # noqa: BLE001
            log.warning("[onchain] %s (query %s) failed: %s", metric, qid, exc)
            continue
        if s is not None and not s.empty:
            frames[metric] = s
            log.info("[onchain] %s: %d points (%s -> %s)", metric, len(s), s.index.min(), s.index.max())
    if not frames:
        return pd.DataFrame()
    out = pd.DataFrame(frames).sort_index()
    out.index.name = "ts"
    return out


def update_onchain(config, fetcher=None) -> pd.DataFrame:
    """Refresh and persist the on-chain store; return the full frame.
    ``fetcher(config)`` overrides the live Dune fetch for tests."""
    if not config.use_onchain:
        return pd.DataFrame()
    fetcher = fetcher or fetch_onchain
    fresh = fetcher(config)
    if fresh is None or fresh.empty:
        log.info("[onchain] no data; features will be neutral")
        return _load_store(config)
    fresh = fresh[~fresh.index.duplicated(keep="last")].sort_index()
    if os.path.exists(config.onchain_path):       # merge with history (Dune windows are limited)
        prev = _load_store(config)
        fresh = pd.concat([prev, fresh])
        fresh = fresh[~fresh.index.duplicated(keep="last")].sort_index()
    config.ensure_dirs()
    fresh.to_csv(config.onchain_path)
    log.info("[onchain] store holds %d rows x %d metrics", len(fresh), fresh.shape[1])
    return fresh


def _load_store(config) -> pd.DataFrame:
    if os.path.exists(config.onchain_path):
        df = pd.read_csv(config.onchain_path, index_col=0, parse_dates=True)
        df.index.name = "ts"
        return df
    return pd.DataFrame()


def get_recent_onchain(config, fetcher=None) -> pd.DataFrame:
    """On-chain frame for a live inference; falls back to the stored frame."""
    if not config.use_onchain:
        return pd.DataFrame()
    if fetcher is not None:
        return fetcher(config)
    try:
        df = fetch_onchain(config)
        if not df.empty:
            return df
    except Exception as exc:  # noqa: BLE001
        log.warning("[onchain] live fetch failed (%s); using store", exc)
    return _load_store(config)
