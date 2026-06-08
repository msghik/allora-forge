"""FastAPI inference server for an Allora worker node.

Exposes the contract ``allora-offchain-node`` calls:

    GET /inference/{token}   -> the model's 24h log-return prediction (a number)

Plus ``/health`` and ``/metadata`` for ops. The current promoted ``predict.pkl``
is loaded once and hot-reloaded whenever the trainer promotes a new model.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from . import data, export, monitor, onchain, registry
from .config import Config

log = logging.getLogger("forge.server")
logging.basicConfig(level=os.environ.get("ALLORA_LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s | %(message)s")

config = Config.from_env()

_predict = None
_mtime = 0.0
_last_value = None   # last good prediction, served on transient failure to preserve liveness


def _maybe_load() -> None:
    """Load (or reload) the current predict.pkl if it changed on disk."""
    global _predict, _mtime
    path = config.current_predict
    if not os.path.exists(path):
        _predict = None
        return
    mtime = os.path.getmtime(path)
    if _predict is None or mtime != _mtime:
        _predict = export.load_predict(path)
        _mtime = mtime
        log.info("loaded model from %s", path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _maybe_load()
    except Exception:  # noqa: BLE001
        log.exception("could not load model at startup (trainer may not have run yet)")
    yield


app = FastAPI(title="Allora Forge Worker", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    _maybe_load()
    meta = registry.get_current_metadata(config) or {}
    return {"status": "ok", "has_model": _predict is not None,
            "model_version": meta.get("version"), "symbol": config.symbol}


@app.get("/metadata")
def metadata() -> dict:
    meta = registry.get_current_metadata(config)
    if not meta:
        raise HTTPException(status_code=503, detail="no model promoted yet")
    return meta


@app.get("/scoreboard")
def scoreboard() -> dict:
    """Live skill on matured predictions (reconciled vs realized 1h returns)."""
    return monitor.live_scoreboard(config)


@app.get("/inference/{token}")
def inference(token: str):
    """Return the 1h log-return prediction for the configured asset.

    Responds with a bare JSON number, matching the allora-offchain-node worker
    contract. The ``token`` path segment is echoed in logs but the worker serves a
    single configured symbol. For **liveness**, a transient failure falls back to
    the last good prediction rather than erroring out (a never-dark worker).
    """
    global _last_value
    _maybe_load()
    if _predict is None:
        raise HTTPException(status_code=503, detail="no model available yet")
    meta = registry.get_current_metadata(config) or {}

    try:
        df = data.get_recent_candles(config, config.symbol)
        if df.empty:
            raise RuntimeError("no market data for primary symbol")
        ref_df = None
        if meta.get("cross_symbol"):  # cross-asset model needs the reference asset too
            ref_df = data.get_recent_candles(config, meta["cross_symbol"])
            if ref_df.empty:
                raise RuntimeError(f"no market data for {meta['cross_symbol']}")
        fut_df = None
        if meta.get("futures_symbol"):  # futures: funding/OI (neutral if empty)
            fut_df = data.get_recent_futures(config, meta["futures_symbol"])
        onchain_df = None
        if meta.get("use_onchain"):     # on-chain: Dune metrics (neutral if empty)
            onchain_df = onchain.get_recent_onchain(config)
        value = _predict(df, ref_df, fut_df, onchain_df)
        monitor.log_prediction(config, value, float(df["close"].iloc[-1]),
                               meta.get("version", "unknown"))
        _last_value = value
        log.info("inference token=%s -> %.6f", token, value)
        return value
    except Exception as exc:  # noqa: BLE001 -- never go dark while a model is loaded
        if _last_value is not None:
            log.warning("inference failed (%s); serving last good value %.6f", exc, _last_value)
            return _last_value
        log.exception("inference failed and no prior value to serve")
        raise HTTPException(status_code=503, detail=f"inference unavailable: {exc}")
