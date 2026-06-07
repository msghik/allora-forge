"""Monitoring: a metrics history, live-prediction logging, alerting, and
reconciliation of past predictions against realized 24h returns so you can track
the model's *real* out-of-sample skill (not just backtest numbers)."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone

import numpy as np

log = logging.getLogger("forge.monitor")


def append_jsonl(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def alert(config, msg: str, level: str = "error") -> None:
    getattr(log, level if level in ("warning", "error", "info") else "error")(msg)
    if config.alert_webhook:
        try:
            import requests
            requests.post(config.alert_webhook,
                          json={"text": f"[forge:{level}] {msg}"}, timeout=10)
        except Exception:  # noqa: BLE001
            log.exception("failed to post alert webhook")


def log_prediction(config, value: float, ref_close: float, model_version: str) -> None:
    """Record a live inference so it can later be scored against reality."""
    now = datetime.now(timezone.utc)
    append_jsonl(config.predictions_path, {
        "ts": now.isoformat(),
        "mature_at": (now + timedelta(minutes=config.horizon_minutes)).isoformat(),
        "symbol": config.symbol,
        "predicted_log_return": float(value),
        "ref_close": float(ref_close),
        "model_version": model_version,
    })


def reconcile(config) -> dict:
    """Score matured predictions against the realized 24h log return using the
    local data store. Appends results and returns a live scoreboard."""
    from . import data

    preds = read_jsonl(config.predictions_path)
    if not preds:
        return {"n": 0}
    store = data.load_store(config)
    if store.empty:
        return {"n": 0}

    done = {r["ts"] for r in read_jsonl(_recon_path(config))}
    now = datetime.now(timezone.utc)
    new = 0
    for p in preds:
        if p["ts"] in done:
            continue
        mature = datetime.fromisoformat(p["mature_at"])
        if mature > now:
            continue
        # nearest stored close at/after maturity
        future = store[store.index >= mature.replace(tzinfo=None)]
        if future.empty:
            continue
        realized = float(np.log(future["close"].iloc[0] / p["ref_close"]))
        append_jsonl(_recon_path(config), {
            "ts": p["ts"], "predicted": p["predicted_log_return"],
            "realized": realized,
            "correct_direction": bool(np.sign(p["predicted_log_return"]) == np.sign(realized)),
            "model_version": p.get("model_version"),
        })
        new += 1
    board = live_scoreboard(config)
    log.info("reconciled %d new prediction(s); live=%s", new, board)
    return board


def live_scoreboard(config) -> dict:
    rec = read_jsonl(_recon_path(config))
    if not rec:
        return {"n": 0}
    pred = np.array([r["predicted"] for r in rec], dtype=float)
    real = np.array([r["realized"] for r in rec], dtype=float)
    da = float(np.mean(np.sign(pred) == np.sign(real)))
    corr = float(np.corrcoef(pred, real)[0, 1]) if len(rec) > 1 and pred.std() > 0 else 0.0
    return {"n": len(rec), "live_directional_acc": da, "live_pearson_r": corr}


def _recon_path(config) -> str:
    return os.path.join(config.models_dir, "reconciliations.jsonl")
