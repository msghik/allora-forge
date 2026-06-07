"""Daily pipeline: ingest -> features -> train -> evaluate -> gate -> version
-> export. This is the entrypoint the scheduler calls every day.

    python -m forge.pipeline --once     # run a single cycle
    python -m forge.pipeline --loop     # run forever, retraining daily
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import time
from datetime import datetime, timedelta, timezone

from . import data, evaluate, export, monitor, registry, train
from .config import Config
from .features import add_features

log = logging.getLogger("forge.pipeline")


def run_once(config: Config, fetcher=None) -> dict:
    """Run one full retrain/gate/export cycle. Returns a result summary."""
    config.ensure_dirs()

    # 1. Ingest fresh data and apply the rolling training window.
    full = data.update_data(config, fetcher=fetcher)
    df = data.window(full, config.train_window_days)

    # 2. Features + 24h target.
    feats = add_features(df)
    X, y = train.build_target(feats, config.horizon_hours)
    if len(X) < config.min_train_rows:
        raise RuntimeError(f"insufficient data: {len(X)} rows < "
                           f"min_train_rows={config.min_train_rows}")

    # 3. Chronological split + train both candidates.
    X_tr, X_val, y_tr, y_val = train.chrono_split(X, y, config.val_fraction)
    ridge, ridge_p = train.train_ridge(X_tr, y_tr, config)
    lgbm, lgbm_p = train.train_lgbm(X_tr, y_tr, config)

    # 4. Out-of-sample evaluation; pick the winner by Pearson r.
    cand = {
        "Ridge": (evaluate.evaluate("Ridge", y_val, ridge.predict(X_val)), ridge_p),
        "LightGBM": (evaluate.evaluate("LightGBM", y_val, lgbm.predict(X_val)), lgbm_p),
    }
    baseline_da = evaluate.baseline_directional_acc(y_val)
    win_name = max(cand, key=lambda k: (cand[k][0]["pearson_r"],
                                        cand[k][0]["directional_acc"]))
    win_metrics, win_params = cand[win_name]
    log.info("winner=%s metrics=%s (baseline_da=%.3f)", win_name, win_metrics, baseline_da)

    # 5. Refit winner on all (windowed) data and export predict.pkl into a version dir.
    version = registry.new_version_id()
    vdir = config.version_dir(version)
    os.makedirs(vdir, exist_ok=True)
    final_model = train.fit_final(win_name, win_params, X, y, config)
    export.export_predict(final_model, list(X.columns), os.path.join(vdir, "predict.pkl"))

    metadata = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": registry.git_sha(),
        "model": win_name,
        "params": win_params,
        "metrics": win_metrics,            # the metrics the gate compares on
        "candidates": {k: v[0] for k, v in cand.items()},
        "baseline_directional_acc": baseline_da,
        "data_range": [str(df.index.min()), str(df.index.max())],
        "n_train_rows": len(X_tr),
        "n_val_rows": len(X_val),
        "feature_cols": list(X.columns),
        "horizon_hours": config.horizon_hours,
        "symbol": config.symbol,
    }
    registry.save_metadata(config, version, metadata)

    # 6. Promotion gate vs current production model.
    current = registry.get_current_metadata(config)
    cur_metrics = current.get("metrics") if current else None
    promote, reason = registry.gate(win_metrics, cur_metrics, config)
    if promote:
        registry.promote(config, version)
        # Convenience copy for manual Forge upload (gitignored).
        shutil.copy2(os.path.join(vdir, "predict.pkl"), "predict.pkl")
    else:
        monitor.alert(config, f"candidate {version} NOT promoted: {reason}", level="warning")

    record = {"ts": datetime.now(timezone.utc).isoformat(), "version": version,
              "promoted": promote, "reason": reason, "winner": win_name,
              "metrics": win_metrics, "baseline_directional_acc": baseline_da}
    registry.append_metrics(config, record)
    log.info("cycle done: version=%s promoted=%s (%s)", version, promote, reason)
    return record


def _seconds_until(hour_utc: int) -> float:
    now = datetime.now(timezone.utc)
    nxt = now.replace(hour=hour_utc, minute=0, second=0, microsecond=0)
    if nxt <= now:
        nxt += timedelta(days=1)
    return (nxt - now).total_seconds()


def run_loop(config: Config) -> None:
    """Retrain immediately if there is no model, then every day at the configured
    UTC hour. Failures are alerted but never kill the loop."""
    if registry.get_current_metadata(config) is None:
        _safe_cycle(config)
    while True:
        secs = _seconds_until(config.retrain_hour_utc)
        log.info("next retrain in %.1f h (at %02d:00 UTC)", secs / 3600, config.retrain_hour_utc)
        time.sleep(secs)
        _safe_cycle(config)


def _safe_cycle(config: Config) -> None:
    try:
        run_once(config)
    except Exception as exc:  # noqa: BLE001 -- keep the daily loop alive
        log.exception("retrain cycle failed")
        monitor.alert(config, f"retrain cycle FAILED: {exc}", level="error")
    try:
        monitor.reconcile(config)
    except Exception:  # noqa: BLE001
        log.exception("reconciliation failed")


def _setup_logging() -> None:
    logging.basicConfig(level=os.environ.get("ALLORA_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")


def main() -> None:
    _setup_logging()
    ap = argparse.ArgumentParser(description="Forge daily retrain pipeline")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--once", action="store_true", help="run a single cycle")
    g.add_argument("--loop", action="store_true", help="run forever, daily")
    args = ap.parse_args()
    config = Config.from_env()
    if args.once:
        run_once(config)
    else:
        run_loop(config)


if __name__ == "__main__":
    main()
