"""Daily pipeline: ingest -> features -> train -> evaluate (competition metrics)
-> calibrate -> gate -> version -> export. Entrypoint for the scheduler.

    python -m forge.pipeline --once     # one cycle
    python -m forge.pipeline --loop     # run forever, retraining daily
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import time
from datetime import datetime, timedelta, timezone

from . import data, evaluate, export, features, monitor, registry, train
from .config import Config
from .features import build_features

log = logging.getLogger("forge.pipeline")


def _fmt(m: dict) -> str:
    return (f"DA={m['directional_acc']:.3f}(p={m['da_p']:.3f},ci_lo={m['da_ci_low']:.3f}) "
            f"r={m['pearson_r']:.3f}(p={m['pearson_p']:.3f}) "
            f"logasp={m['log_aspect_ratio']:.2f} "
            f"wrmse_impr={m['wrmse_impr']:.2%} zptae_impr={m['zptae_impr']:.2%} n={m['n']}")


def run_once(config: Config, fetcher=None) -> dict:
    """Run one full retrain/calibrate/gate/export cycle."""
    config.ensure_dirs()

    # 1. Ingest fresh data (primary + optional cross-asset) and window it.
    cross_prefix = config.cross_prefix
    df = data.window(data.update_data(config, config.symbol, fetcher=fetcher),
                     config.train_window_days)
    ref_df = None
    if cross_prefix:
        ref_df = data.window(data.update_data(config, config.cross_symbol, fetcher=fetcher),
                             config.train_window_days)

    # 2. Features (single-asset + optional cross-asset) + 1h-ahead target.
    feature_cols = features.active_feature_cols(cross_prefix)
    feats = build_features(df, ref_df=ref_df, cross_prefix=cross_prefix)
    X, y = train.build_target(feats, config.horizon_steps, feature_cols)
    if len(X) < config.min_train_rows:
        raise RuntimeError(f"insufficient data: {len(X)} rows < "
                           f"min_train_rows={config.min_train_rows}")

    # 3. Chronological split + train both candidates.
    X_tr, X_val, y_tr, y_val = train.chrono_split(X, y, config.val_fraction)
    ridge, ridge_p = train.train_ridge(X_tr, y_tr, config)
    lgbm, lgbm_p = train.train_lgbm(X_tr, y_tr, config)

    # 4. For each candidate, search the calibration grid for the ratio that passes
    #    the most whitelist criteria; score on non-overlapping 1h windows.
    step = config.horizon_steps if config.eval_nonoverlap else 1
    cands = {}
    for name, model, params in (("Ridge", ridge, ridge_p), ("LightGBM", lgbm, lgbm_p)):
        raw_tr, raw_val = model.predict(X_tr), model.predict(X_val)
        best = None
        for ratio in config.calibration_ratio_grid:
            scale = train.calibration_scale(y_tr.values, raw_tr, ratio)
            m = evaluate.competition_metrics(y_val.values, raw_val * scale,
                                             step=step, power=config.zptae_power)
            m["whitelist_passed"] = evaluate.whitelist_report(m)["passed"]
            key = (m["whitelist_passed"], m["directional_acc"], m["zptae_impr"])
            if best is None or key > best["key"]:
                best = {"key": key, "ratio": ratio, "metrics": m}
        cands[name] = {"metrics": best["metrics"], "params": params, "ratio": best["ratio"]}
        log.info("candidate %-8s ratio=%.2f | %s wl=%d/%d", name, best["ratio"],
                 _fmt(best["metrics"]), best["metrics"]["whitelist_passed"], len(evaluate.WHITELIST))

    # Winner by whitelist criteria passed, then DA, then Pearson r.
    win_name = max(cands, key=lambda k: (cands[k]["metrics"]["whitelist_passed"],
                                         cands[k]["metrics"]["directional_acc"],
                                         cands[k]["metrics"]["pearson_r"]))
    win = cands[win_name]

    # 5. Refit winner on all (windowed) data; recalibrate at its ratio; export.
    version = registry.new_version_id()
    vdir = config.version_dir(version)
    os.makedirs(vdir, exist_ok=True)
    final_model = train.fit_final(win_name, win["params"], X, y, config)
    scale = train.calibration_scale(y.values, final_model.predict(X), win["ratio"])
    export.export_predict(final_model, list(X.columns), os.path.join(vdir, "predict.pkl"),
                          scale=scale, cross_prefix=cross_prefix)

    wl = evaluate.whitelist_report(win["metrics"])
    log.info("winner=%s ratio=%.2f scale=%.3f whitelist=%d/%d",
             win_name, win["ratio"], scale, wl["passed"], wl["total"])

    metadata = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": registry.git_sha(),
        "model": win_name,
        "params": win["params"],
        "calibration_ratio": win["ratio"],
        "scale": scale,
        "metrics": win["metrics"],          # the metrics the gate compares on (incl. whitelist_passed)
        "whitelist": wl,
        "candidates": {k: v["metrics"] for k, v in cands.items()},
        "data_range": [str(df.index.min()), str(df.index.max())],
        "n_train_rows": len(X_tr),
        "n_val_rows": len(X_val),
        "feature_cols": list(X.columns),
        "n_features": len(feature_cols),
        "timeframe": config.timeframe,
        "horizon_steps": config.horizon_steps,
        "horizon_minutes": config.horizon_minutes,
        "symbol": config.symbol,
        "cross_symbol": config.cross_symbol if cross_prefix else "",
    }
    registry.save_metadata(config, version, metadata)

    # 6. Promotion gate vs current production model.
    current = registry.get_current_metadata(config)
    cur_metrics = current.get("metrics") if current else None
    promote, reason = registry.gate(win["metrics"], cur_metrics, config)
    if promote:
        registry.promote(config, version)
        shutil.copy2(os.path.join(vdir, "predict.pkl"), "predict.pkl")  # convenience for Forge upload
    else:
        monitor.alert(config, f"candidate {version} NOT promoted: {reason}", level="warning")

    m = win["metrics"]
    record = {"ts": datetime.now(timezone.utc).isoformat(), "version": version,
              "symbol": config.symbol, "promoted": promote, "reason": reason,
              "winner": win_name, "scale": scale, "whitelist_passed": wl["passed"],
              "directional_acc": m["directional_acc"], "pearson_r": m["pearson_r"],
              "log_aspect_ratio": m["log_aspect_ratio"], "zptae_impr": m["zptae_impr"],
              "wrmse_impr": m["wrmse_impr"]}
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
    """Retrain immediately if there is no model, then daily at the configured UTC
    hour. Failures are alerted but never kill the loop."""
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
