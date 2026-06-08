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

import numpy as np

from . import data, estimators, evaluate, export, features, monitor, registry, train
from .config import Config
from .features import build_features

log = logging.getLogger("forge.pipeline")


def _fmt(m: dict) -> str:
    return (f"DA={m['directional_acc']:.3f}(p={m['da_p']:.3f},ci_lo={m['da_ci_low']:.3f}) "
            f"r={m['pearson_r']:.3f}(p={m['pearson_p']:.3f}) "
            f"logasp={m['log_aspect_ratio']:.2f} "
            f"wrmse_impr={m['wrmse_impr']:.2%} zptae_impr={m['zptae_impr']:.2%} n={m['n']}")


def _std(a) -> float:
    s = float(np.std(np.asarray(a, dtype=float)))
    return s or 1.0


def _build_candidates(X_tr, y_tr, config):
    """Train base learners on the train split and assemble candidate predictors
    (each exposing ``.predict``). Returns ``[{name, model, kind, weight}], params``.
    """
    w = train.recency_weights(X_tr.index, config.recency_half_life_days)
    w_clf = train._combine_weights(w, y_tr.values)

    ridge, ridge_p = train.train_ridge(X_tr, y_tr, config, sample_weight=w)
    lgbm, lgbm_p = train.train_lgbm(X_tr, y_tr, config, sample_weight=w)
    params = {"alpha": ridge_p["alpha"], "lgbm_n": lgbm_p["n_estimators"]}

    cands = [{"name": "Ridge", "model": ridge, "kind": "ridge"},
             {"name": "LightGBM", "model": lgbm, "kind": "lgbm"}]

    if config.use_classifier:
        clf, clf_p = train.train_lgbm_classifier(X_tr, y_tr, config, sample_weight=w_clf)
        params["clf_n"] = clf_p["n_estimators"]
        cands.append({"name": "LGBM-Clf", "kind": "clf",
                      "model": estimators.SignMagnitudePredictor(clf)})
        if config.use_ensemble:
            reg_std = _std(lgbm.predict(X_tr))
            sgn_std = _std(2.0 * clf.predict_proba(X_tr)[:, 1] - 1.0)
            for wt in config.ensemble_weights:
                cands.append({"name": f"Blend-{wt:.2f}", "kind": "blend", "weight": wt,
                              "model": estimators.BlendPredictor(lgbm, clf, wt, reg_std, sgn_std)})
    return cands, params


def _refit_winner(win, params, X, y, config):
    """Refit the winning candidate's base learners on the full (windowed) data."""
    w = train.recency_weights(X.index, config.recency_half_life_days)
    kind = win["kind"]
    if kind == "ridge":
        return train.fit_ridge(params["alpha"], X, y, w)
    if kind == "lgbm":
        return train.fit_final("LightGBM", {"n_estimators": params["lgbm_n"]}, X, y, config, w)

    # classifier-based candidates
    w_clf = train._combine_weights(w, y.values)
    clf = train.make_lgbm_classifier(config, n_estimators=params["clf_n"])
    clf.fit(X, (y.values > 0).astype(int), sample_weight=w_clf)
    if kind == "clf":
        return estimators.SignMagnitudePredictor(clf)
    lgbm = train.fit_final("LightGBM", {"n_estimators": params["lgbm_n"]}, X, y, config, w)
    reg_std = _std(lgbm.predict(X))
    sgn_std = _std(2.0 * clf.predict_proba(X)[:, 1] - 1.0)
    return estimators.BlendPredictor(lgbm, clf, win["weight"], reg_std, sgn_std)


def run_once(config: Config, fetcher=None, futures_fetcher=None) -> dict:
    """Run one full retrain/calibrate/gate/export cycle."""
    config.ensure_dirs()

    # 1. Ingest fresh data (primary + optional cross-asset + optional futures).
    cross_prefix = config.cross_prefix
    df = data.window(data.update_data(config, config.symbol, fetcher=fetcher),
                     config.train_window_days)
    ref_df = None
    if cross_prefix:
        ref_df = data.window(data.update_data(config, config.cross_symbol, fetcher=fetcher),
                             config.train_window_days)
    fut_df = None
    if config.use_futures:
        fut_df = data.update_futures(config, config.futures_symbol, fetcher=futures_fetcher)

    # 2. Features (all enabled blocks) + 1h-ahead target.
    feature_cols = features.active_feature_cols(cross_prefix, config.use_orderflow,
                                                config.use_futures)
    feats = build_features(df, ref_df=ref_df, cross_prefix=cross_prefix, fut_df=fut_df,
                           use_orderflow=config.use_orderflow, use_futures=config.use_futures)
    X, y = train.build_target(feats, config.horizon_steps, feature_cols)
    if len(X) < config.min_train_rows:
        raise RuntimeError(f"insufficient data: {len(X)} rows < "
                           f"min_train_rows={config.min_train_rows}")

    # 3. Purged chronological split + train all candidates.
    X_tr, X_val, y_tr, y_val = train.chrono_split(X, y, config.val_fraction, config.purge)
    cands, params = _build_candidates(X_tr, y_tr, config)

    # 4. For each candidate, search the calibration grid for the ratio that passes
    #    the most whitelist criteria; score on non-overlapping 1h windows.
    step = config.horizon_steps if config.eval_nonoverlap else 1
    for cand in cands:
        raw_tr, raw_val = cand["model"].predict(X_tr), cand["model"].predict(X_val)
        best = None
        for ratio in config.calibration_ratio_grid:
            scale = train.calibration_scale(y_tr.values, raw_tr, ratio)
            m = evaluate.competition_metrics(y_val.values, raw_val * scale,
                                             step=step, power=config.zptae_power)
            m["whitelist_passed"] = evaluate.whitelist_report(m)["passed"]
            key = (m["whitelist_passed"], m["directional_acc"], m["zptae_impr"])
            if best is None or key > best["key"]:
                best = {"key": key, "ratio": ratio, "metrics": m}
        cand["metrics"], cand["ratio"] = best["metrics"], best["ratio"]
        log.info("candidate %-10s ratio=%.2f | %s wl=%d/%d", cand["name"], cand["ratio"],
                 _fmt(cand["metrics"]), cand["metrics"]["whitelist_passed"], len(evaluate.WHITELIST))

    # Winner by whitelist criteria passed, then DA, then Pearson r.
    win = max(cands, key=lambda c: (c["metrics"]["whitelist_passed"],
                                    c["metrics"]["directional_acc"],
                                    c["metrics"]["pearson_r"]))

    # 5. Refit winner on all (windowed) data; recalibrate at its ratio; export.
    version = registry.new_version_id()
    vdir = config.version_dir(version)
    os.makedirs(vdir, exist_ok=True)
    final_model = _refit_winner(win, params, X, y, config)
    scale = train.calibration_scale(y.values, final_model.predict(X), win["ratio"])
    export.export_predict(final_model, list(X.columns), os.path.join(vdir, "predict.pkl"),
                          scale=scale, cross_prefix=cross_prefix,
                          use_orderflow=config.use_orderflow, use_futures=config.use_futures)

    wl = evaluate.whitelist_report(win["metrics"])
    log.info("winner=%s ratio=%.2f scale=%.3f whitelist=%d/%d",
             win["name"], win["ratio"], scale, wl["passed"], wl["total"])

    metadata = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": registry.git_sha(),
        "model": win["name"],
        "params": params,
        "calibration_ratio": win["ratio"],
        "scale": scale,
        "metrics": win["metrics"],          # the metrics the gate compares on (incl. whitelist_passed)
        "whitelist": wl,
        "candidates": {c["name"]: c["metrics"] for c in cands},
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
        "use_orderflow": config.use_orderflow,
        "use_futures": config.use_futures,
        "futures_symbol": config.futures_symbol if config.use_futures else "",
        "recency_half_life_days": config.recency_half_life_days,
        "purge_steps": config.purge,
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
              "winner": win["name"], "scale": scale, "whitelist_passed": wl["passed"],
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
