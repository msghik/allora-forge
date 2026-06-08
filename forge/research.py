"""Walk-forward ablation -- measure each feature block's *robust* contribution.

Single 1-fold validation at n~1747 has a DA CI of ~+/-0.023, so a single run can't
tell signal from noise (the reason DA bounced 0.510 -> 0.521 -> 0.495 across runs).
This evaluates each feature configuration over several expanding walk-forward folds
and reports mean +/- std DA, so feature decisions are evidence-based.

    python -m forge.research            # uses the local stores (no refetch)
    python -m forge.research --folds 6

Calibration is irrelevant here (DA and Pearson r are invariant to positive
scaling), so we score the raw model outputs directly.
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd

from . import data, evaluate, onchain, train
from .config import Config
from .features import (FEATURE_COLS, FUTURES_COLS, ONCHAIN_COLS, ORDERFLOW_COLS,
                       active_feature_cols, build_features, cross_feature_cols)

log = logging.getLogger("forge.research")


def _load_full_features(config: Config) -> pd.DataFrame:
    """Build the full feature frame from the local stores (every block on)."""
    df = data.window(data.load_store(config, config.symbol), config.train_window_days)
    if df.empty:
        raise RuntimeError("no local OHLCV store; run the pipeline once first")
    ref = None
    if config.cross_prefix:
        ref = data.window(data.load_store(config, config.cross_symbol), config.train_window_days)
    fut = None
    if config.use_futures:
        p = config.futures_path_for(config.futures_symbol)
        if os.path.exists(p):
            fut = pd.read_csv(p, index_col=0, parse_dates=True)
    oc = onchain._load_store(config) if config.use_onchain else None
    return build_features(df, ref_df=ref, cross_prefix=config.cross_prefix, fut_df=fut,
                          onchain_df=oc, use_orderflow=True, use_futures=config.use_futures,
                          use_onchain=config.use_onchain)


def _configs(config: Config) -> dict:
    """Cumulative feature-block configs, including only blocks actually built
    (futures/on-chain are present only when enabled)."""
    base = list(FEATURE_COLS)
    cross = cross_feature_cols(config.cross_prefix) if config.cross_prefix else []
    bc = base + cross
    cfgs = {"base": base}
    if cross:
        cfgs["base+cross"] = bc
    cfgs["+orderflow"] = bc + list(ORDERFLOW_COLS)
    if config.use_futures:
        cfgs["+futures"] = bc + list(FUTURES_COLS)
        cfgs["+of+fut"] = bc + list(ORDERFLOW_COLS) + list(FUTURES_COLS)
    if config.use_onchain:
        full = bc + list(ORDERFLOW_COLS)
        if config.use_futures:
            full += list(FUTURES_COLS)
        cfgs["+all+onchain"] = full + list(ONCHAIN_COLS)
    return cfgs


def _folds(n: int, k: int):
    """k contiguous validation blocks over the last ~half of the data (expanding train)."""
    val_len = n // (2 * k)
    start = n - k * val_len
    return [(start + i * val_len, start + (i + 1) * val_len) for i in range(k)], val_len


def _score(pred, y, step):
    p, t = np.asarray(pred)[::step], np.asarray(y)[::step]
    da = float(np.mean(np.sign(p) == np.sign(t)))
    r, _ = evaluate._pearson(t, p)
    return da, r


def walk_forward_ablation(config: Config, n_folds: int = 5) -> dict:
    feats = _load_full_features(config)
    cfgs = _configs(config)
    step = config.horizon_steps if config.eval_nonoverlap else 1
    purge = config.purge

    # one target; reuse the same rows for every config (fair comparison)
    all_cols = sorted({c for cols in cfgs.values() for c in cols})
    X_all, y = train.build_target(feats, config.horizon_steps, all_cols)
    n = len(X_all)
    folds, val_len = _folds(n, n_folds)
    log.info("ablation: %d rows, %d folds x %d val rows (~%d non-overlap/fold), purge=%d",
             n, n_folds, val_len, val_len // step, purge)

    results = {name: {"lgbm_da": [], "ridge_da": [], "lgbm_r": []} for name in cfgs}
    for fi, (v0, v1) in enumerate(folds):
        tr_end = max(0, v0 - purge)
        idx_tr, idx_va = X_all.index[:tr_end], X_all.index[v0:v1]
        y_tr, y_va = y.loc[idx_tr], y.loc[idx_va]
        w = train.recency_weights(idx_tr, config.recency_half_life_days)
        for name, cols in cfgs.items():
            Xtr, Xva = X_all.loc[idx_tr, cols], X_all.loc[idx_va, cols]
            ridge = train.fit_ridge(10.0, Xtr, y_tr, w)
            lgbm = train.make_lgbm(config, n_estimators=600)
            lgbm.fit(Xtr, y_tr, sample_weight=w)
            r_da, _ = _score(ridge.predict(Xva), y_va.values, step)
            l_da, l_r = _score(lgbm.predict(Xva), y_va.values, step)
            results[name]["ridge_da"].append(r_da)
            results[name]["lgbm_da"].append(l_da)
            results[name]["lgbm_r"].append(l_r)
        log.info("fold %d/%d done (val %s -> %s)", fi + 1, n_folds,
                 idx_va.min(), idx_va.max())

    print(f"\n{'config':<24} {'LGBM DA':>16} {'Ridge DA':>16} {'LGBM r':>9}  {'DA>.55':>6}")
    print("-" * 76)
    summary = {}
    for name, r in results.items():
        lda, rda, lr = np.array(r["lgbm_da"]), np.array(r["ridge_da"]), np.array(r["lgbm_r"])
        best = max(lda.mean(), rda.mean())
        frac = float(np.mean(np.maximum(lda, rda) > 0.55))
        summary[name] = {"lgbm_da": lda.mean(), "ridge_da": rda.mean(),
                         "lgbm_da_std": lda.std(), "lgbm_r": lr.mean(), "best": best}
        print(f"{name:<24} {lda.mean():.3f} +/- {lda.std():.3f}   "
              f"{rda.mean():.3f} +/- {rda.std():.3f}   {lr.mean():>6.3f}  {frac:>6.0%}")
    print("-" * 76)
    winner = max(summary, key=lambda k: summary[k]["best"])
    print(f"best mean DA: {winner} ({summary[winner]['best']:.3f})  "
          f"-- baseline 0.5; CI half-width ~{0.98 / np.sqrt(val_len // step):.3f}/fold\n")
    return summary


def tune_lgbm(config: Config, n_folds: int = 5) -> list:
    """Grid-search LightGBM hyperparameters by **mean walk-forward DA** on the active
    feature config (the metric we're judged on, not MSE). Prints the ranking."""
    feats = _load_full_features(config)
    cols = active_feature_cols(config.cross_prefix, True, config.use_futures, config.use_onchain)
    X, y = train.build_target(feats, config.horizon_steps, cols)
    n = len(X)
    folds, val_len = _folds(n, n_folds)
    step = config.horizon_steps if config.eval_nonoverlap else 1
    grid = [{"num_leaves": nl, "max_depth": md, "min_child_samples": mc,
             "n_estimators": ne, "learning_rate": lr}
            for nl in (15, 31, 63) for md in (3, 5) for mc in (100, 400)
            for ne in (600,) for lr in (0.02, 0.05)]
    log.info("tuning %d param sets x %d folds on %d features (%d rows)", len(grid), n_folds, len(cols), n)

    scored = []
    for params in grid:
        das, rs = [], []
        for v0, v1 in folds:
            tr_end = max(0, v0 - config.purge)
            idx_tr, idx_va = X.index[:tr_end], X.index[v0:v1]
            w = train.recency_weights(idx_tr, config.recency_half_life_days)
            m = train.make_lgbm(config, **params)
            m.fit(X.loc[idx_tr], y.loc[idx_tr], sample_weight=w)
            da, r = _score(m.predict(X.loc[idx_va]), y.loc[idx_va].values, step)
            das.append(da)
            rs.append(r)
        scored.append((float(np.mean(das)), float(np.std(das)), float(np.mean(rs)), params))
    scored.sort(key=lambda t: -t[0])

    print(f"\n{'mean DA':>9} {'std':>6} {'mean r':>7}  params")
    print("-" * 78)
    for da, sd, r, p in scored[:8]:
        ps = f"leaves={p['num_leaves']} depth={p['max_depth']} min_child={p['min_child_samples']} lr={p['learning_rate']}"
        print(f"{da:>9.3f} {sd:>6.3f} {r:>7.3f}  {ps}")
    best = scored[0]
    print("-" * 78)
    print(f"best: mean DA {best[0]:.3f} +/- {best[1]:.3f} | {best[3]}")
    print("set these in forge/config.py lgbm_params if they beat the current default.\n")
    return scored


def main() -> None:
    logging.basicConfig(level=os.environ.get("ALLORA_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    ap = argparse.ArgumentParser(description="Walk-forward feature ablation / tuning")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tune", action="store_true", help="DA-targeted LightGBM hyperparameter search")
    args = ap.parse_args()
    config = Config.from_env()
    if args.tune:
        tune_lgbm(config, n_folds=args.folds)
    else:
        walk_forward_ablation(config, n_folds=args.folds)


if __name__ == "__main__":
    main()
