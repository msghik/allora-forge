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
                       build_features, cross_feature_cols)

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


def main() -> None:
    logging.basicConfig(level=os.environ.get("ALLORA_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    ap = argparse.ArgumentParser(description="Walk-forward feature ablation")
    ap.add_argument("--folds", type=int, default=5)
    walk_forward_ablation(Config.from_env(), n_folds=ap.parse_args().folds)


if __name__ == "__main__":
    main()
