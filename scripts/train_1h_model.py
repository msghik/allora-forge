#!/usr/bin/env python3
"""Train + evaluate + export a predict.pkl for the 1h BTC/USD log-return topic.

Target (per the competition spec): log_return = ln(price_{t+1h} / price_t),
i.e. ``target_bars=1`` at ``interval="1h"``. The exported ``predict(nonce)``
returns the predicted **log-return as a float** — for log-return topics you
submit the log-return itself, NOT a price.

v2 — designed around the three failure modes of v1 (0-2/7 grades):

* compact stationary features (~30) instead of 240 raw normalized OHLCV
  columns — at 1h the signal-to-noise ratio is brutal and feature count is
  variance you pay for;
* the model learns a **vol-normalized** target z = r / sigma100 (sigma100 =
  trailing std of the last 100 hourly log-returns, the same reference std the
  competition's ZPTAE uses). This makes 2024 and 2026 regimes comparable, so
  long histories help instead of hurting;
* **recency-weighted** training (exponential half-life) instead of truncating
  history — old samples fade, they aren't thrown away;
* Huber objective (fat tails shouldn't drag the fit) and a final **shrinkage
  calibration**: predictions are scaled by the lambda that maximizes
  ZPTAE-proxy improvement vs the zero baseline, subject to the whitelist's
  log-aspect-ratio bound |log10(std(pred)/std(true))| <= 0.5;
* two model families A/B'd on the same folds: a return **regressor** and a
  sign **classifier** mapped to (2*p_up - 1) * sigma100 — when the edge is
  directional-only (DA significant, Pearson r ~ 0) the classifier puts the
  model's capacity where the signal actually is.

Requires the Allora Forge Builder Kit and an ALLORA_API_KEY (free at
https://developer.allora.network). Run as a script from the repo root —
cloudpickle captures closures by value only when this file executes as
__main__:

    export ALLORA_API_KEY=UP-...
    python scripts/train_1h_model.py

Env knobs: DAYS_OF_HISTORY (1000), INPUT_BARS (128), HALF_LIFE_DAYS (270),
VOL_NORM_TARGET (1; set 0 to A/B raw-target training), FAMILIES (reg,clf),
STRICT_ASPECT (0; set 1 to always keep the whitelist aspect band),
DATA_SOURCE (allora|binance), PREDICT_PKL.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import cloudpickle
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit

from allora_forge_builder_kit import AlloraMLWorkflow, PerformanceEvaluator

# --- competition task: 1h BTC/USD log-return ---
INTERVAL = "1h"
TARGET_BARS = 1                 # 1 bar ahead at 1h interval = 1 hour
INPUT_BARS = int(os.environ.get("INPUT_BARS", "128"))   # >=101 so sigma100 fits
DAYS_OF_HISTORY = int(os.environ.get("DAYS_OF_HISTORY", "1000"))

# --- regime handling ---
SIGMA_WINDOW = 100              # matches the topic's ZPTAE reference std
VOL_NORM_TARGET = os.environ.get("VOL_NORM_TARGET", "1") != "0"
HALF_LIFE_DAYS = float(os.environ.get("HALF_LIFE_DAYS", "270"))
Z_CLIP = 6.0                    # clip vol-normalized target tails

# --- model search (small, conservative grid; expand once the loop works) ---
# Two families are searched and A/B'd on the same folds:
#   reg — Huber regression on the (vol-normalized) return
#   clf — sign classifier; prediction = (2*p_up - 1) * sigma100, i.e. direction
#         conviction scaled by current vol. Wins when the edge is directional
#         (DA significant) but magnitude correlation is ~0.
N_SPLITS = 3
N_ESTIMATORS_MAX = 800
N_ESTIMATORS_CHECKPOINTS = [200, 400, 800]
LEARNING_RATES = [0.02, 0.05]
MAX_DEPTHS = [3, 5]
NUM_LEAVES = [15, 31]
FIXED_COMMON = dict(
    min_child_samples=100, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbose=-1,
)
REG_EXTRA = dict(objective="huber", alpha=1.0)
FAMILIES = tuple(os.environ.get("FAMILIES", "reg,clf").split(","))

# --- magnitude calibration ---
SHRINK_GRID = [0.25, 0.4, 0.6, 0.8, 1.0, 1.3]
ASPECT_BOUND = 0.5              # whitelist: |log10(std(pred)/std(true))| < 0.5
# STRICT_ASPECT=1 always picks a whitelist-compliant loudness, even when a
# quieter scale would score better on ZPTAE (leaderboard vs whitelist tradeoff).
STRICT_ASPECT = os.environ.get("STRICT_ASPECT", "0") != "0"

RET_LAGS = (1, 2, 3, 4, 6, 12, 24, 48, 96)
VOL_WINDOWS = (6, 24, 96)

OUT_PKL = os.environ.get("PREDICT_PKL", "predict.pkl")


def resolve_data_source() -> tuple[str, list[str], dict]:
    """Prefer the Allora/Tiingo source (matches how topics are scored);
    fall back to Binance klines when no API key is available."""
    forced = os.environ.get("DATA_SOURCE", "").strip().lower()
    api_key = os.environ.get("ALLORA_API_KEY", "").strip()
    if not api_key:
        for p in (".allora_api_key", os.path.join(os.path.dirname(__file__), "..", ".allora_api_key")):
            if os.path.exists(p):
                api_key = open(p).read().strip()
                break
    if forced == "binance" or (not api_key and forced != "allora"):
        print("Data source: binance (no ALLORA_API_KEY found)" if not api_key else "Data source: binance (forced)")
        return "binance", ["BTCUSDT"], {}
    if not api_key:
        sys.exit("ALLORA_API_KEY required for DATA_SOURCE=allora — get one free at https://developer.allora.network")
    print("Data source: allora (Tiingo via Atlas)")
    return "allora", ["btcusd"], {"api_key": api_key}


def matrices_from_workflow_df(df: pd.DataFrame, n_bars: int):
    """Pull the kit's normalized OHLCV window back into (n, bars) matrices."""
    cols = lambda f: [f"feature_{f}_{i}" for i in range(n_bars)]
    C = df[cols("close")].to_numpy(dtype=float)
    H = df[cols("high")].to_numpy(dtype=float)
    L = df[cols("low")].to_numpy(dtype=float)
    V = df[cols("volume")].to_numpy(dtype=float)
    when = pd.to_datetime(df["open_time"], utc=True)
    return C, H, L, V, when


def compact_features(C, H, L, V, when) -> tuple[pd.DataFrame, np.ndarray]:
    """~30 stationary features per row + sigma100 (trailing hourly-return std).

    OHLC are normalized by the window's last close, so log-return and range
    features computed here equal those on raw prices (scale cancels).
    """
    eps = 1e-12
    B = C.shape[1]
    logc = np.log(np.maximum(C, eps))
    r1 = np.diff(logc, axis=1)                       # (n, B-1) hourly log-returns
    sw = min(SIGMA_WINDOW, r1.shape[1])
    sigma100 = r1[:, -sw:].std(axis=1) + eps

    f: dict[str, np.ndarray] = {}
    for lag in RET_LAGS:
        if lag <= B - 1:
            f[f"ret_{lag}"] = logc[:, -1] - logc[:, -1 - lag]
    for w in VOL_WINDOWS:
        if w <= r1.shape[1]:
            f[f"rv_{w}"] = r1[:, -w:].std(axis=1)
    if "rv_96" in f:
        f["rv_ratio_6_96"] = f["rv_6"] / (f["rv_96"] + eps)
        f["rv_ratio_24_96"] = f["rv_24"] / (f["rv_96"] + eps)
    for lag in (1, 6, 24):                            # scale-free momentum
        if f"ret_{lag}" in f:
            f[f"zret_{lag}"] = f[f"ret_{lag}"] / (sigma100 * np.sqrt(lag))
    g = np.clip(r1[:, -14:], 0, None).mean(axis=1)    # RSI-style balance
    l = np.clip(-r1[:, -14:], 0, None).mean(axis=1)
    f["rsi_14"] = 100.0 * g / (g + l + eps)
    hi24 = H[:, -24:].max(axis=1)
    lo24 = L[:, -24:].min(axis=1)
    f["hl_range_24"] = hi24 - lo24                    # already in last-close units
    f["range_pos_24"] = (C[:, -1] - lo24) / (hi24 - lo24 + eps)
    f["vol_z_24"] = (V[:, -1] - V[:, -24:].mean(axis=1)) / (V[:, -24:].std(axis=1) + eps)
    f["vol_trend"] = V[:, -6:].mean(axis=1) / (V[:, -48:].mean(axis=1) + eps)
    f["sigma_100"] = sigma100
    hour = when.dt.hour.to_numpy()
    dow = when.dt.dayofweek.to_numpy()
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    f["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return pd.DataFrame(f, index=when.index), sigma100


def power_tanh(x: np.ndarray, p: float = 1.5) -> np.ndarray:
    """Smooth bounded transform — a stand-in for the competition's power-tanh.
    Only used to compare *relative* loss vs the zero baseline."""
    return np.tanh(np.abs(x)) ** p


def zptae_proxy(y_true: np.ndarray, y_pred: np.ndarray, window: int = SIGMA_WINDOW) -> float:
    """Mean power-tanh of |error| z-scored by the trailing std of the last
    ``window`` ground-truth log-returns (ref mean 0), like the topic's loss."""
    s = pd.Series(y_true)
    sigma = s.rolling(window, min_periods=window).std().shift(1).to_numpy()
    ok = np.isfinite(sigma) & (sigma > 0)
    if ok.sum() == 0:
        return float("nan")
    z = np.abs(y_pred[ok] - y_true[ok]) / sigma[ok]
    return float(np.mean(power_tanh(z)))


def aspect_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.log10((y_pred.std() + 1e-15) / (y_true.std() + 1e-15)))


def tune_shrink(y_true: np.ndarray, y_pred: np.ndarray, verbose: bool = False) -> tuple[float, float, float]:
    """Pick a scale lambda for the predictions, balancing two pulls:

    * ZPTAE/WRMSE want weak signals scaled way down (the RMSE-optimal scale is
      lambda* = cov(pred, true)/var(pred), tiny when correlation is low);
    * the whitelist's |log10(std(pred)/std(true))| <= ASPECT_BOUND sets a
      *floor* on loudness — silence is not an option.

    Candidates: fixed grid + lambda* + the aspect-feasibility boundaries.
    Policy: take the best feasible lambda unless the unconstrained best beats
    it by more than 2pp of improvement, in which case prefer score (set
    STRICT_ASPECT=1 to always stay in the whitelist band).
    Returns (lambda, zptae_improvement, aspect).
    """
    eps = 1e-15
    zp_zero = zptae_proxy(y_true, np.zeros_like(y_pred))
    sy, sp = y_true.std() + eps, y_pred.std() + eps
    lam_floor = (10.0 ** -ASPECT_BOUND) * sy / sp     # quietest feasible
    lam_ceil = (10.0 ** ASPECT_BOUND) * sy / sp       # loudest feasible
    lam_star = float(np.dot(y_pred, y_true) / (np.dot(y_pred, y_pred) + eps))
    cands = set(SHRINK_GRID) | {lam_floor * 1.01, lam_ceil * 0.99}
    if lam_star > 0:
        cands |= {lam_star, min(max(lam_star, lam_floor * 1.01), lam_ceil * 0.99)}

    best, best_feasible = None, None
    for lam in sorted(c for c in cands if c > 0):
        zp = zptae_proxy(y_true, lam * y_pred)
        imp = 1.0 - zp / zp_zero if zp_zero and np.isfinite(zp) else float("-inf")
        asp = aspect_ratio(y_true, lam * y_pred)
        cand = (imp, lam, asp)
        if best is None or imp > best[0]:
            best = cand
        if abs(asp) <= ASPECT_BOUND and (best_feasible is None or imp > best_feasible[0]):
            best_feasible = cand
    chosen = best
    if best_feasible is not None and (STRICT_ASPECT or best_feasible[0] >= best[0] - 0.02):
        chosen = best_feasible
    elif verbose and best_feasible is not None:
        print(f"  note: taking lambda={best[1]:.3f} for score; aspect {best[2]:+.2f} "
              f"violates the +/-{ASPECT_BOUND} whitelist bound "
              f"(best feasible was {best_feasible[0]:+.2%} at lambda={best_feasible[1]:.3f};"
              f" STRICT_ASPECT=1 forces compliance)")
    imp, lam, asp = chosen
    return lam, imp, asp


def main() -> None:
    print("=" * 78)
    print("1h BTC/USD log-return — train / evaluate / export (v2)")
    print("=" * 78)
    print(f"history={DAYS_OF_HISTORY}d  input_bars={INPUT_BARS}  "
          f"vol_norm_target={VOL_NORM_TARGET}  half_life={HALF_LIFE_DAYS}d")

    data_source, tickers, dm_kwargs = resolve_data_source()
    workflow = AlloraMLWorkflow(
        tickers=tickers,
        number_of_input_bars=INPUT_BARS,
        target_bars=TARGET_BARS,
        interval=INTERVAL,
        data_source=data_source,
        **dm_kwargs,
    )

    start_date = datetime.now(timezone.utc) - timedelta(days=DAYS_OF_HISTORY)
    print(f"\n[1/5] Backfilling {DAYS_OF_HISTORY} days of {INTERVAL} {tickers} ...")
    try:
        workflow.backfill(start=start_date)
    except Exception as e:  # cached parquet may still cover us
        print(f"  backfill warning: {e} — trying locally cached data")

    print("[2/5] Building features ...")
    df_all = workflow.get_full_feature_target_dataframe(start_date=start_date).reset_index()
    df_all = df_all.dropna(subset=["target"]).reset_index(drop=True)

    C, H, L, V, when = matrices_from_workflow_df(df_all, INPUT_BARS)
    X, sigma100 = compact_features(C, H, L, V, when)
    feature_cols = list(X.columns)
    y_raw = df_all["target"].to_numpy(dtype=float)
    y_train_space = np.clip(y_raw / sigma100, -Z_CLIP, Z_CLIP) if VOL_NORM_TARGET else y_raw

    # Recency weights: a sample HALF_LIFE_DAYS old counts half as much.
    age_days = (when.max() - when).dt.total_seconds().to_numpy() / 86400.0
    weights = 0.5 ** (age_days / HALF_LIFE_DAYS)

    print(f"  {len(X):,} samples, {len(feature_cols)} features "
          f"({when.min()} → {when.max()})")

    y_sign = (y_raw > 0).astype(int)

    print("[3/5] Walk-forward grid search (selecting on calibrated ZPTAE-proxy improvement) ...")
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=TARGET_BARS)
    evaluator = PerformanceEvaluator()
    results = []
    n = 0
    for family in FAMILIES:
        for lr in LEARNING_RATES:
            for depth in MAX_DEPTHS:
                for leaves in NUM_LEAVES:
                    fold_models = []
                    for train_idx, test_idx in tscv.split(X):
                        if family == "clf":
                            m = LGBMClassifier(n_estimators=N_ESTIMATORS_MAX, learning_rate=lr,
                                               max_depth=depth, num_leaves=leaves, **FIXED_COMMON)
                            m.fit(X.iloc[train_idx], y_sign[train_idx],
                                  sample_weight=weights[train_idx])
                        else:
                            m = LGBMRegressor(n_estimators=N_ESTIMATORS_MAX, learning_rate=lr,
                                              max_depth=depth, num_leaves=leaves,
                                              **FIXED_COMMON, **REG_EXTRA)
                            m.fit(X.iloc[train_idx], y_train_space[train_idx],
                                  sample_weight=weights[train_idx])
                        fold_models.append((m, test_idx))
                    for n_est in N_ESTIMATORS_CHECKPOINTS:
                        n += 1
                        pred = np.full(len(X), np.nan)
                        for m, test_idx in fold_models:
                            if family == "clf":
                                p_up = m.predict_proba(X.iloc[test_idx], num_iteration=n_est)[:, 1]
                                pred[test_idx] = 2.0 * p_up - 1.0   # conviction in [-1, 1]
                            else:
                                pred[test_idx] = m.predict(X.iloc[test_idx], num_iteration=n_est)
                        mask = np.isfinite(pred)
                        # back to log-return units before any scoring: the classifier's
                        # conviction is always vol-scaled, the regressor only if it was
                        # trained in z-space
                        scale = sigma100[mask] if (family == "clf" or VOL_NORM_TARGET) else 1.0
                        pred_lr = pred[mask] * scale
                        lam, imp, asp = tune_shrink(y_raw[mask], pred_lr)
                        da = float(np.mean(np.sign(pred_lr) == np.sign(y_raw[mask])))
                        results.append({"family": family, "n_estimators": n_est,
                                        "learning_rate": lr, "max_depth": depth,
                                        "num_leaves": leaves, "lambda": lam,
                                        "zptae_imp": imp, "aspect": asp,
                                        "da": da, "mask": mask, "pred_lr": pred_lr})
                        print(f"  [{family} {n:2d}] n={n_est:3d} lr={lr:.2f} d={depth} l={leaves:2d} -> "
                              f"zptae_imp={imp:+.2%} (lam={lam:.2f}, aspect={asp:+.2f}, DA={da:.4f})")

    results.sort(key=lambda r: (r["zptae_imp"], r["da"]), reverse=True)
    for family in FAMILIES:
        fam_best = next(r for r in results if r["family"] == family)
        print(f"  best {family}: zptae_imp={fam_best['zptae_imp']:+.2%} DA={fam_best['da']:.4f} "
              f"aspect={fam_best['aspect']:+.2f}")
    best = results[0]
    lam = best["lambda"]
    print(f"\n[4/5] Best: family={best['family']} n={best['n_estimators']} "
          f"lr={best['learning_rate']} d={best['max_depth']} l={best['num_leaves']} lambda={lam:.2f}")
    print("  NOTE: lambda and config were chosen on the same OOS folds — expect the live")
    print("  numbers to be a bit weaker. The full 7-metric report on calibrated preds:")
    y_oos = y_raw[best["mask"]]
    # re-run calibration verbosely once, so any score-vs-whitelist tradeoff is shown
    tune_shrink(y_oos, best["pred_lr"], verbose=True)
    p_oos = lam * best["pred_lr"]
    report = evaluator.evaluate(y_true=pd.Series(y_oos), y_pred=pd.Series(p_oos))
    evaluator.print_report(report, detailed=False)
    zp_imp = best["zptae_imp"]
    print(f"  ZPTAE proxy improvement vs zero: {zp_imp:+.2%} (whitelist target > +20%)")
    print(f"  log-aspect ratio: {aspect_ratio(y_oos, p_oos):+.3f} (whitelist: within ±{ASPECT_BOUND})")

    print("[5/5] Training production model on all data and exporting ...")
    hp = dict(n_estimators=best["n_estimators"], learning_rate=best["learning_rate"],
              max_depth=best["max_depth"], num_leaves=best["num_leaves"])
    family = best["family"]
    if family == "clf":
        final_model = LGBMClassifier(**hp, **FIXED_COMMON)
        final_model.fit(X, y_sign, sample_weight=weights)
    else:
        final_model = LGBMRegressor(**hp, **FIXED_COMMON, **REG_EXTRA)
        final_model.fit(X, y_train_space, sample_weight=weights)

    ticker = tickers[0]
    vol_norm = VOL_NORM_TARGET
    n_bars = INPUT_BARS

    def predict(nonce: int | None = None) -> float:
        """Return the predicted 1h BTC/USD **log-return** (not a price)."""
        live_row = workflow.get_live_features(ticker=ticker)
        if live_row is None or len(live_row) == 0:
            raise ValueError("could not fetch live features")
        live_row = live_row.reset_index()
        if "open_time" not in live_row.columns:   # fall back to "now" for time feats
            live_row["open_time"] = pd.Timestamp.now(tz="UTC")
        Cl, Hl, Ll, Vl, wl = matrices_from_workflow_df(live_row, n_bars)
        X_live, sigma_live = compact_features(Cl, Hl, Ll, Vl, wl)
        if family == "clf":
            p_up = float(final_model.predict_proba(X_live[feature_cols])[0, 1])
            log_ret = lam * (2.0 * p_up - 1.0) * float(sigma_live[0])
        else:
            raw = float(final_model.predict(X_live[feature_cols])[0])
            log_ret = lam * raw * (float(sigma_live[0]) if vol_norm else 1.0)
        if abs(log_ret) > 0.2:
            print(f"warning: implausible 1h log-return {log_ret:+.4f}")
        print(f"1h BTC log-return prediction: {log_ret:+.6f}")
        return float(log_ret)

    print("  smoke-testing predict() against live data ...")
    test_val = predict()
    if not np.isfinite(test_val):
        sys.exit("predict() returned a non-finite value; not exporting")

    with open(OUT_PKL, "wb") as f:
        cloudpickle.dump(predict, f)
    print(f"\nSaved {OUT_PKL}. Deploy with your registered Forge wallet:")
    print("  TOPIC_ID=<id from scripts/find_topic_id.py> python scripts/deploy_my_worker.py")


if __name__ == "__main__":
    main()
