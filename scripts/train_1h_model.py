#!/usr/bin/env python3
"""Train + evaluate + export a predict.pkl for the 1h BTC/USD log-return topic.

Target (per the competition spec): log_return = ln(price_{t+1h} / price_t),
i.e. ``target_bars=1`` at ``interval="1h"``. The exported ``predict(nonce)``
returns the predicted **log-return as a float** — for log-return topics you
submit the log-return itself, NOT a price.

Requires the Allora Forge Builder Kit (pip install from its repo) and an
ALLORA_API_KEY (free at https://developer.allora.network). Run as a script
from the repo root — cloudpickle captures closures by value only when this
file executes as __main__:

    export ALLORA_API_KEY=UP-...
    python scripts/train_1h_model.py

Env overrides: DAYS_OF_HISTORY, INPUT_BARS, DATA_SOURCE (allora|binance).
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import cloudpickle
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit

from allora_forge_builder_kit import AlloraMLWorkflow, PerformanceEvaluator

# --- competition task: 1h BTC/USD log-return ---
INTERVAL = "1h"
TARGET_BARS = 1                # 1 bar ahead at 1h interval = 1 hour
INPUT_BARS = int(os.environ.get("INPUT_BARS", "48"))
DAYS_OF_HISTORY = int(os.environ.get("DAYS_OF_HISTORY", "500"))

# --- model search (small, conservative grid; expand once the loop works) ---
N_SPLITS = 3
N_ESTIMATORS_MAX = 500
N_ESTIMATORS_CHECKPOINTS = [100, 300, 500]
LEARNING_RATES = [0.01, 0.05]
MAX_DEPTHS = [3, 5]
NUM_LEAVES = [15, 31]

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


def power_tanh(x: np.ndarray, p: float = 1.5) -> np.ndarray:
    """Smooth bounded transform with a power-law-ish ramp — a stand-in for the
    competition's power-tanh. The official ZPTAE constant/tail differs slightly;
    this proxy is only used to compare *relative* loss vs the zero baseline."""
    return np.tanh(np.abs(x)) ** p


def zptae_proxy(y_true: np.ndarray, y_pred: np.ndarray, window: int = 100) -> float:
    """Mean power-tanh of |error| z-scored by the trailing std of the last
    ``window`` ground-truth log-returns (ref mean 0), like the topic's loss."""
    s = pd.Series(y_true)
    sigma = s.rolling(window, min_periods=window).std().shift(1).to_numpy()
    ok = np.isfinite(sigma) & (sigma > 0)
    if ok.sum() == 0:
        return float("nan")
    z = np.abs(y_pred[ok] - y_true[ok]) / sigma[ok]
    return float(np.mean(power_tanh(z)))


def main() -> None:
    print("=" * 78)
    print("1h BTC/USD log-return — train / evaluate / export")
    print("=" * 78)

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

    print("[2/5] Building feature/target dataframe ...")
    df_all = workflow.get_full_feature_target_dataframe(start_date=start_date).reset_index()

    # Engineered momentum features on top of the kit's normalized OHLCV bars.
    def engineer_returns(row):
        closes = np.array([row[f"feature_close_{i}"] for i in range(INPUT_BARS)])
        out = {}
        for name, lag in (("ret_1h", 1), ("ret_6h", 6), ("ret_12h", 12), ("ret_24h", 24)):
            out[name] = (
                np.log(closes[-1] + 1e-8) - np.log(closes[-1 - lag] + 1e-8)
                if INPUT_BARS > lag else 0.0
            )
        return pd.Series(out)

    base_feature_cols = [c for c in df_all.columns if c.startswith("feature_")]
    engineered = df_all.apply(engineer_returns, axis=1)
    df_all = pd.concat([df_all, engineered], axis=1)
    feature_cols = base_feature_cols + list(engineered.columns)
    df_all = df_all.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
    print(f"  {len(df_all):,} samples, {len(feature_cols)} features "
          f"({df_all['open_time'].min()} → {df_all['open_time'].max()})")

    print("[3/5] Walk-forward grid search ...")
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=TARGET_BARS)
    evaluator = PerformanceEvaluator()
    results = []
    n = 0
    for lr in LEARNING_RATES:
        for depth in MAX_DEPTHS:
            for leaves in NUM_LEAVES:
                fold_models = []
                for train_idx, test_idx in tscv.split(df_all):
                    m = LGBMRegressor(
                        n_estimators=N_ESTIMATORS_MAX, learning_rate=lr, max_depth=depth,
                        num_leaves=leaves, random_state=42, verbose=-1,
                    )
                    m.fit(df_all.iloc[train_idx][feature_cols], df_all.iloc[train_idx]["target"])
                    fold_models.append((m, test_idx))
                for n_est in N_ESTIMATORS_CHECKPOINTS:
                    n += 1
                    df_all["pred"] = np.nan
                    for m, test_idx in fold_models:
                        df_all.iloc[test_idx, df_all.columns.get_loc("pred")] = m.predict(
                            df_all.iloc[test_idx][feature_cols], num_iteration=n_est
                        )
                    mask = ~df_all["pred"].isna()
                    metrics = evaluator.evaluate(y_true=df_all.loc[mask, "target"],
                                                 y_pred=df_all.loc[mask, "pred"])
                    results.append({"n_estimators": n_est, "learning_rate": lr, "max_depth": depth,
                                    "num_leaves": leaves, "mask": mask.copy(),
                                    "pred": df_all["pred"].copy(), **metrics})
                    print(f"  [{n:2d}] n={n_est:3d} lr={lr:.2f} d={depth} l={leaves:2d} -> "
                          f"{metrics['num_passed']}/7 ({metrics['grade']})")

    results.sort(key=lambda r: (r["num_passed"], r["score"]), reverse=True)
    best = results[0]
    print(f"\n[4/5] Best config: n={best['n_estimators']} lr={best['learning_rate']} "
          f"d={best['max_depth']} l={best['num_leaves']}")
    best_report = evaluator.evaluate(y_true=df_all.loc[best["mask"], "target"],
                                     y_pred=best["pred"][best["mask"]])
    evaluator.print_report(best_report, detailed=False)

    # Competition-style loss proxy vs the zero-prediction baseline (out-of-sample preds).
    y_true = df_all.loc[best["mask"], "target"].to_numpy()
    y_pred = best["pred"][best["mask"]].to_numpy()
    zp_model = zptae_proxy(y_true, y_pred)
    zp_zero = zptae_proxy(y_true, np.zeros_like(y_pred))
    if np.isfinite(zp_model) and np.isfinite(zp_zero) and zp_zero > 0:
        print(f"  ZPTAE proxy: model={zp_model:.4f} zero-baseline={zp_zero:.4f} "
              f"improvement={(1 - zp_model / zp_zero):+.1%} (whitelist target > +20%)")

    print("[5/5] Training production model on all data and exporting ...")
    final_model = LGBMRegressor(
        n_estimators=best["n_estimators"], learning_rate=best["learning_rate"],
        max_depth=best["max_depth"], num_leaves=best["num_leaves"],
        random_state=42, verbose=-1,
    )
    final_model.fit(df_all[feature_cols], df_all["target"])

    ticker = tickers[0]

    def predict(nonce: int | None = None) -> float:
        """Return the predicted 1h BTC/USD **log-return** (not a price)."""
        live_row = workflow.get_live_features(ticker=ticker)
        if live_row is None or len(live_row) == 0:
            raise ValueError("could not fetch live features")
        live_returns = engineer_returns(live_row.iloc[0])
        feats = pd.concat([live_row[base_feature_cols].iloc[0], live_returns])
        log_ret = float(final_model.predict(feats[feature_cols].values.reshape(1, -1))[0])
        if abs(log_ret) > 0.2:
            print(f"warning: implausible 1h log-return {log_ret:+.4f}")
        print(f"1h BTC log-return prediction: {log_ret:+.6f}")
        return log_ret

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
