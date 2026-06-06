"""
Allora Forge -- Phase 3: Model Training, Validation & Export for Topic 69
========================================================================

Topic 69 is the open 1-day (24h) BTC/USD price-prediction topic on 1-hour
candles. This script finishes the pipeline started in the notebook (Phase 1:
data fetch, Phase 2: ``add_features``) and produces the artifact the Allora
Model Forge expects: a single, self-contained ``predict.pkl``.

Run end-to-end::

    pip install -r requirements.txt
    python phase3_train_export.py

Pipeline:
  1. Fetch (or load cached) 1h BTC OHLCV data        (Phase 1, notebook Cell 2)
  2. Engineer the canonical 11-feature set            (Phase 2, notebook Cell 3)
  3. Build the 24h forward log-return target          (Step 1)
  4. Chronological 80/20 split (no shuffle)           (Step 2)
  5. Train regularized Ridge + LightGBM               (Step 3)
  6. Backtest: Pearson r + Directional Accuracy       (Step 4)
  7. Retrain the validation winner on ALL data and
     export a self-contained ``predict()``            (Step 5)

Each top-level function maps to a notebook cell, so the code can also be pasted
back into the original Jupyter notebook.

IMPORTANT: run this as a *script* (``python phase3_train_export.py``), not as an
imported module. cloudpickle serializes functions defined in ``__main__`` *by
value*; that is what makes ``predict.pkl`` reload in a fresh Forge process.
"""

import os
import pickle
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical configuration
# ---------------------------------------------------------------------------
HORIZON = 24  # forecast horizon in hours -> Topic 69 is a 1-day forecast

# The exact 11-feature set produced by add_features (notebook Cell 3).
FEATURE_COLS = [
    "log_return",
    "rsi",
    "bb_width",
    "bb_pct",
    "volatility_24h",
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_6",
    "hour_sin",
    "hour_cos",
]

DATA_CSV = os.environ.get("ALLORA_DATA_CSV", "btc_1h.csv")
PREDICT_PKL = os.environ.get("ALLORA_PREDICT_PKL", "predict.pkl")


# ===========================================================================
# Phase 1 (Cell 2): Data fetching
# ===========================================================================
def fetch_data(symbol="BTC/USDT", timeframe="1h", days_back=180):
    """Fetch paginated historical OHLCV from Binance US into a DataFrame
    indexed by a tz-naive ``date`` DatetimeIndex."""
    import ccxt  # imported lazily so training/export works without it offline

    print(f"Fetching {symbol} {timeframe} data for the last {days_back} days...")
    exchange = ccxt.binanceus({"enableRateLimit": True})
    since = exchange.parse8601((datetime.utcnow() - timedelta(days=days_back)).isoformat())

    all_candles = []
    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=since)
        if not candles:
            break
        all_candles += candles
        since = candles[-1][0] + 1
        if candles[-1][0] >= exchange.milliseconds() - 60_000:
            break

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("date").drop(columns=["timestamp"])
    print(f"Success! Final dataset shape: {df.shape}")
    return df


def load_data(symbol="BTC/USDT", timeframe="1h", days_back=180, csv_path=DATA_CSV):
    """Load cached OHLCV from ``csv_path`` if present, otherwise fetch and
    cache it. Lets the script run in network-restricted environments."""
    if csv_path and os.path.exists(csv_path):
        print(f"Loading cached data from {csv_path} ...")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df.index.name = "date"
        return df
    df = fetch_data(symbol=symbol, timeframe=timeframe, days_back=days_back)
    try:
        df.to_csv(csv_path)
        print(f"Cached data to {csv_path}")
    except OSError as exc:
        print(f"(Could not cache data: {exc})")
    return df


# ===========================================================================
# Phase 2 (Cell 3): Feature engineering -- the SINGLE source of truth
# ===========================================================================
def add_features(df):
    """Engineer the canonical Topic 69 feature set from raw OHLCV candles.

    Same features and semantics as the notebook's Cell 3 ``add_features`` (RSI-14,
    Bollinger 20/2 bandwidth & %B, 24h vol, return lags, cyclical hour), but the
    two technical indicators are computed in **pure pandas/numpy** instead of via
    ``pandas_ta``. This removes a fragile, version-sensitive dependency so the
    cloudpickled ``predict.pkl`` is self-sufficient anywhere it is scored, and it
    keeps training and inference perfectly consistent. numpy/pandas are imported
    inside the function so the captured code carries its own dependencies.

    Requires a DatetimeIndex (the cyclical hour encoding uses ``index.hour``).
    Adds no forward-looking columns, so it is safe to call on live data.
    """
    import numpy as np
    import pandas as pd  # noqa: F401  (kept for self-sufficiency)

    data = df.copy()

    # 1. Log returns (additive, symmetric).
    data["log_return"] = np.log(data["close"] / data["close"].shift(1))

    # 2a. RSI (Wilder's smoothing, length 14).
    rsi_len = 14
    delta = data["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / rsi_len, min_periods=rsi_len, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_len, min_periods=rsi_len, adjust=False).mean()
    rs = avg_gain / avg_loss
    data["rsi"] = 100.0 - 100.0 / (1.0 + rs)  # avg_loss==0 -> rs=inf -> rsi=100

    # 2b. Bollinger Bands (length 20, std 2): bandwidth (BBB) and %B (BBP).
    bb_len, bb_std = 20, 2.0
    mid = data["close"].rolling(bb_len).mean()
    sd = data["close"].rolling(bb_len).std(ddof=0)
    upper = mid + bb_std * sd
    lower = mid - bb_std * sd
    data["bb_width"] = 100.0 * (upper - lower) / mid          # bandwidth %
    data["bb_pct"] = (data["close"] - lower) / (upper - lower)  # position within bands

    # 3. Rolling realized volatility (risk) -- 24h std of returns.
    data["volatility_24h"] = data["log_return"].rolling(window=24).std()

    # 4. Lagged returns (memory).
    for lag in (1, 2, 3, 6):
        data[f"return_lag_{lag}"] = data["log_return"].shift(lag)

    # 5. Cyclical hour-of-day encoding (the 24h clock).
    data["hour_sin"] = np.sin(2 * np.pi * data.index.hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data.index.hour / 24)

    data.dropna(inplace=True)
    return data


# ===========================================================================
# Phase 3, Step 1: Target construction (24h forward log return)
# ===========================================================================
def build_target(df_features, horizon=HORIZON):
    r"""target_t = ln(Close_{t+H} / Close_t).

    Implemented with ``close.shift(-H)`` to pull the future close back onto the
    current row, then drop the ``horizon`` trailing NaN boundary rows.
    """
    data = df_features.copy()
    data["target"] = np.log(data["close"].shift(-horizon) / data["close"])
    data = data.dropna(subset=["target"])
    X = data[FEATURE_COLS].copy()
    y = data["target"].copy()
    return X, y


# ===========================================================================
# Phase 3, Step 2: Chronological split (no shuffle -> no lookahead)
# ===========================================================================
def chrono_split(X, y, train_frac=0.8):
    cut = int(len(X) * train_frac)
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


# ===========================================================================
# Phase 3, Step 3: Model training (regularized Ridge + LightGBM)
# ===========================================================================
def train_ridge(X_tr, y_tr):
    """StandardScaler + Ridge; alpha chosen by TimeSeriesSplit CV on train
    only. Returns (fitted_pipeline, best_alpha)."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
    grid = {"ridge__alpha": [0.1, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0]}
    gs = GridSearchCV(
        pipe, grid, cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_mean_squared_error", n_jobs=-1,
    )
    gs.fit(X_tr, y_tr)
    best_alpha = gs.best_params_["ridge__alpha"]
    print(f"  Ridge: best alpha = {best_alpha}")
    return gs.best_estimator_, best_alpha


# Strong regularization for noisy crypto returns.
LGBM_PARAMS = dict(
    n_estimators=2000,        # upper bound; early stopping picks the real count
    learning_rate=0.03,
    num_leaves=31,
    max_depth=4,
    min_child_samples=80,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


def train_lgbm(X_tr, y_tr):
    """LightGBM with early stopping on an inner tail of the train set (the
    outer 20% validation stays fully out-of-sample). Returns (model_fit_on_
    full_train, best_iteration)."""
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    inner_cut = int(len(X_tr) * 0.85)
    X_in, X_es = X_tr.iloc[:inner_cut], X_tr.iloc[inner_cut:]
    y_in, y_es = y_tr.iloc[:inner_cut], y_tr.iloc[inner_cut:]

    probe = LGBMRegressor(**LGBM_PARAMS)
    probe.fit(
        X_in, y_in,
        eval_set=[(X_es, y_es)],
        eval_metric="l2",
        callbacks=[early_stopping(stopping_rounds=50, verbose=False), log_evaluation(0)],
    )
    best_iter = probe.best_iteration_ or LGBM_PARAMS["n_estimators"]
    print(f"  LightGBM: best_iteration = {best_iter}")

    # Refit on the FULL train set using the chosen number of trees.
    final_params = {**LGBM_PARAMS, "n_estimators": best_iter}
    model = LGBMRegressor(**final_params)
    model.fit(X_tr, y_tr)
    return model, best_iter


# ===========================================================================
# Phase 3, Step 4: Backtesting & statistical evaluation
# ===========================================================================
def evaluate(name, y_true, y_pred):
    """Pearson r, directional accuracy, MAE, RMSE on out-of-sample data."""
    from scipy.stats import pearsonr

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, p = pearsonr(y_true, y_pred)
    da = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    return {"model": name, "pearson_r": float(r), "p_value": float(p),
            "directional_acc": da, "mae": mae, "rmse": rmse}


def print_table(results, baseline_da):
    print(f"  {'model':<10}{'pearson_r':>12}{'dir_acc':>10}{'mae':>12}{'rmse':>12}")
    for d in results:
        print(f"  {d['model']:<10}{d['pearson_r']:>12.4f}"
              f"{d['directional_acc']:>10.2%}{d['mae']:>12.6f}{d['rmse']:>12.6f}")
    print(f"  (baseline directional accuracy, always-up: {baseline_da:.2%})")


# ===========================================================================
# Phase 3, Step 5: Self-contained predict() + cloudpickle export
# ===========================================================================
def make_predict(model, feature_cols):
    """Build the single callable the Forge expects. It runs the EXACT same
    add_features pipeline, takes the latest engineered row, and returns one
    float -- the model's 1-day (24h) log-return prediction."""

    def predict(df):
        import numpy as np  # noqa: F401  -- self-sufficient at inference time
        import pandas as pd

        d = df.copy()
        # Robustness: coerce a DatetimeIndex if a known time column is present
        # (add_features needs index.hour).
        if not isinstance(d.index, pd.DatetimeIndex):
            for col in ("date", "timestamp"):
                if col in d.columns:
                    unit = "ms" if col == "timestamp" else None
                    d = d.set_index(pd.to_datetime(d[col], unit=unit))
                    break

        feats = add_features(d)  # captured by value via cloudpickle
        if len(feats) == 0:
            raise ValueError(
                "Not enough candle history to compute features for a prediction "
                "(need ~50+ 1h candles)."
            )
        x_last = feats[feature_cols].iloc[[-1]]
        return float(model.predict(x_last)[0])

    return predict


def save_predict(predict_fn, path=PREDICT_PKL):
    import cloudpickle

    with open(path, "wb") as f:
        cloudpickle.dump(predict_fn, f)
    print(f"Saved self-contained predict() -> {path} ({os.path.getsize(path):,} bytes)")


def self_test(path, raw_df):
    """Reload with the STANDARD-LIBRARY pickle and run once, proving the
    artifact is portable to a fresh process."""
    with open(path, "rb") as f:
        reloaded = pickle.load(f)
    sample = reloaded(raw_df.tail(120).copy())
    assert isinstance(sample, float), "predict must return a float"
    print(f"[self-test] pickle.load OK -- sample 24h log-return prediction: {sample:.6f}")
    return sample


# ===========================================================================
# Orchestration
# ===========================================================================
def main():
    print("=" * 72)
    print("Allora Forge -- Phase 3: Train / Validate / Export  (Topic 69, 24h BTC)")
    print("=" * 72)

    # Phases 1-2
    raw = load_data(days_back=180)
    feats = add_features(raw)
    print(f"Engineered features: {feats.shape}  ({len(FEATURE_COLS)} feature cols)")

    # Step 1-2
    X, y = build_target(feats, HORIZON)
    print(f"After 24h target + dropna: X={X.shape}, y={y.shape}")
    X_tr, X_val, y_tr, y_val = chrono_split(X, y, 0.8)
    print(f"Chronological split -> train={len(X_tr)}  val={len(X_val)}")

    # Step 3
    print("\n[Step 3] Training models on the 80% train set...")
    ridge_model, ridge_alpha = train_ridge(X_tr, y_tr)
    lgbm_model, lgbm_best_iter = train_lgbm(X_tr, y_tr)

    # Step 4
    print("\n[Step 4] Out-of-sample validation metrics:")
    results = [
        evaluate("Ridge", y_val, ridge_model.predict(X_val)),
        evaluate("LightGBM", y_val, lgbm_model.predict(X_val)),
    ]
    baseline_da = float(np.mean(np.asarray(y_val, dtype=float) > 0))
    print_table(results, baseline_da)

    winner = max(results, key=lambda d: (d["pearson_r"], d["directional_acc"]))
    print(f"\nSelected winner (by Pearson r): {winner['model']}")

    # Step 5: retrain winner on ALL data, then export
    print(f"\n[Step 5] Retraining {winner['model']} on the full dataset and exporting...")
    if winner["model"] == "Ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        final_model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=ridge_alpha)),
        ])
        final_model.fit(X, y)
    else:
        from lightgbm import LGBMRegressor

        final_model = LGBMRegressor(**{**LGBM_PARAMS, "n_estimators": lgbm_best_iter})
        final_model.fit(X, y)

    predict = make_predict(final_model, FEATURE_COLS)
    save_predict(predict, PREDICT_PKL)
    self_test(PREDICT_PKL, raw)

    print("\nDone. Submit predict.pkl to the Allora Model Forge for Topic 69.")


if __name__ == "__main__":
    main()
