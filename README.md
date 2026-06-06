# Allora Forge — Topic 69 Worker (24h BTC/USD)

A BTC/USD price-prediction worker for **Allora Topic 69** (open 1-day forecast on
1-hour candles). It comes in two layers:

1. **One-shot pipeline** (`phase3_train_export.py`) — train/validate/export a
   single `predict.pkl`. Great for the notebook workflow and a first Forge entry.
2. **Self-updating MLOps system** (`forge/` + Docker) — retrains **every day** on
   fresh data, gates each new model against the current one (auto-rollback),
   versions everything, and serves inferences to an Allora worker node. See
   **[docs/MLOPS.md](docs/MLOPS.md)**.

## Production: self-updating worker (full stack)

```bash
cp .env.example .env            # optional config
docker compose up -d --build    # trainer (daily retrain) + inference (:8000)
curl localhost:8000/health
curl localhost:8000/inference/BTC
```
Trains on first boot, then daily. The promoted artifact lands at
`models/current/predict.pkl` (upload to the Forge) and the inference server is
ready for [`allora-offchain-node`](https://github.com/allora-network/allora-offchain-node)
(see `allora/config.example.json`). Local equivalents: `make train`, `make serve`,
`make test`.

## Quick start (one-shot pipeline)

```bash
pip install -r requirements.txt
python phase3_train_export.py
```

This fetches ~180 days of 1h BTC OHLCV (or loads a cached `btc_1h.csv`),
engineers the canonical feature set, trains **Ridge** and **LightGBM**, prints
out-of-sample metrics, retrains the validation winner on all data, and writes
`predict.pkl`.

## Pipeline (`phase3_train_export.py`)

| Step | What | Function |
|------|------|----------|
| 1 | 24h forward log-return target `ln(Close₍ₜ₊₂₄₎ / Closeₜ)`, drop boundary NaNs | `build_target` |
| 2 | Chronological 80/20 split (no shuffle → no lookahead) | `chrono_split` |
| 3 | Regularized `StandardScaler+Ridge` (alpha via `TimeSeriesSplit`) and `LGBMRegressor` (early stopping on an inner-train tail) | `train_ridge`, `train_lgbm` |
| 4 | Pearson *r* + Directional Accuracy (+ MAE/RMSE, up-move baseline) | `evaluate` |
| 5 | Retrain the winner on all data, wrap in `predict`, `cloudpickle` → `predict.pkl` | `make_predict`, `save_predict` |

The feature engineering (`add_features`) is the **single source of truth** used
by both training and inference — the canonical 11 features:

```
log_return, rsi, bb_width, bb_pct, volatility_24h,
return_lag_1, return_lag_2, return_lag_3, return_lag_6, hour_sin, hour_cos
```

RSI-14 (Wilder) and Bollinger 20/2 (bandwidth + %B) are computed in **pure
pandas/numpy** rather than `pandas_ta`, so the exported artifact carries no
fragile technical-analysis dependency. The feature definitions are unchanged;
training and inference use the exact same code.

## The `predict` contract

```python
import pickle
predict = pickle.load(open("predict.pkl", "rb"))
y_hat = predict(raw_ohlcv_df)   # -> float: predicted 24h log return
```

- **Input:** a raw OHLCV DataFrame (`open, high, low, close, volume`). A
  `DatetimeIndex` is expected; if absent, `predict` will coerce one from a
  `date` or `timestamp` column.
- **Output:** a single `float`, the model's 1-day (24h) log-return prediction
  for the most recent candle.

### Why `cloudpickle` (not plain `pickle.dump`)
Standard `pickle.dump(predict)` stores a function *by reference*
(`__main__.predict`) — the closed-over model and the `add_features` code do
**not** travel with it, so it fails to reload in the Forge's fresh scoring
process. `cloudpickle.dump` serializes `predict` + the model + `add_features`
*by value* into one self-contained file that reloads anywhere via plain
`pickle.load`. (Run the builder as a **script**, not an import, so `add_features`
lives in `__main__` and is captured by value.)

## Caveats

- **Inference deps are minimal.** `predict` calls `add_features` (pure
  pandas/numpy) and the trained model, so the scoring environment only needs
  `numpy`, `pandas`, and the winning model's library (`scikit-learn` for Ridge,
  or `lightgbm`). No `pandas_ta`.
- **Lookback.** `predict` needs ≥ ~50 1h candles so the latest row survives
  `add_features`' `dropna` (Bollinger=20, 24h vol=24, RSI=14, lags≤6).
- **Network.** `fetch_data` hits Binance US. In a restricted environment, drop a
  `btc_1h.csv` next to the script (or set `ALLORA_DATA_CSV`) and it loads that
  instead.
- **Optimistic metrics.** Overlapping 24h targets induce residual
  autocorrelation, so validation *r* / directional accuracy are mildly
  optimistic vs. truly independent samples. A small positive *r* and a
  directional accuracy above the up-move baseline indicate a real edge; retrain
  regularly on fresh data for live performance.

`predict.pkl` is git-ignored as a reproducible artifact — regenerate it with the
command above. Remove the entry in `.gitignore` if you want to commit it for
submission.
