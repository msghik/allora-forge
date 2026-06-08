# Allora Forge — BTC/USD log-return worker

A BTC/USD log-return prediction worker for the **Allora Model Forge**. It comes
in two layers:

1. **Self-updating MLOps system** (`forge/` + Docker) — the production worker for
   the **1-hour BTC/USD log-return competition** (5-minute cadence). Retrains
   **every day** on fresh 5-minute data, scores itself with the competition's own
   metrics (ZPTAE + the whitelist bundle), variance-calibrates and gates each new
   model against the current one (auto-rollback), versions everything, and serves
   inferences to an Allora worker node. See **[docs/MLOPS.md](docs/MLOPS.md)**.
2. **One-shot pipeline** (`phase3_train_export.py`) — a standalone notebook-style
   script that trains/validates/exports a single `predict.pkl` for the 24h topic.
   Kept as a reference example.

## Production: self-updating competition worker (full stack)

```bash
cp .env.example .env            # asset / exchange / calibration knobs
docker compose up -d --build    # trainer (daily retrain) + inference (:8000)
curl localhost:8000/health
curl localhost:8000/inference/BTC
```
Models on 5-minute candles with a 1-hour (12-bar) horizon; trains on first boot,
then daily. The promoted artifact lands at `models/current/predict.pkl` (upload to
the Forge) and the inference server is ready for
[`allora-offchain-node`](https://github.com/allora-network/allora-offchain-node)
(see `allora/config.example.json`). Local equivalents: `make train`, `make serve`,
`make test`.

Because the whitelist is **directional-accuracy heavy**, the worker trains
**sign-aware** candidates (a directional LightGBM classifier + regressor×classifier
blends, recency-weighted with a purged split) and feeds them up to **66 scale-free
features**: single-asset technicals, **cross-asset ETH↔BTC** lead-lag, **real order
flow** (taker-buy volume / CVD from raw klines), **futures positioning** (funding
rate + open interest), and **on-chain** (stablecoin supply, CEX net-flow, DEX
volume, activity via Dune — see [docs/ONCHAIN.md](docs/ONCHAIN.md)). The model
therefore takes `predict(df, ref_df, fut_df, onchain_df)` and the **server fetches
all of them** (controlled by `cross_symbol` / `futures_symbol` / Dune config);
missing alt-data degrades to neutral and `predict.pkl` stays self-contained. Order
flow + futures need a binance-family / futures-enabled exchange (set
`ALLORA_EXCHANGE=binance` / `ALLORA_FUTURES_EXCHANGE` outside the US). For the ETH
topic, run a second stack with `ALLORA_SYMBOL=ETH/USDT`, `ALLORA_CROSS_SYMBOL=BTC/USDT`,
`ALLORA_FUTURES_SYMBOL=ETH/USDT:USDT`.

## Reference: one-shot 24h pipeline

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
