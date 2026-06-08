# Self-updating competition worker — MLOps design

A daily-retraining system for the Forge **1-hour BTC/USD log-return** topic
(polled every 5 minutes). It keeps the model current with the market, never
promotes a worse model, versions everything, scores itself with the
competition's own metrics, and serves inferences to an Allora worker node.

## The competition (what we optimize for)
- **Target:** `log_return = ln(price_in_1h / price_now)`.
- **Cadence:** inference requested every 5 minutes → we model on **5-minute candles** with a **12-bar (=1h) horizon**, so every poll yields a fresh prediction.
- **Loss:** ZPTAE (z-transformed power-tanh absolute error; reference std over the last 100 non-overlapping 1h returns).
- **Whitelist bundle** (aim for a *clear majority*): `DA > 0.55`, `DA 95% CI low > 0.52`, `DA p < 0.05`, `Pearson r > 0.05`, `Pearson p < 0.05`, `WRMSE-vs-zero > 10%`, `WZPTAE-vs-zero > 20%`, `|log10(std_pred/std_true)| < 0.5`.

### Two non-obvious consequences (baked into the design)
1. **No shrink-to-zero.** A heavily-regularized model with small predictions minimizes raw error but **fails the log-aspect-ratio** criterion (`std_pred ≪ std_true`). We add a **variance-calibration** step so `std(pred) ≈ std(true)`.
2. **A real tension** exists: WRMSE/WZPTAE-vs-zero reward shrinkage; log-aspect forbids it. That's why "a clear majority" — not all — is the bar. Each cycle **searches a calibration grid** (`calibration_ratio_grid`) and keeps the ratio that passes the **most** whitelist criteria, so the trade-off is resolved automatically against the scoreboard. Set `ALLORA_CALIBRATION_RATIO` to pin one ratio and disable the search.

## Architecture

```
                 ┌────────── daily schedule (forge.pipeline --loop) ──────────┐
                 ▼                                                             │
 ccxt ─> [1 ingest 5m]─> data/ohlcv_*.csv (append+dedup+gap-check, data.py)   │
                 │                                                             │
                 ▼                                                             │
        [2 features]  add_features (features.py) ── single source of truth    │
                 │                                                             │
                 ▼                                                             │
   [3 train]  build 1h target (12 bars) → chrono split → Ridge + LightGBM     │
                 │                                                             │
                 ▼                                                             │
   [4 evaluate]  competition metrics on NON-OVERLAPPING 1h windows            │
                 │   (DA+CI+p, Pearson+p, log-aspect, WRMSE/ZPTAE-vs-zero)    │
                 ▼                                                             │
   [5 calibrate + gate]  std-match predictions; promote on zptae_impr,        │
                 │        Pearson-r floor, no-regression vs current           │
                 ▼                                                             │
   [6 registry]  models/<version>/{predict.pkl, metadata.json} + current/ ────┘
                 │
        ┌────────┴───────────────┐
        ▼                        ▼
 [7a Forge artifact]      [7b inference server]  FastAPI (server.py)
   models/current/          GET /inference/{token} → number
   predict.pkl  ──upload    loads current predict.pkl, hot-reloads
                            ← allora-offchain-node (allora/config.example.json)
                 │
                 ▼
   [8 monitor]  metrics.jsonl (whitelist_passed/cycle) · log live predictions
                · reconcile vs realized 1h return → live DA/correlation · alerts
```

## Components (`forge/`)
| Module | Responsibility |
|--------|----------------|
| `config.py` | Typed `Config`: `timeframe=5m`, `horizon_steps=12`, rolling window, calibration grid, model params, paths; env overrides. |
| `data.py` | Per-symbol incremental 5m OHLCV ingest (BTC **and** the cross asset), dedup/gap checks, CSV store, recent-candle fetch (store fallback). |
| `features.py` | Canonical pure-pandas `add_features` (~37 single-asset features) **+ `add_cross_features`** (~11 cross-asset: ETH momentum/vol/lags, return spread, rolling correlation & beta) → 48 total via `build_features`. All scale-free. |
| `train.py` | 1h target, chronological split, regularized Ridge (TS-CV α) + LightGBM (early stopping), **variance-calibration** factor. |
| `evaluate.py` | Competition metrics (ZPTAE surrogate, WRMSE, DA + Wilson CI + binomial p, Pearson + p, log-aspect) on non-overlapping windows + whitelist pass/fail. |
| `registry.py` | Versioned store, **promotion gate** (whitelist-criteria-passed → ZPTAE tiebreak, Pearson floor, no-regression), rollback, metrics log. |
| `export.py` | `predict()` closure with calibration `scale` + cloudpickle (features by value → portable predict.pkl). |
| `pipeline.py` | Cycle (`--once`) / daily loop (`--loop`): per-candidate **calibration-ratio search** + **whitelist-aware** winner selection. |
| `server.py` | FastAPI worker: `/inference/{token}`, `/health`, `/metadata`; hot-reloads on promotion. |
| `monitor.py` | Live-prediction logging, reconciliation vs realized 1h returns, alerts. |

## Run it (full stack)
```bash
cp .env.example .env          # set asset / exchange / calibration ratio
docker compose up -d --build  # trainer (daily) + inference (:8000)
curl localhost:8000/health
curl localhost:8000/inference/BTC
```
First boot trains immediately (cold-start fetch of ~120 days of 5m candles), then retrains daily at `ALLORA_RETRAIN_HOUR_UTC`. Inferences use the latest promoted model on every 5-min poll.

> **Data source:** default `binanceus` is US-only. Outside the US set `ALLORA_EXCHANGE=binance` (or `coinbase`/`kraken`) and a matching `ALLORA_SYMBOL`.

### Wire to Allora
- **Forge upload:** `models/current/predict.pkl` (also copied to repo-root on each promotion).
- **Live worker node:** run [`allora-offchain-node`](https://github.com/allora-network/allora-offchain-node) with the BTC-1h topic id and `InferenceEndpoint = http://inference:8000/inference/{Token}` — see `allora/config.example.json`.
- **Add ETH:** copy the stack with `ALLORA_SYMBOL=ETH/USDT`, `ALLORA_TOPIC_TOKEN=ETH` (own data store + registry), and add the ETH topic to the offchain config.

## Operate
- **Logs:** `docker compose logs -f trainer` — each cycle prints both candidates' full metric line and the winner's whitelist count.
- **History:** `models/metrics.jsonl` — one row per cycle (winner, DA, r, log-aspect, zptae_impr, `whitelist_passed`, promoted?).
- **Live skill:** after 1h, predictions mature →
  `python -c "from forge.config import Config; from forge import monitor; print(monitor.live_scoreboard(Config.from_env()))"`.
- **Rollback:** `registry.rollback(Config.from_env())`.
- **Calibration:** searched automatically each cycle; set `ALLORA_CALIBRATION_RATIO` to pin one ratio and disable the search.
- **Alerts:** set `ALLORA_ALERT_WEBHOOK` for training failures / rejected (regressing) candidates.

## Improving DA (the real alpha)
The model uses ~48 scale-free features (single-asset multi-scale momentum/vol,
trend, oscillators, volume/order-flow, microstructure, regime, cyclical time
**plus cross-asset ETH↔BTC** lead-lag/spread/correlation/beta), a per-cycle
calibration search, and whitelist-aware selection. The cross-asset model needs
both assets at inference, so `predict(df, ref_df)` takes a reference DataFrame and
the **server fetches both** (controlled by `cross_symbol`); `predict.pkl` stays
self-contained. Clearing `DA > 0.55` on 1h crypto is still hard — remaining
levers, roughly by expected impact:
- **Alt-data**: order-book imbalance/depth, funding/open-interest, on-chain (gas,
  active addresses) — the data sources the rules suggest.
- **ZPTAE-direct training** (custom objective) and **model ensembling**.
- **More history** (`ALLORA_TRAIN_WINDOW_DAYS=365`) to tighten the DA CI.

> Note: WRMSE/WZPTAE-improvement-over-zero are effectively unreachable at realistic
> r (~0.1) — they'd need r≈0.4+. Target the **DA cluster + Pearson + log-aspect**
> (six achievable criteria); the framework scores all eight per cycle in
> `models/metrics.jsonl`.

## Local (no Docker)
```bash
make dev      # install runtime + test deps
make train    # one cycle  -> models/current/predict.pkl
make serve    # inference server on :8000
make test     # offline end-to-end tests
```
