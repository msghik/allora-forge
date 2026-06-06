# Self-updating Topic 69 worker — MLOps design

A daily-retraining system that keeps the BTC 24h model current with market
behavior, never promotes a worse model, versions everything, and serves
inferences to an Allora worker node.

## Why
A model trained once goes stale as the market's regime, volatility, and "culture"
shift. This system retrains every day on fresh candles, **gates** each new model
against the one in production (promote only if as-good-or-better, else keep the
incumbent — automatic rollback safety), and serves the current model 24/7.

## Architecture

```
                 ┌─────────────── daily schedule (forge.pipeline --loop) ───────┐
                 ▼                                                               │
 ccxt ──> [1 ingest] ──> data/ohlcv.csv  (append + dedup + gap-check, data.py)  │
                 │                                                               │
                 ▼                                                               │
        [2 features]  add_features (features.py) ── single source of truth      │
                 │                                                               │
                 ▼                                                               │
   [3 train]  build 24h target → chrono split → Ridge + LightGBM (train.py)     │
                 │                                                               │
                 ▼                                                               │
   [4 evaluate]  Pearson r · directional acc · MAE/RMSE (evaluate.py)           │
                 │                                                               │
                 ▼                                                               │
   [5 gate]  edge vs baseline AND ≥ current production r (registry.gate)         │
                 │  pass → promote            fail → keep prod (rollback-safe)   │
                 ▼                                                               │
   [6 registry]  models/<version>/{predict.pkl, metadata.json} + current/  ─────┘
                 │
        ┌────────┴───────────────┐
        ▼                        ▼
 [7a Forge artifact]      [7b inference server]  FastAPI (server.py)
   models/current/          GET /inference/{token} → number
   predict.pkl  ──upload    loads models/current/predict.pkl, hot-reloads
   to Forge (Topic 69)      ← allora-offchain-node (allora/config.example.json)
                 │
                 ▼
   [8 monitor]  metrics.jsonl · log live predictions · reconcile vs realized
                24h return → live directional-acc / correlation · alerts (monitor.py)
```

## Components (`forge/`)
| Module | Responsibility |
|--------|----------------|
| `config.py` | One typed `Config` (symbol, horizon=24, rolling window, model grid, gate thresholds, paths); env overrides. |
| `data.py` | Incremental OHLCV ingest, dedup/gap checks, CSV store, recent-candle fetch (with store fallback). |
| `features.py` | Canonical `add_features` (pure pandas/numpy) + `FEATURE_COLS`. |
| `train.py` | 24h target, chronological split, regularized Ridge (TS-CV alpha) + LightGBM (early stopping). |
| `evaluate.py` | Pearson r, directional accuracy, MAE/RMSE, baseline DA. |
| `registry.py` | Versioned model store, **promotion gate**, `current` pointer, rollback, metrics log. |
| `export.py` | `predict()` closure + cloudpickle (features pickled **by value** → portable predict.pkl). |
| `pipeline.py` | Orchestrates a cycle (`--once`) and the daily loop (`--loop`). |
| `server.py` | FastAPI worker: `/inference/{token}`, `/health`, `/metadata`; hot-reloads on promotion. |
| `monitor.py` | Live-prediction logging, reconciliation vs realized returns, alerts. |

## Key design decisions (defaults, all in `config.py`)
- **Rolling 365-day window** — adapts to regime shifts; switch to expanding by raising `train_window_days`.
- **Gated promotion** — a candidate must show a real edge (Pearson r > 0 **and** directional accuracy ≥ 0.50) **and** match/beat the current model's r within `gate_tolerance`. Otherwise the incumbent stays — a bad day can't degrade you.
- **Immutable versioned registry** — every model keeps `metadata.json` (metrics, data range, params, git SHA) for audit and one-command rollback.
- **Live reconciliation** — every served prediction is logged and later compared to the realized 24h return, so you track *real* out-of-sample skill, not just backtests.
- **Portable artifact** — `predict.pkl` carries its own feature code (cloudpickle by-value), so it runs in the Forge scoring env and in our server without the `forge` package.

## Run it (full stack)
```bash
cp .env.example .env          # optional: set symbol / alert webhook
docker compose up -d --build  # starts: trainer (daily) + inference (port 8000)
```
- First boot trains immediately (cold-start fetch of ~window history), then retrains daily at `ALLORA_RETRAIN_HOUR_UTC`.
- Check it: `curl localhost:8000/health` and `curl localhost:8000/inference/BTC`.

### Wire to Allora
- **Topic 69 / Forge:** upload `models/current/predict.pkl` (also copied to repo-root `predict.pkl` on each promotion).
- **Live worker node:** run [`allora-offchain-node`](https://github.com/allora-network/allora-offchain-node) with `worker[].parameters.InferenceEndpoint = http://inference:8000/inference/{Token}` for `topicId: 69` — see `allora/config.example.json` (adapt to your node version).

## Operate
- **Logs:** `docker compose logs -f trainer` / `inference`.
- **History:** `models/metrics.jsonl` (one row per cycle: winner, metrics, promoted, reason).
- **Live skill:** `models/reconciliations.jsonl` → `monitor.live_scoreboard`.
- **Rollback:** `python -c "from forge.config import Config; from forge import registry; registry.rollback(Config.from_env())"`.
- **Alerts:** set `ALLORA_ALERT_WEBHOOK` to be notified on training failure or a rejected (regressing) candidate.

## Local (no Docker)
```bash
make dev      # install runtime + test deps
make train    # one cycle  -> models/current/predict.pkl
make serve    # inference server on :8000
make test     # offline end-to-end tests
```
