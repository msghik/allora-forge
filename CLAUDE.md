# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A self-updating worker for the **Allora Model Forge** competition: **1-hour-ahead
BTC/USD log-return** prediction, polled every 5 minutes, scored with ZPTAE + a
whitelist metric bundle. Two layers:

- `forge/` + `docker-compose.yml` — the production MLOps stack (daily retrain →
  gate → version → serve). This is where almost all work happens.
- `phase3_train_export.py` — a standalone one-shot reference script (24h topic).
  Kept as an example; not part of the live system.

## Commands

```bash
make dev          # pip install runtime + test deps (requirements*.txt)
make test         # pytest -q  (the suite is fully offline / synthetic)
pytest tests/test_system.py::test_run_once_sign_aware_portable -q   # a single test
make train        # one retrain/gate/export cycle  (python -m forge.pipeline --once)
make serve        # inference server on :8000       (uvicorn forge.server:app)
make docker-up    # build + run trainer(--loop) + inference, sharing ./data ./models

# Evidence-based feature/hyperparameter decisions (uses local stores, no refetch):
python -m forge.research --folds 6          # walk-forward DA per feature block
python -m forge.research --tune --folds 6   # DA-targeted LightGBM hyperparameter search
```
All behavior is configured by `ALLORA_*` env vars read in `Config.from_env()`
(see `.env.example`); `docker-compose.yml`'s `x-env` anchor supplies defaults.

## The pipeline cycle (`forge/pipeline.py :: run_once`)

ingest → features → 1h target → **purged** chrono split → train candidates →
per-candidate **calibration-grid search** + **whitelist-aware selection** →
promotion **gate** → version + export → promote (+ copy `predict.pkl` to repo root).
`run_loop` does this daily and reconciles matured live predictions. Fetchers are
injectable (`fetcher`, `futures_fetcher`, `onchain_fetcher`) so everything runs
offline in tests.

## Architecture invariants (the things that span files and break easily)

**1. The `predict.pkl` portability contract.** `forge/export.py` cloudpickles a
`predict(df, ref_df=None, fut_df=None, onchain_df=None)` closure, registering
`forge.features` and `forge.estimators` for **pickle-by-value**. It must reload
with plain `pickle.load` in an environment where `forge` is NOT importable (the
Forge scoring process) and return a `float`. Therefore: the closure must not
import `forge`; `features.py` and `estimators.py` must stay self-contained (pure
`numpy`/`pandas`/`sklearn`/`lightgbm` — **no `pandas_ta`**, no project imports
inside feature functions). After any change to features or the model, the artifact
must be re-exported (a retrain cycle does this).

**2. `features.py` is the single source of truth.** The exact same `build_features`
runs in training and inference. The active feature list is **config-determined**,
not data-determined: `active_feature_cols(cross_prefix, use_orderflow, use_futures,
use_onchain)` must match what `build_features` produces. Optional blocks (order
flow / cross / futures / on-chain) **degrade to neutral** (0 / NaN-filled) when a
data source is missing, so the column set is stable across train/serve.

**3. Never blanket-`dropna` once the frame carries optional raw/alt columns.**
Order-flow raw columns or futures/on-chain series can be all-NaN (source
unavailable); a blanket `dropna()` then wipes every row ("insufficient data: 0
rows"). Each block subsets its `dropna`/`fillna` to its own columns; `build_target`
drops only on `feature_cols + target`. There is a regression test for this.

**4. Inference must never go dark.** `forge/server.py` hot-reloads `predict.pkl` by
mtime and, on any transient failure, serves the last good prediction rather than
erroring — liveness is a (discretionary) whitelist factor. The server reads the
current model's metadata to decide which inputs to fetch (cross/futures/on-chain).

## Competition scoring & the calibration tension (`forge/evaluate.py`)

The whitelist (`WHITELIST` dict) wants, over a long baseline: DA > 0.55, DA CI-low
> 0.52, DA p < 0.05, Pearson r > 0.05, Pearson p < 0.05, WRMSE-vs-zero > 10%,
WZPTAE-vs-zero > 20%, |log-aspect| < 0.5. Two non-obvious consequences baked into
the design:

- **No shrink-to-zero.** A low-variance model minimizes raw error but fails
  log-aspect, so each cycle searches `calibration_ratio_grid` (a `std(pred)/std(true)`
  scale) for the ratio passing the most criteria. Scaling is DA/Pearson-invariant.
- **WRMSE/WZPTAE-improvement and log-aspect are mutually exclusive at realistic r**
  (~0.1): improvement needs shrinkage, log-aspect forbids it. So the achievable
  target is the **DA cluster + Pearson + log-aspect** (a "clear majority"), not all
  8. Metrics are evaluated on **non-overlapping** 1h windows; the gate
  (`registry.py`) promotes on `whitelist_passed` (Pearson-r floor, ZPTAE tiebreak,
  no-regression vs current).

## Working style for this repo

DA on 1h crypto is ~0.53 and a single validation window has a ±0.015 standard
error, so **single-run DA differences are usually noise**. Before adding features
or changing models, validate with `forge.research` (walk-forward, multiple folds);
keep only what beats its noise band. Models train Ridge + LightGBM regressors **and**
a directional classifier + blends (`forge/estimators.py`), selected on directional
accuracy. Detailed design is in `docs/MLOPS.md`; alt-data setup in `docs/ONCHAIN.md`;
deployment in `docs/GOLIVE.md`.
