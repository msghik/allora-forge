# Deploying on Topic 72 with the Allora Forge Builder Kit

The [**Allora Forge Builder Kit**](https://github.com/allora-network/allora-forge-builder-kit)
is the current, officially-recommended way to run a Forge worker — it handles data,
feature engineering, training, evaluation, wallet/faucet, the worker process, and a
dashboard. For **topic 72 (1h BTC/USD log-return, 5-min cadence)** it is the fastest
path. This repo's custom stack (`forge/`) remains a valid alternative and a useful
research bench (`python -m forge.research`); the insights below carry straight over.

> Verify specifics against your freshly-cloned kit (its API evolves). Lines marked
> **[verify]** are the few that matter for topic 72.

## 1. Install
```bash
git clone https://github.com/allora-network/allora-forge-builder-kit.git
cd allora-forge-builder-kit
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install . && python -m pip install -r requirements.txt
# data: a free key from developer.allora.network, OR use data_source="binance"
echo "UP-..." > .allora_api_key && export ALLORA_API_KEY=$(cat .allora_api_key)
```

## 2. Train a topic-72 model
Start from `notebooks/example_topic_77_bitcoin_5min_walkthrough.py` (the 5-minute
example is the closest structure) and change it for topic 72:

- **[verify] Horizon = 1h on 5m bars** — set the workflow to a 12-bar-ahead target:
  ```python
  TICKERS = ["btcusd"]
  INTERVAL = "5m"
  NUMBER_OF_INPUT_BARS = 48
  TARGET_BARS = 12          # 12 * 5m = 1 hour ahead   (topic_77 used 1)
  TOPIC_ID = 72
  workflow = AlloraMLWorkflow(tickers=TICKERS, number_of_input_bars=NUMBER_OF_INPUT_BARS,
                              target_bars=TARGET_BARS, interval=INTERVAL,
                              data_source="allora", api_key=api_key)
  ```
- **[verify] Return the LOG-RETURN, not a price.** Topic 72's target *is* the log
  return, so the `predict(nonce)` closure must return it directly — delete the
  topic_77 price conversion:
  ```python
  predicted_log_return = final_model.predict(live_features[feature_cols].values.reshape(1, -1))[0]
  return float(predicted_log_return * SCALE)     # NOT current_price * exp(...)
  ```
- **Model (carry over our findings):** a *regularized* LightGBM generalizes best at
  this signal level —
  ```python
  LGBMRegressor(n_estimators=600, learning_rate=0.02, max_depth=3,
                num_leaves=31, min_child_samples=400, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0,
                random_state=42, verbose=-1)
  ```
  Keep the example's `TimeSeriesSplit(gap=TARGET_BARS)` walk-forward (it's the purged
  CV we rely on). Expect DA ≈ **0.53** — that clears the kit's 0.52 bar but don't
  chase 0.55; it's the data ceiling.

## 3. Calibrate for the kit's grade (the opposite of mainnet)
The kit's evaluator has **no log-aspect constraint**, so **shrink** predictions to
earn the error-improvement metrics (DA/Pearson are scale-invariant, so they don't
move). Grid-search a `SCALE` < 1 on the evaluator:
```python
from allora_forge_builder_kit import PerformanceEvaluator
evaluator = PerformanceEvaluator(workflow)
for s in (1.0, 0.7, 0.5, 0.35, 0.25):
    m = evaluator.evaluate(y_true=df_all.loc[mask,'target'],
                           y_pred=s * df_all.loc[mask,'pred'])
    print(s, m)            # pick the s with the best WRMSE/CZAR improvement + grade
```
Bake the winning `s` into the closure as `SCALE`. (On our custom mainnet stack we
do the reverse — `ALLORA_CALIBRATION_RATIO` keeps std≈1 for the log-aspect rule.)

## 4. Save the artifact
```python
import cloudpickle
with open("predict.pkl", "wb") as f:
    cloudpickle.dump(predict, f)
```

## 5. Deploy to topic 72
```bash
TOPIC_ID=72 python deploy_worker.py
python -m allora_forge_builder_kit.web_dashboard   # monitor on :8787
```

### Use your REGISTERED wallet (`allo1uv65ppemwjlz0grevxz3u7hxjjg6jpqh7cz5lt`)
`deploy_worker.py` **auto-creates a new wallet** in `worker_keys/` by default. The
competition scores the address you registered in the Forge, so do **one** of:
- **Import your wallet first, then deploy** (verified API): register your existing
  key, then re-run deploy — the kit picks the imported identity because it isn't yet
  used for the topic:
  ```python
  from allora_forge_builder_kit import WorkerManager
  wm = WorkerManager()
  wm.ensure_identity(alias="forge",
                     address="allo1uv65ppemwjlz0grevxz3u7hxjjg6jpqh7cz5lt",
                     mnemonic="<YOUR SEED PHRASE>")   # never commit/share this
  ```
  then `TOPIC_ID=72 python deploy_worker.py` and confirm the printed address is yours.
- **Or** let it create a wallet and **register THAT address** on the Forge instead.

Either way, confirm the worker's submitting address matches what the Forge shows as
"Participating", and that the wallet is faucet-funded. **Never commit/share the mnemonic.**

## 6. Verify & keep alive
- Dashboard (`:8787`) and the topic-72 leaderboard show your worker + Hammers.
- Hammers reward **accuracy + liveness**, so keep the process up and the wallet funded.
- Submissions open/close on topic windows; the kit handles the timing.

## What transfers from this repo's research
- **DA ceiling ≈ 0.53** on 1h BTC from public data — set expectations accordingly.
- **Regularized LightGBM + purged walk-forward** (validated by the kit's own example).
- **Order-flow ≈ the only feature block that helped, marginally**; cross/futures/on-chain
  were within noise — don't over-engineer features.
- Use `python -m forge.research --folds 6` / `--tune` here to vet any feature/param
  idea before porting it into the kit model.
