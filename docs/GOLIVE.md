# Going live — deploy the worker & start the whitelist baseline

The model is **ceilinged at DA ≈ 0.53** on this data class (see [MLOPS.md](MLOPS.md)),
so whitelisting now hinges on what the rules actually reward: a **clear majority of
the criteria integrated over a long baseline**, **liveness**, **statistical
significance**, and **Hammers** (network contribution). All of those accrue from
running a stable worker live for a long time. This is that checklist.

## 0. Prereqs
- The stack builds and a model is promoted (`docker compose up -d --build`;
  `curl localhost:8000/health` shows `"has_model": true`).
- `.env` set for live data + the locked config (order flow on, futures off):
  ```
  ALLORA_EXCHANGE=binanceus          # or binance outside the US
  ALLORA_USE_ORDERFLOW=true
  ALLORA_FUTURES_SYMBOL=             # empty: ablation showed futures hurt 1h DA
  ALLORA_TRAIN_WINDOW_DAYS=365
  ```

## 1. Produce & verify the artifact
```bash
docker compose exec trainer python -m forge.pipeline --once   # trains, gates, promotes
curl -s localhost:8000/inference/BTC                          # must return a bare number
curl -s localhost:8000/metadata | python -m json.tool        # model, n_features, metrics
```
The promoted `predict.pkl` is at `models/current/predict.pkl` (also copied to the
repo root on each promotion).

## 2. Upload to the Forge
Upload `models/current/predict.pkl` on the competition's Forge page for the **1h
BTC/USD log-return** topic. It reloads with plain `pickle.load` and is called as
`predict(df, ref_df)` (the Forge supplies BTC + the ETH reference); it needs only
`numpy`, `pandas`, `scikit-learn`, `lightgbm` — no `forge`, no `pandas_ta`.

## 3. Inference server (already running)
The `inference` service serves `GET /inference/{token}` on `:8000` and **hot-reloads**
whenever the trainer promotes a new model. Keep the `trainer` service in `--loop`
so it retrains daily and reconciles live predictions.

## 4. allora-offchain-node (live submissions)
The live worker is [`allora-offchain-node`](https://github.com/allora-network/allora-offchain-node);
it polls our endpoint and submits to the topic.
1. Create/fund a testnet worker wallet (see that repo's README for `allorad` key
   creation + the faucet). You need the **key name** and **restore mnemonic**.
2. Copy `allora/config.example.json` to the offchain node as its `config.json` and fill in:
   - `wallet.addressKeyName`, `wallet.addressRestoreMnemonic`
   - `worker[0].topicId` → the **1h BTC topic id** from the competition page
   - `parameters.InferenceEndpoint` → reach this repo's server:
     - same Docker network: `http://inference:8000/inference/{Token}`
     - otherwise: `http://<this-host-ip>:8000/inference/{Token}`
3. Start it (join this stack's network so `inference` resolves, e.g.):
   ```bash
   docker run --rm --network allora-forge_default \
     -v $PWD/config.json:/app/config.json alloranetwork/allora-offchain-node:latest
   ```
   (Match the image/flags to the offchain-node repo's current compose.)

## 5. Verify it's live
- Offchain-node logs show it fetching `/inference/BTC` every loop and **submitting**
  to the topic (a tx hash per round).
- `curl -s localhost:8000/scoreboard` → live DA/correlation on **matured** predictions
  (populates ~1h after the first submissions, once `monitor.reconcile` runs in the loop).
- Until the node is wired, a stand-in keeps the baseline accruing:
  ```bash
  */5 * * * * curl -s localhost:8000/inference/BTC >> /var/log/forge_infer.log
  ```

## 6. Monitor the baseline (the whitelist, over time)
```bash
# live skill on realized 1h returns (the number the team's discretion sees):
curl -s localhost:8000/scoreboard
docker compose exec trainer python -c "from forge.config import Config; from forge import monitor; print(monitor.reconcile(Config.from_env()))"
# per-retrain backtest history:
docker compose exec trainer tail -f models/metrics.jsonl
```
At true DA ≈ 0.53 the **significance** criteria (Pearson r/p, DA p-value, log-aspect,
and DA-CI-lower-bound as `n` grows) strengthen with baseline length, even though the
headline `DA > 0.55` stays just out of reach. Longer + uninterrupted = stronger case.

## 7. Liveness checklist (don't lose points to ops)
- [ ] `trainer` runs `--loop` (daily retrain + reconcile) and restarts on failure
      (`restart: unless-stopped` in compose).
- [ ] `inference` is `restart: unless-stopped`; `/health` is green.
- [ ] The worker **never goes dark**: `/inference` falls back to the last good value
      on a transient error (built in) — verify you always get a number.
- [ ] Wallet stays funded (testnet faucet) so submissions don't fail.
- [ ] Set `ALLORA_ALERT_WEBHOOK` to get pinged on training failures / regressions.
- [ ] Watch for gaps in `models/predictions.jsonl` (missed submissions hurt liveness).
