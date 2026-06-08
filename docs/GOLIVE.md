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

## 4. allora-offchain-node (live submissions to topic 72)
The live worker is [`allora-offchain-node`](https://github.com/allora-network/allora-offchain-node);
its `api-worker-reputer` adapter GETs our `/inference/{Token}` endpoint (a bare
number) and submits it to the topic. Run it as its own stack pointed at our server:

```bash
# fund the worker wallet first (testnet faucet) -- use YOUR address:
#   allo1uv65ppemwjlz0grevxz3u7hxjjg6jpqh7cz5lt
git clone https://github.com/allora-network/allora-offchain-node && cd allora-offchain-node
cp config.example.json config.json          # base on THEIR file so the schema matches your version
```
Edit `config.json` — set the wallet + the worker entry (values from this repo's
`allora/config.example.json`):
- `wallet.addressKeyName`: any label (e.g. `forge-worker`)
- `wallet.addressRestoreMnemonic`: the mnemonic that restores `allo1uv65…` (never commit it)
- `wallet.nodeRpc`: the current testnet RPC (e.g. `https://allora-rpc.testnet-1.testnet.allora.network/`)
- `wallet.submitTx`: **true** (false = dry-run, earns nothing)
- `worker[0].topicId`: **72**
- `worker[0].inferenceEntrypointName`: `api-worker-reputer`
- `worker[0].parameters.Token`: `BTC`
- `worker[0].parameters.InferenceEndpoint`: `http://HOST_IP:8000/inference/{Token}`
  (our inference server; `HOST_IP` = your server's IP, or `172.17.0.1` for the
  Docker bridge gateway on Linux — confirm the offchain container can curl it)

```bash
chmod +x init.config && ./init.config       # imports the wallet, exports the config for compose
docker compose up -d --build
docker compose logs -f                       # watch it fetch /inference/BTC and submit (tx hash/round)
```

> Our inference stack must be up first (`docker compose up -d` in this repo) so
> `HOST_IP:8000` answers. Keep both running.

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
