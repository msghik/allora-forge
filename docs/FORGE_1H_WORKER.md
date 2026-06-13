# Entering the Forge: 1h BTC/USD Log-Return — step by step

This guide takes you from zero to a worker submitting 1-hour BTC/USD
log-return predictions on the Allora **testnet**, credited to your
registered Forge wallet:

```
allo1uv65ppemwjlz0grevxz3u7hxjjg6jpqh7cz5lt
```

It uses the [Allora Forge Builder Kit](https://github.com/allora-network/allora-forge-builder-kit)
(the Allora MDK repo is private/unavailable; the builder kit is the supported
path) plus three helper scripts in this repo's `scripts/`.

> **The one rule that matters:** the worker must sign submissions with the
> *same wallet address you registered on forge.allora.network*. The builder
> kit auto-creates a fresh wallet by default — predictions from that wallet
> are invisible to your Forge account. `scripts/deploy_my_worker.py` exists
> precisely to prevent that.

---

## 0. What you are entering

| | |
|---|---|
| Task | predict `ln(price_{t+1h} / price_t)` for BTC/USD |
| Network | Allora **testnet** (Forge competitions run on testnet) |
| Scoring | modified ZPTAE: abs error z-scored by the rolling std of the last 100 ground-truth 1h log-returns, passed through a power-tanh |
| Submission | the chain opens a window every epoch; your worker is polled and replies with one float |
| Output | the **log-return itself** (e.g. `+0.0021`), *not* a price |

## 1. Set up the environment

```bash
git clone https://github.com/allora-network/allora-forge-builder-kit.git
cd allora-forge-builder-kit
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install . && python -m pip install -r requirements.txt
```

Get a free API key at [developer.allora.network](https://developer.allora.network):

```bash
echo "UP-..." > .allora_api_key
export ALLORA_API_KEY=$(cat .allora_api_key)
```

The scripts in this repo run from this repo's root inside that same venv:

```bash
cd /path/to/allora-forge
pip install cosmpy   # only extra needed; the kit brings the rest
```

## 2. Find the topic ID

The Forge page doesn't show the chain topic ID. Scan testnet for it:

```bash
python scripts/find_topic_id.py
# fallback if the pattern misses: python scripts/find_topic_id.py --all
```

Pick the **active** topic whose metadata names the 1-hour BTC/USD log-return
competition. Everything below uses `TOPIC_ID=<that id>`.

## 3. Fund your registered wallet

Request testnet ALLO for **your registered address** (gas for registering +
submitting):

- Faucet: <https://faucet.testnet.allora.run> → paste `allo1uv65ppemwjlz0grevxz3u7hxjjg6jpqh7cz5lt`
- The worker runtime also auto-drips from the faucet on startup if the balance is empty.

Check the balance:

```bash
curl https://allora-api.testnet.allora.network/cosmos/bank/v1beta1/balances/allo1uv65ppemwjlz0grevxz3u7hxjjg6jpqh7cz5lt
```

## 4. Train and export the model

```bash
export ALLORA_API_KEY=$(cat /path/to/allora-forge-builder-kit/.allora_api_key)
python scripts/train_1h_model.py
```

This backfills ~500 days of 1h candles, runs a walk-forward LightGBM grid
search, prints the kit's 7-metric report **plus a ZPTAE-proxy improvement vs
the zero baseline**, then smoke-tests and writes `predict.pkl` whose
`predict(nonce)` fetches live data and returns the 1h log-return.

How the printed metrics map to the Forge **mainnet-whitelist criteria**:

| Evaluator metric | Whitelist criterion |
|---|---|
| Directional Accuracy | DA > 0.55 |
| DA CI lower bound | DA 95% CI lower bound > 0.52 |
| DA p-value | < 0.05 |
| Pearson r / p-value | r > 0.05, p < 0.05 |
| WRMSE improvement | > 10% vs zero log-return |
| ZPTAE proxy improvement (script) | WZPTAE > 20% vs zero log-return |

Also watch the **log-aspect ratio** criterion: `log10(std(pred)/std(true))`
must stay within ±0.5 — heavily regularized models that predict near-zero
everywhere fail it. The trainer's shrink calibration handles this for you:
it scales predictions to the loudness that maximizes the ZPTAE proxy while
preferring the whitelist's aspect band.

### Improving your score

The 1h horizon is the noisiest task on the Forge — `0/7` on a first run is
normal, and even good live models typically sit around 3–5/7 at any moment.
The whitelist integrates over long baselines, so liveness + consistency
matter as much as any single backtest. The binding metric is **Pearson r**:
once r genuinely exceeds ~0.05, the calibrated WRMSE/ZPTAE improvements
follow almost automatically. Knobs to iterate on (all env vars):

| Knob | Default | Effect |
|---|---|---|
| `DAYS_OF_HISTORY` | 1000 | More (weighted) samples → lower-variance fit. Don't truncate to "recent regime only" — see below. |
| `HALF_LIFE_DAYS` | 270 | Recency weighting: a sample this old counts half. Lower = more regime-adaptive, higher = more data-efficient. |
| `VOL_NORM_TARGET` | 1 | Train on `r/σ₁₀₀` instead of raw returns. Set `0` to A/B it. |
| `INPUT_BARS` | 128 | Lookback window; ≥101 keeps the σ₁₀₀ feature exact. |
| `FAMILIES` | `reg,clf` | Model families to A/B: return regressor and sign classifier (`(2·p_up−1)·σ₁₀₀`). The classifier wins when the edge is directional-only (DA significant, r ≈ 0). |
| `STRICT_ASPECT` | 0 | `1` = always pick a loudness inside the whitelist's ±0.5 aspect band, even when a quieter scale would score better on ZPTAE. Use it when optimizing for whitelisting rather than the leaderboard. |

**Leaderboard vs whitelist:** with a weak signal, the ZPTAE-optimal model is
*quiet* (predictions much smaller than true returns), but the whitelist's
aspect bound demands `std(pred) ≥ ~0.32·std(true)`. At that loudness, beating
the zero baseline on pure error terms needs roughly `r ≳ 0.16`. The trainer
prints both tracks (per-family best + the calibration note) so you always
know which side of the tradeoff your artifact is on.

**On training only on the last year ("regimes change"):** truncating history
is the bluntest regime tool and usually loses at this noise level — a weak
signal needs every sample you can get. The script handles regime drift two
better ways: *recency weighting* (old data fades smoothly instead of being
deleted) and *vol-normalizing the target* (dividing by trailing σ makes a
quiet 2024 hour and a violent 2026 hour statistically comparable, which is
also exactly how the topic's ZPTAE z-scores errors). If you want to test the
hypothesis anyway, run `DAYS_OF_HISTORY=365 HALF_LIFE_DAYS=10000` and compare
— that's a fair A/B.

Beyond knobs, the levers that actually move r at 1h: cross-asset lead-lag
features (ETH/SOL returns), order-flow/funding data, and intraday seasonality
interactions. The kit's `allora_research_model_skills/` bundle has three
structured methodologies for this search.

## 5. Deploy with YOUR wallet (the part that bit you)

You need the **mnemonic (seed phrase)** of the registered wallet. Where it
lives depends on how you created the address:

- **Keplr / Leap browser wallet** → Settings → "View recovery phrase".
- **`allorad keys add <name>`** → it printed the mnemonic at creation
  (`allorad keys export` can re-derive keys but not the phrase — if you lost
  it, see the note below).
- **A previous builder-kit run** → `worker_keys/<alias>.key` files each hold
  one mnemonic; `python -c "from cosmpy.aerial.wallet import LocalWallet;
  print(LocalWallet.from_mnemonic(open('worker_keys/X.key').read().strip(),'allo').address())"`
  tells you which file is which address.

> **Lost the mnemonic?** You cannot run a worker as that address. Easiest
> fix: create a wallet you control, and update the wallet address in your
> Forge account settings to the new one — then use that here.

Deploy (it verifies the mnemonic actually derives your address before doing
anything):

```bash
TOPIC_ID=<id> python scripts/deploy_my_worker.py
# mnemonic via prompt, or: export ALLORA_WALLET_MNEMONIC="word1 word2 ..."
```

On start, the worker registers on the topic (small one-time fee), then waits
for each epoch's submission window and replies with `predict(nonce)`.

## 6. Verify and monitor

One command triages everything (process, log, balance, whitelist,
registration, submissions) and prints a verdict:

```bash
python scripts/check_worker.py --topic <TOPIC_ID>
```

Right after deploying, `worker_registered=false` is normal — the SDK sends
the registration tx at startup/first window and it needs gas, so give it a
few epochs. If it stays false, the script's verdict tells you why (empty
wallet, whitelist gate, dead process, or an error in the log).

The raw checks, if you prefer curl:

```bash
# registered on the topic?
curl https://allora-api.testnet.allora.network/emissions/v9/worker_registered/<TOPIC_ID>/allo1uv65ppemwjlz0grevxz3u7hxjjg6jpqh7cz5lt

# submissions landing?
curl https://allora-api.testnet.allora.network/emissions/v9/topics/<TOPIC_ID>/workers/allo1uv65ppemwjlz0grevxz3u7hxjjg6jpqh7cz5lt/latest_inference

# local dashboards / logs
python -m allora_forge_builder_kit.web_dashboard        # http://localhost:8787
tail -f worker_logs/worker_<TOPIC_ID>_allo1uv65...log
```

Within a few epochs you should appear on the competition leaderboard
(Hammers accrue per scored epoch). Scores for an epoch are revealed after
the 1h ground truth matures.

## 7. Keep it alive and improving

- The worker must run 24/7 — a small VPS is plenty. `WorkerManager.start_all()`
  restarts enabled workers after a reboot.
- Retrain on fresh data regularly (daily is reasonable) and redeploy with the
  same command — `deploy_my_worker.py` passes `replace=True`, which swaps the
  artifact without touching your wallet.
- Liveness and consistency matter for mainnet whitelisting, not just accuracy.

### Alternative: reuse this repo's MLOps stack

This repo's self-updating stack (daily retrain → gate → serve) targets topic
69's 24h horizon, but the horizon is just config. To point it at the 1h task:

```bash
ALLORA_HORIZON_HOURS=1 make train     # 1h target: ln(close[t+1]/close[t]) on 1h bars
make serve                            # GET /inference/BTC → predicted 1h log-return
```

Then run [allora-offchain-node](https://github.com/allora-network/allora-offchain-node)
with `allora/config.example.json`, setting `topicId` to the 1h topic and
`addressRestoreMnemonic` to your registered wallet's mnemonic. This is the
heavier path; start with the builder kit one above, switch later if you want
the retrain/gate automation.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Predictions never appear on your Forge account | Worker is signing with an auto-created wallet, not your registered one. Redeploy via `scripts/deploy_my_worker.py`; check `wm.list_identities()`. |
| `this mnemonic derives allo1xyz...` error | Wrong seed phrase. Locate the right one (section 5) or update the Forge account's wallet address. |
| Worker exits immediately | `worker_logs/*.log` has the reason — usually missing `ALLORA_API_KEY`, empty balance (faucet), or a `predict.pkl` that raises. |
| `predict.pkl` fails to load on redeploy | Export it by running `train_1h_model.py` as a script (cloudpickle captures by value only from `__main__`). |
| `TypeError: cannot pickle '_thread.lock'` on export | Fixed in the current trainer: the artifact rebuilds its live data connection lazily instead of capturing the workflow (Binance's websocket client holds locks). Pull the latest `scripts/train_1h_model.py`. |
| Trained on `binance` unintentionally | The trainer falls back to Binance when `ALLORA_API_KEY` isn't in the environment — check the "Data source:" line at startup. Either source works, but be consistent between runs you compare; the artifact records which source it uses for live data (printed at export). |
| Submissions revert / out of gas | Balance ran dry — hit the faucet again. |
| Worker log shows `EOFError: Ran out of input` | The deployed `predict.pkl` is empty/truncated (a training run crashed mid-export). Re-run the trainer and redeploy — both scripts now guard against this (atomic verified export; deploy refuses unloadable artifacts). |
| Submissions fail with `failed to validate worker data bundle: signature verification failed: unauthorized` (gas is still consumed) | **Not** a wallet/registration/timing problem — the chain checks the bundle signature *before* registration (a single clean submission would pass). It's the SDK submitting the **same nonce twice** (it runs a polling loop *and* a websocket subscription); the two bundle-signings race and corrupt each other, so both land on-chain and fail validation. Tell-tale: failing nonces show two `👉 Found new nonce` / two predictions, while a nonce handled by one path succeeds. Fixes, in order: (1) `pip install -U allora-sdk` — this dual-path dedupe is the SDK's job and newer versions are the real fix; (2) introspect your installed SDK for a single-path/dedupe option — `python -c "import inspect,allora_sdk.worker as w; print(inspect.signature(w.AlloraWorker.__init__)); print(inspect.signature(w.AlloraWorker.run))"` — and pass it through `worker_runtime`; (3) the current trainer memoizes the prediction per nonce and caches live data (`LIVE_CACHE_TTL`, default 90s), cutting inference from ~25s to instant so the race window nearly closes and late submissions stop. Retrain + redeploy to pick up (3). |
| Faucet says "IP added to blocklist" | The faucet blocks datacenter/VPS IPs and IPs with repeated requests (the SDK auto-drip retries can trip it). The block is on the requesting IP, not your wallet: request from your phone (mobile data) or home browser instead — tokens go to the address no matter where the request comes from. Fallback: ask in the Allora Discord. |
| Scores are terrible despite good backtest | Check you're submitting a **log-return**, not a price; check the topic ID; remember 1h log-returns are tiny (±0.002 typical) so a price-scale output destroys your ZPTAE. |
