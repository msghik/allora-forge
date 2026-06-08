# On-chain alt-data via Dune Analytics

On-chain flows are the rules-endorsed frontier for pushing **directional accuracy**.
The worker pulls them from **Dune** saved queries and turns them into scale-free,
recency-aware features that join the model alongside price / order-flow / futures.

Each canonical metric maps to **one Dune saved query** that returns exactly two
columns — **`ts`** (timestamp) and **`value`**. The system fetches the latest
cached results (cheap), merges with history, forward-fills onto the 5-minute grid,
and z-scores. Any metric you don't configure (or that fails) degrades to neutral,
so the worker never breaks.

## Setup
1. Create the queries below on Dune (one per metric). Each **must** output `ts, value`.
2. Copy each query's id from its URL (`dune.com/queries/<ID>`).
3. Set the env vars (see `.env.example`):
   ```
   ALLORA_DUNE_API_KEY=<your key>
   ALLORA_DUNE_QUERY_STABLE_SUPPLY=<id>
   ALLORA_DUNE_QUERY_CEX_NETFLOW=<id>
   ALLORA_DUNE_QUERY_DEX_VOLUME=<id>
   ALLORA_DUNE_QUERY_ACTIVE_ADDR=<id>
   # ALLORA_DUNE_EXECUTE=true   # re-run queries each cycle (spends credits) instead of cached results
   ```
4. **Recommended:** schedule each query to refresh on Dune (hourly) so cached
   results stay fresh without spending API credits, or set `ALLORA_DUNE_EXECUTE=true`.
5. `docker compose up -d --build` then `... python -m forge.pipeline --once`. The
   log shows `[onchain] <metric>: N points` per metric and `n_features` rises by 6.

> The SQL below are **starting templates** built on Dune spellbook tables. Verify
> table/column names against the current Dune schema and your plan's access, and
> keep the output as exactly `ts, value`. Each metric is independent — start with
> whichever you can get working; the rest stay neutral.

## Queries

### `stable_supply` — USDT+USDC supply on Ethereum (mint/burn pressure)
```sql
WITH flows AS (
  SELECT date_trunc('hour', evt_block_time) AS ts,
         SUM(CASE WHEN "from" = 0x0000000000000000000000000000000000000000 THEN CAST(value AS double)/1e6
                  WHEN "to"   = 0x0000000000000000000000000000000000000000 THEN -CAST(value AS double)/1e6
                  ELSE 0 END) AS net_mint
  FROM erc20_ethereum.evt_Transfer
  WHERE contract_address IN (
        0xdAC17F958D2ee523a2206206994597C13D831ec7,   -- USDT
        0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48)   -- USDC
    AND ("from" = 0x0000000000000000000000000000000000000000
      OR "to"   = 0x0000000000000000000000000000000000000000)
    AND evt_block_time > now() - interval '400' day
  GROUP BY 1)
SELECT ts, SUM(net_mint) OVER (ORDER BY ts) AS value
FROM flows ORDER BY ts;
```

### `cex_netflow` — net token flow INTO centralized exchanges (USD; +inflow = sell pressure)
```sql
WITH cex AS (SELECT address FROM cex_evms.addresses WHERE blockchain = 'ethereum')
SELECT date_trunc('hour', t.block_time) AS ts,
       SUM(CASE WHEN t.to     IN (SELECT address FROM cex) THEN t.amount_usd
                WHEN t."from" IN (SELECT address FROM cex) THEN -t.amount_usd
                ELSE 0 END) AS value
FROM tokens_ethereum.transfers t
WHERE t.block_time > now() - interval '60' day
  AND (t.to IN (SELECT address FROM cex) OR t."from" IN (SELECT address FROM cex))
GROUP BY 1 ORDER BY 1;
```

### `dex_volume` — total DEX volume (USD, risk-on proxy)
```sql
SELECT date_trunc('hour', block_time) AS ts, SUM(amount_usd) AS value
FROM dex.trades
WHERE block_time > now() - interval '60' day
GROUP BY 1 ORDER BY 1;
```

### `active_addr` — active Ethereum addresses (network activity)
```sql
SELECT date_trunc('hour', block_time) AS ts, COUNT(DISTINCT "from") AS value
FROM ethereum.transactions
WHERE block_time > now() - interval '60' day
GROUP BY 1 ORDER BY 1;
```

## Features produced (`forge/features.py :: add_onchain_features`)
| Metric | Features |
|--------|----------|
| `stable_supply` | `stable_supply_chg24` (24h log change), `stable_supply_z` (z of mint/burn) |
| `cex_netflow` | `cex_netflow_z` (instant z), `cex_netflow_sum24_z` (24h-sum z) |
| `dex_volume` | `dex_vol_z` (log-z) |
| `active_addr` | `active_addr_z` (log-z) |

All are z-scored / log-changed → scale-free, forward-filled from the metric's
native (hourly) cadence, and neutral (0) wherever data is missing.

> **Horizon caveat.** On-chain flows are strongest at multi-hour/daily horizons;
> at the 1h target their marginal lift may be modest. Treat this as an experiment —
> watch DA on the next `--once` and the live reconciliation, and prune metrics that
> don't earn their place.
