#!/usr/bin/env python3
"""Triage a deployed worker: local process + log + on-chain state, one command.

    python scripts/check_worker.py --topic 72
    python scripts/check_worker.py --topic 72 --address allo1...

Checks, in order:
  1. worker process alive (worker_state.db + /proc)
  2. recent log lines, highlighting common failure patterns
  3. wallet balance (registration + submissions need gas)
  4. topic active / epoch cadence / submission window
  5. topic worker-whitelist enabled? is this address whitelisted?
  6. worker_registered on the topic
  7. latest on-chain inference from this address

Stdlib only — run it on the worker machine from the directory you deployed
from (where worker_state.db and worker_logs/ live).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import urllib.request
from pathlib import Path

API_BASES = {
    "testnet": "https://allora-api.testnet.allora.network",
    "mainnet": "https://allora-api.mainnet.allora.network",
}
EMISSIONS = "emissions/v9"
DEFAULT_ADDRESS = os.environ.get(
    "FORGE_WALLET_ADDRESS", "allo1uv65ppemwjlz0grevxz3u7hxjjg6jpqh7cz5lt"
)
LOG_ERROR_RX = re.compile(
    r"error|exception|traceback|insufficient|fee|faucet|whitelist|denied|"
    r"unauthorized|api.?key|balance|failed|reject", re.IGNORECASE)


def get_json(base: str, path: str, timeout: float = 15.0):
    url = f"{base}/{path}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode()), None
    except Exception as e:
        return None, f"{e}"


def check_process(topic: int, address: str) -> tuple[str, int | None]:
    db = Path("worker_state.db")
    if not db.exists():
        return "no worker_state.db in cwd (run from your deploy directory)", None
    try:
        with sqlite3.connect(db) as conn:
            row = conn.execute(
                "SELECT status, last_pid FROM workers WHERE topic_id=? AND address=?",
                (topic, address)).fetchone()
    except Exception as e:
        return f"could not read worker_state.db: {e}", None
    if not row:
        return "no worker record for this topic/address in worker_state.db", None
    status, pid = row
    alive = False
    if pid:
        try:
            os.kill(int(pid), 0)
            alive = True
        except OSError:
            alive = False
    return f"db status={status!r} pid={pid} alive={alive}", (int(pid) if alive else None)


def show_log(topic: int, address: str, lines: int = 40) -> list[str]:
    path = Path("worker_logs") / f"worker_{topic}_{address}.log"
    if not path.exists():
        print(f"  (no log file at {path})")
        return []
    tail = path.read_text(errors="replace").splitlines()[-lines:]
    flagged = []
    for ln in tail:
        mark = " <-- " if LOG_ERROR_RX.search(ln) else "     "
        if LOG_ERROR_RX.search(ln):
            flagged.append(ln)
        print(f"  {mark}{ln[:160]}")
    return flagged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topic", type=int, default=int(os.environ.get("TOPIC_ID", "0")), required=False)
    ap.add_argument("--address", default=DEFAULT_ADDRESS)
    ap.add_argument("--network", choices=API_BASES, default="testnet")
    ap.add_argument("--api", default=None)
    ap.add_argument("--log-lines", type=int, default=40)
    args = ap.parse_args()
    if not args.topic:
        sys.exit("--topic (or TOPIC_ID env) is required")
    base = args.api or API_BASES[args.network]
    topic, addr = args.topic, args.address
    verdicts: list[str] = []

    print(f"== 1. worker process (topic {topic}, {addr}) ==")
    msg, alive_pid = check_process(topic, addr)
    print(f"  {msg}")
    if alive_pid is None:
        verdicts.append("worker process is NOT running — restart it: "
                        "python -c \"from allora_forge_builder_kit import WorkerManager; "
                        f"wm=WorkerManager(); wm.start_worker({topic}, '{addr}')\"")

    print(f"== 2. last {args.log_lines} log lines ==")
    flagged = show_log(topic, addr, args.log_lines)
    if flagged:
        verdicts.append(f"log contains {len(flagged)} suspicious line(s) (marked <--) — "
                        "they usually name the problem directly")

    print("== 3. wallet balance ==")
    bal, err = get_json(base, f"cosmos/bank/v1beta1/balances/{addr}")
    uallo = 0
    if bal is not None:
        for c in bal.get("balances", []):
            print(f"  {c.get('amount')} {c.get('denom')}")
            if c.get("denom") == "uallo":
                uallo = int(c.get("amount", "0"))
        if not bal.get("balances"):
            print("  (empty — 0 uallo)")
        if uallo == 0:
            verdicts.append("wallet has 0 uallo — registration/submission txs cannot be paid. "
                            "Get testnet ALLO at https://faucet.testnet.allora.run (the SDK "
                            "auto-drip can be rate-limited); then restart the worker")
    else:
        print(f"  query failed: {err}")

    print("== 4. topic state ==")
    t, err = get_json(base, f"{EMISSIONS}/topics/{topic}")
    if t and t.get("topic"):
        tt = t["topic"]
        print(f"  metadata: {tt.get('metadata')!r}")
        print(f"  epoch_length={tt.get('epoch_length')} blocks, "
              f"submission_window={tt.get('worker_submission_window')} blocks, "
              f"ground_truth_lag={tt.get('ground_truth_lag')}")
    active, _ = get_json(base, f"{EMISSIONS}/is_topic_active/{topic}")
    print(f"  active: {active.get('is_active') if active else 'query failed'}")
    if active and not active.get("is_active"):
        verdicts.append("topic is INACTIVE — no submission windows open; double-check the topic id")

    print("== 5. whitelist ==")
    wl_on, wl_err = get_json(base, f"{EMISSIONS}/is_topic_worker_whitelist_enabled/{topic}")
    if wl_on is None:
        print(f"  query failed: {wl_err}")
    else:
        enabled = bool(wl_on.get("is_topic_worker_whitelist_enabled"))
        print(f"  topic worker whitelist enabled: {enabled}")
        if enabled:
            me, _ = get_json(base, f"{EMISSIONS}/is_whitelisted_topic_worker/{topic}/{addr}")
            ok = bool(me and me.get("is_whitelisted_topic_worker"))
            print(f"  this address whitelisted: {ok}")
            if not ok:
                verdicts.append("topic gates workers and this address is NOT whitelisted — "
                                "make sure this wallet is the one registered on forge.allora.network "
                                "and that you've joined/registered for this competition on the Forge "
                                "page (whitelisting can lag a bit after joining)")

    print("== 6. registration ==")
    reg, reg_err = get_json(base, f"{EMISSIONS}/worker_registered/{topic}/{addr}")
    registered = bool(reg and reg.get("is_registered"))
    print(f"  worker_registered: {registered if reg is not None else f'query failed: {reg_err}'}")

    print("== 7. latest inference ==")
    inf, err = get_json(base, f"{EMISSIONS}/topics/{topic}/workers/{addr}/latest_inference")
    if inf and inf.get("latest_inference"):
        li = inf["latest_inference"]
        print(f"  value={li.get('value')} block={li.get('block_height')}")
    else:
        print(f"  none yet ({err or 'not found'})")

    if bal is None and t is None and reg is None:
        verdicts.append("chain API unreachable from this machine — check connectivity "
                        "or pass --api with a working REST endpoint")

    print("\n== verdict ==")
    if reg is None and not verdicts:
        verdicts.append("could not confirm registration (query failed) — re-run when the "
                        "API is reachable")
    if not verdicts:
        if registered and inf and inf.get("latest_inference"):
            print("  All good: registered and submitting. Scores appear after the 1h ground "
                  "truth matures; watch the Forge leaderboard and the web dashboard.")
        elif registered:
            print("  Registered but no inference yet — normal right after registration. "
                  "Submission happens in each epoch's window; re-check in ~10-15 min.")
        else:
            print("  No blocking issue found, but not registered yet. The SDK registers at "
                  "startup/first window; with funds present this should flip within a few "
                  "epochs. Re-run this check in ~10 min; if still false, read the full log.")
    else:
        for i, v in enumerate(verdicts, 1):
            print(f"  {i}. {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
