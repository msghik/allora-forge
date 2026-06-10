#!/usr/bin/env python3
"""Find the on-chain topic ID for a Forge competition (e.g. 1h BTC/USD log-return).

Forge competition pages don't always show the chain topic ID, but every
competition maps to a topic on the Allora chain whose ``metadata`` string
describes it. This script scans all topics via the public REST API and
highlights the ones matching a BTC + log-return + 1-hour pattern.

Stdlib only — no dependencies. Run it on a machine with open internet:

    python scripts/find_topic_id.py                       # scan testnet
    python scripts/find_topic_id.py --all                 # print every topic
    python scripts/find_topic_id.py --address allo1...    # also check registration
    python scripts/find_topic_id.py --pattern "eth.*8h"   # custom search
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request

API_BASES = {
    "testnet": "https://allora-api.testnet.allora.network",
    "mainnet": "https://allora-api.mainnet.allora.network",
}
EMISSIONS = "emissions/v9"

# Matches e.g. "1h BTC/USD Log-Return", "BTC/USD - Log Returns - 1h", "1 hour BTC ..."
DEFAULT_PATTERN = r"btc.*(log[\s_-]?return).*((?<!\d)1\s?h(our)?\b)|((?<!\d)1\s?h(our)?\b).*btc.*(log[\s_-]?return)"


def get_json(url: str, timeout: float = 15.0):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def fetch_topic(base: str, topic_id: int) -> dict | None:
    try:
        data = get_json(f"{base}/{EMISSIONS}/topics/{topic_id}")
    except Exception:
        return None
    topic = data.get("topic") or {}
    if not topic:
        return None
    try:
        active = bool(get_json(f"{base}/{EMISSIONS}/is_topic_active/{topic_id}").get("is_active"))
    except Exception:
        active = None
    return {
        "id": int(topic.get("id", topic_id)),
        "metadata": topic.get("metadata", ""),
        "loss_method": topic.get("loss_method", ""),
        "epoch_length": topic.get("epoch_length", ""),
        "submission_window": topic.get("worker_submission_window", ""),
        "active": active,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--network", choices=API_BASES, default="testnet")
    ap.add_argument("--api", default=None, help="override REST API base URL")
    ap.add_argument("--pattern", default=DEFAULT_PATTERN, help="regex matched against topic metadata (case-insensitive)")
    ap.add_argument("--all", action="store_true", help="print every topic, not just matches")
    ap.add_argument("--address", default=None, help="also check whether this allo1... address is registered on matching topics")
    args = ap.parse_args()

    base = args.api or API_BASES[args.network]
    try:
        next_id = int(get_json(f"{base}/{EMISSIONS}/next_topic_id")["next_topic_id"])
    except Exception as e:
        print(f"Could not reach {base}: {e}", file=sys.stderr)
        print("Check your connection, or pass --api with a working REST endpoint.", file=sys.stderr)
        return 1

    print(f"Scanning topics 1..{next_id - 1} on {args.network} ({base})\n")
    rx = re.compile(args.pattern, re.IGNORECASE)
    matches = []
    for tid in range(1, next_id):
        t = fetch_topic(base, tid)
        if t is None:
            continue
        is_match = bool(rx.search(t["metadata"]))
        if is_match:
            matches.append(t)
        if args.all or is_match:
            flag = "  <-- MATCH" if is_match else ""
            print(f"  [{t['id']:>3}] active={t['active']} epoch={t['epoch_length']:>6} "
                  f"window={t['submission_window']:>4} loss={t['loss_method']:<10} {t['metadata']!r}{flag}")

    if not matches:
        print("\nNo topic matched the pattern. Re-run with --all and eyeball the metadata strings;")
        print("then use that topic ID directly. Patterns differ slightly between competitions.")
        return 2

    print(f"\n{len(matches)} match(es). Use the ACTIVE one as TOPIC_ID for training/deploy.")
    if args.address:
        for t in matches:
            try:
                reg = get_json(f"{base}/{EMISSIONS}/worker_registered/{t['id']}/{args.address}")
                print(f"  topic {t['id']}: worker_registered({args.address}) = {reg.get('is_registered')}")
            except Exception as e:
                print(f"  topic {t['id']}: registration check failed: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
