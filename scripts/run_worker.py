#!/usr/bin/env python3
"""Single-path worker runner — websocket-only submissions, no polling race.

Why this exists
---------------
``AlloraWorker`` runs TWO submission paths at once: a polling loop
(``polling_interval``, default 120s) *and* a websocket subscription to
worker-submission-window events. When both fire for the same nonce, the SDK
builds and signs two bundles whose signing races and corrupts each other —
the chain then rejects BOTH with::

    failed to validate worker data bundle: signature verification failed: unauthorized

(while still charging gas). Upgrading the SDK does not change this — the
latest ``AlloraWorker`` still exposes both ``polling_interval`` and the
websocket event type. On topic 72 the only submission that SUCCEEDED was the
one a single (websocket) path handled.

This runner sets ``polling_interval`` to a very large value, so only the
websocket path submits — one clean bundle per nonce. It mirrors the builder
kit's ``worker_runtime`` otherwise (same float/finite guards, same network +
wallet config), and stops any WorkerManager-run worker for the same
topic/address first so you never have two processes double-submitting.

Usage
-----
    # uses worker_keys/ that deploy_my_worker.py already created, and the
    # already-funded wallet:
    TOPIC_ID=72 python scripts/run_worker.py

    # or be explicit:
    python scripts/run_worker.py --topic 72 --predict predict.pkl \
        --mnemonic-file worker_keys/<alias>.key --polling-interval 86400

Keep it running under tmux/systemd. Logs go to stdout (and worker_logs/ if
you redirect). Verify with: python scripts/check_worker.py --topic 72
"""
from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
import tempfile
from pathlib import Path

DEFAULT_ADDRESS = os.environ.get(
    "FORGE_WALLET_ADDRESS", "allo1uv65ppemwjlz0grevxz3u7hxjjg6jpqh7cz5lt"
)
# Large but finite: the websocket drives submissions; polling effectively
# never fires (one startup poll at most). ~1 day.
DEFAULT_POLLING_INTERVAL = int(os.environ.get("WORKER_POLLING_INTERVAL", "86400"))
_TESTNET_FAUCET_URL = "https://faucet.testnet.allora.run"


def _load_api_key(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("ALLORA_API_KEY")
    if env:
        return env
    for path in (".allora_api_key", "notebooks/.allora_api_key"):
        if os.path.exists(path):
            return open(path).read().strip()
    raise SystemExit("ALLORA_API_KEY not found (env var or .allora_api_key file)")


def _resolve_mnemonic_file(args: argparse.Namespace) -> str:
    """Find the wallet key file: explicit flag, env mnemonic, the file
    WorkerManager already wrote, or a lone worker_keys/*.key."""
    if args.mnemonic_file:
        return args.mnemonic_file
    env_mn = os.environ.get("ALLORA_WALLET_MNEMONIC", "").strip()
    if env_mn:
        fd, tmp = tempfile.mkstemp(prefix="wkey_", suffix=".key")
        os.write(fd, env_mn.encode())
        os.close(fd)
        os.chmod(tmp, 0o600)
        return tmp
    # WorkerManager keystore lookup by address
    try:
        from allora_forge_builder_kit import WorkerManager
        kf = WorkerManager(reconcile_on_start=False)._get_key_file_for_address(args.address)
        if kf:
            return str(kf)
    except Exception:
        pass
    keys = sorted(Path("worker_keys").glob("*.key")) if Path("worker_keys").exists() else []
    if len(keys) == 1:
        return str(keys[0])
    raise SystemExit(
        "Could not locate the wallet key file. Pass --mnemonic-file "
        "worker_keys/<alias>.key, or set ALLORA_WALLET_MNEMONIC.")


def _stop_manager_worker(topic_id: int, address: str) -> None:
    """Stop a WorkerManager-run worker for this topic/address so we don't run
    two processes that both submit (which would re-create the very race we're
    fixing)."""
    try:
        from allora_forge_builder_kit import WorkerManager
        wm = WorkerManager(reconcile_on_start=False)
        st = wm.status_worker(topic_id, address)
        if st and st.get("status") == "running":
            print(f"Stopping WorkerManager-run worker for topic {topic_id} (avoids double-submit)...")
            wm.stop_worker(topic_id, address)
    except Exception as e:
        print(f"  (could not check/stop a manager worker: {e})")


def _build_network(network: str, no_faucet: bool):
    from allora_sdk.rpc_client.config import AlloraNetworkConfig
    cfg = AlloraNetworkConfig.mainnet() if network == "mainnet" else AlloraNetworkConfig.testnet()
    if no_faucet:
        cfg.faucet_url = None
    elif network != "mainnet":
        cfg.faucet_url = _TESTNET_FAUCET_URL
    return cfg


async def _run(topic_id, artifact, api_key, mnemonic_file, network, no_faucet,
               polling_interval, debug) -> None:
    import cloudpickle
    from allora_sdk.worker import AlloraWorker
    from allora_sdk.rpc_client.config import AlloraWalletConfig

    with open(artifact, "rb") as f:
        raw_fn = cloudpickle.load(f)

    def run_fn(nonce: int):
        value = raw_fn(nonce)
        try:
            v = float(value)
        except Exception as e:
            raise RuntimeError(f"Invalid inference output type: {value!r}") from e
        if not math.isfinite(v):
            raise RuntimeError(f"Invalid inference output (non-finite): {v}")
        return v

    wallet_cfg = AlloraWalletConfig(mnemonic_file=mnemonic_file)
    net_cfg = _build_network(network, no_faucet)
    worker = AlloraWorker(
        run=run_fn, topic_id=topic_id, api_key=api_key, wallet=wallet_cfg,
        network=net_cfg, polling_interval=polling_interval, debug=debug,
    )
    print(f"Single-path worker live: topic={topic_id} polling_interval={polling_interval}s "
          f"(websocket-driven). Ctrl-C to stop.")
    async for result in worker.run():
        if isinstance(result, Exception):
            print(f"submission error: {result!r}")
        else:
            val = getattr(result, "prediction", result)
            print(f"submitted: {val}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic", type=int, default=int(os.environ.get("TOPIC_ID", "0")))
    ap.add_argument("--predict", default=os.environ.get("PREDICT_PKL", "predict.pkl"))
    ap.add_argument("--address", default=DEFAULT_ADDRESS)
    ap.add_argument("--mnemonic-file", default=None)
    ap.add_argument("--network", default="testnet", choices=["testnet", "mainnet"])
    ap.add_argument("--polling-interval", type=int, default=DEFAULT_POLLING_INTERVAL,
                    help="seconds; large = websocket-only (default ~1 day). "
                         "Set small to re-enable the polling path.")
    ap.add_argument("--faucet", action="store_true",
                    help="allow the SDK faucet drip (default off: you're already funded, "
                         "and the faucet blocklists VPS IPs)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if not args.topic:
        return _die("TOPIC_ID (or --topic) is required — e.g. TOPIC_ID=72 python scripts/run_worker.py")
    if not Path(args.predict).exists():
        return _die(f"{args.predict} not found — run scripts/train_1h_model.py first")
    # fail fast on an empty/truncated artifact (same guard as deploy)
    if Path(args.predict).stat().st_size < 1024:
        return _die(f"{args.predict} is {Path(args.predict).stat().st_size} bytes — likely a "
                    "truncated export. Re-run scripts/train_1h_model.py.")

    api_key = _load_api_key(None)
    mnemonic_file = _resolve_mnemonic_file(args)
    _stop_manager_worker(args.topic, args.address)

    try:
        asyncio.run(_run(
            topic_id=args.topic, artifact=args.predict, api_key=api_key,
            mnemonic_file=mnemonic_file, network=args.network,
            no_faucet=not args.faucet, polling_interval=args.polling_interval,
            debug=args.debug,
        ))
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


def _die(msg: str) -> int:
    print(msg, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
