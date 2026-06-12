#!/usr/bin/env python3
"""Deploy predict.pkl to a topic using YOUR registered Forge wallet.

The Forge only credits predictions submitted from the wallet address you
registered on forge.allora.network. The builder kit's default flow creates a
brand-new wallet per worker — predictions from that wallet would never show
up on your Forge account. This script instead imports your wallet:

  1. takes your mnemonic (env ALLORA_WALLET_MNEMONIC, --mnemonic-file, or prompt)
  2. derives the allo1... address from it and REFUSES to deploy unless it
     matches your registered address (catches wrong-mnemonic mistakes early)
  3. deploys predict.pkl for the topic and starts the worker process

Usage (from the repo root, with the builder kit installed and predict.pkl built):

    export ALLORA_API_KEY=UP-...                # worker runtime needs it
    TOPIC_ID=<id> python scripts/deploy_my_worker.py

The expected address defaults to the one registered on the Forge; override
with --address or FORGE_WALLET_ADDRESS if yours differs.
"""
from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

# The wallet address registered on forge.allora.network — predictions must
# come from this address to count for the competition leaderboard.
DEFAULT_ADDRESS = os.environ.get(
    "FORGE_WALLET_ADDRESS", "allo1uv65ppemwjlz0grevxz3u7hxjjg6jpqh7cz5lt"
)


def read_mnemonic(args: argparse.Namespace) -> str:
    if args.mnemonic_file:
        return Path(args.mnemonic_file).read_text().strip()
    env = os.environ.get("ALLORA_WALLET_MNEMONIC", "").strip()
    if env:
        return env
    return getpass.getpass("Paste the mnemonic (seed phrase) of your Forge wallet: ").strip()


def derive_address(mnemonic: str) -> str:
    from cosmpy.aerial.wallet import LocalWallet

    return str(LocalWallet.from_mnemonic(mnemonic, "allo").address())


def validate_artifact(path: Path) -> None:
    """Refuse to deploy an artifact that can't possibly run.

    A training run that crashes mid-export can leave a truncated/empty
    predict.pkl behind; deploying it crashes the worker at startup with
    'EOFError: Ran out of input'. Catch that here instead.
    """
    size = path.stat().st_size
    if size < 1024:
        sys.exit(f"{path} is only {size} bytes — almost certainly a truncated export "
                 "from a crashed training run. Re-run scripts/train_1h_model.py first.")
    import pickle
    try:
        with open(path, "rb") as f:
            pickle.load(f)
    except Exception as e:
        sys.exit(f"{path} failed to load ({type(e).__name__}: {e}).\n"
                 "Re-run scripts/train_1h_model.py to regenerate it.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topic", type=int, default=int(os.environ.get("TOPIC_ID", "0")),
                    help="topic ID (find it with scripts/find_topic_id.py)")
    ap.add_argument("--artifact", default=os.environ.get("PREDICT_PKL", "predict.pkl"))
    ap.add_argument("--address", default=DEFAULT_ADDRESS,
                    help="expected allo1... address (your registered Forge wallet)")
    ap.add_argument("--mnemonic-file", default=None, help="file containing the wallet mnemonic")
    args = ap.parse_args()

    if not args.topic:
        sys.exit("TOPIC_ID is required — run scripts/find_topic_id.py first, then "
                 "TOPIC_ID=<id> python scripts/deploy_my_worker.py")
    if not Path(args.artifact).exists():
        sys.exit(f"{args.artifact} not found — run scripts/train_1h_model.py first")
    validate_artifact(Path(args.artifact))

    # Worker runtime reads the API key from env or .allora_api_key in cwd.
    if not os.environ.get("ALLORA_API_KEY") and not Path(".allora_api_key").exists():
        sys.exit("ALLORA_API_KEY not set and no .allora_api_key file in cwd — the worker "
                 "process needs it. Get a free key at https://developer.allora.network")

    from allora_forge_builder_kit import WorkerManager

    wm = WorkerManager()
    known = {i.address for i in wm.list_identities()}

    if args.address in known:
        print(f"Wallet {args.address} already imported (worker_keys/) — reusing it.")
        mnemonic = None
    else:
        mnemonic = read_mnemonic(args)
        derived = derive_address(mnemonic)
        if derived != args.address:
            print(f"ERROR: this mnemonic derives {derived}", file=sys.stderr)
            print(f"       but your registered Forge wallet is {args.address}", file=sys.stderr)
            print("Submissions from the wrong address are NOT credited to your Forge "
                  "account. Find the mnemonic of the registered wallet (see "
                  "docs/FORGE_1H_WORKER.md), or pass --address to use this one anyway.",
                  file=sys.stderr)
            return 1
        print(f"Mnemonic verified — derives {derived} (matches registered wallet).")

    result = wm.deploy_worker(
        topic_id=args.topic,
        artifact_path=args.artifact,
        address=args.address,
        mnemonic=mnemonic,
        replace=True,  # re-deploys update the artifact instead of minting a new wallet
    )
    print(f"{result.action}: {result.message}")
    if result.address_assigned != args.address:
        print(f"ERROR: manager assigned {result.address_assigned} instead of "
              f"{args.address} — aborting before start.", file=sys.stderr)
        return 1

    wm.start_worker(args.topic, args.address)
    status = wm.status_worker(args.topic, args.address)
    print(f"Worker status: {status['status']} (pid {status.get('last_pid')})")
    print(f"Log: worker_logs/worker_{args.topic}_{args.address}.log")
    print("\nVerify on-chain once it has run for a few epochs:")
    print(f"  curl https://allora-api.testnet.allora.network/emissions/v9/worker_registered/{args.topic}/{args.address}")
    print(f"  curl https://allora-api.testnet.allora.network/emissions/v9/topics/{args.topic}/workers/{args.address}/latest_inference")
    print("\nDashboards:")
    print("  python -m allora_forge_builder_kit.web_dashboard   # http://localhost:8787")
    print("  python -m allora_forge_builder_kit.workerctl dashboard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
