"""Filesystem model registry: immutable versioned models, a promotion gate that
prevents regressions, a ``current`` pointer the server reads, and rollback.

Layout::

    models/
      2026-06-06T01-00-00Z/   predict.pkl  metadata.json
      2026-06-07T01-00-00Z/   predict.pkl  metadata.json
      current/                predict.pkl  metadata.json   <- active model
      metrics.jsonl                                         <- append-only history
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone

log = logging.getLogger("forge.registry")


def new_version_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return os.environ.get("ALLORA_GIT_SHA", "unknown")


def save_metadata(config, version: str, metadata: dict) -> str:
    path = os.path.join(config.version_dir(version), "metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    return path


def get_current_metadata(config) -> dict | None:
    if os.path.exists(config.current_metadata):
        with open(config.current_metadata) as f:
            return json.load(f)
    return None


def list_versions(config) -> list[str]:
    if not os.path.isdir(config.models_dir):
        return []
    out = []
    for name in os.listdir(config.models_dir):
        d = os.path.join(config.models_dir, name)
        if name != "current" and os.path.isdir(d) \
                and os.path.exists(os.path.join(d, "predict.pkl")):
            out.append(name)
    return sorted(out)


def promote(config, version: str) -> None:
    """Copy a version's artifacts into ``current/`` (atomically enough)."""
    src = config.version_dir(version)
    os.makedirs(config.current_dir, exist_ok=True)
    for fname in ("predict.pkl", "metadata.json"):
        s = os.path.join(src, fname)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(config.current_dir, fname))
    log.info("promoted %s -> current", version)


def rollback(config, to_version: str | None = None) -> str | None:
    """Promote the previous version (or a specific one). Returns the version."""
    versions = list_versions(config)
    if to_version is None:
        cur = get_current_metadata(config)
        cur_v = cur.get("version") if cur else None
        candidates = [v for v in versions if v != cur_v]
        if not candidates:
            log.warning("no version available to roll back to")
            return None
        to_version = candidates[-1]
    promote(config, to_version)
    return to_version


def gate(new_metrics: dict, current_metrics: dict | None, config) -> tuple[bool, str]:
    """Decide whether to promote the new candidate.

    Sanity floor: positive Pearson r (a real, non-anti-predictive signal).
    Primary: number of whitelist criteria passed (more is better). Tiebreak /
    no-regression: the ZPTAE-improvement surrogate within ``gate_tolerance``.
    """
    r = new_metrics.get("pearson_r", 0.0)
    if not (r > 0):
        return False, f"failed baseline: pearson_r={r:.4f} (need > 0)"
    if current_metrics is None:
        return True, "no current model; promoting first candidate"
    new_wl = new_metrics.get("whitelist_passed", 0)
    cur_wl = current_metrics.get("whitelist_passed", 0)
    if new_wl > cur_wl:
        return True, f"whitelist {new_wl} > current {cur_wl}"
    if new_wl < cur_wl:
        return False, f"regression: whitelist {new_wl} < current {cur_wl}"
    new_z = new_metrics.get("zptae_impr", float("-inf"))
    cur_z = current_metrics.get("zptae_impr", float("-inf"))
    if new_z >= cur_z - config.gate_tolerance:
        return True, f"whitelist {new_wl}=={cur_wl}, zptae_impr {new_z:.4f} >= {cur_z:.4f}-tol"
    return False, f"whitelist tie, zptae_impr {new_z:.4f} < {cur_z:.4f}-tol"


def append_metrics(config, record: dict) -> None:
    config.ensure_dirs()
    with open(config.metrics_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
