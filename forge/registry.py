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

    A candidate must (a) show a real edge -- positive Pearson r AND directional
    accuracy >= baseline_min_da -- and (b) be at least as good as the current
    production model on Pearson r within ``gate_tolerance``.
    """
    r = new_metrics["pearson_r"]
    da = new_metrics["directional_acc"]
    if not (r > 0 and da >= config.baseline_min_da):
        return False, (f"failed baseline: r={r:.4f} (need >0), "
                       f"da={da:.3f} (need >={config.baseline_min_da})")
    if current_metrics is None:
        return True, "no current model; promoting first candidate"
    cur_r = current_metrics.get("pearson_r", float("-inf"))
    if r >= cur_r - config.gate_tolerance:
        return True, f"r {r:.4f} >= current {cur_r:.4f} - tol {config.gate_tolerance}"
    return False, f"regression: r {r:.4f} < current {cur_r:.4f} - tol {config.gate_tolerance}"


def append_metrics(config, record: dict) -> None:
    config.ensure_dirs()
    with open(config.metrics_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
