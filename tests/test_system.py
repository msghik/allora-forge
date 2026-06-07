"""End-to-end tests for the Forge worker, fully offline (synthetic OHLCV).

Covers the critical paths: the promotion gate, the daily cycle producing a
portable predict.pkl, and the inference server returning a float.
"""
import os
import tempfile

import numpy as np
import pandas as pd

# Point config at a temp workspace BEFORE importing modules that read env at import.
_TMP = tempfile.mkdtemp(prefix="forge_test_")
os.environ["ALLORA_DATA_DIR"] = os.path.join(_TMP, "data")
os.environ["ALLORA_MODELS_DIR"] = os.path.join(_TMP, "models")

from forge import data as fdata  # noqa: E402
from forge import export, pipeline, registry  # noqa: E402
from forge.config import Config  # noqa: E402


def _synth(n=24 * 160, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    ret = rng.normal(0, 0.005, n)
    close = 30000 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.001, n)))
    vol = np.abs(rng.normal(100, 20, n))
    df = pd.DataFrame({"open": open_, "high": high, "low": low,
                       "close": close, "volume": vol}, index=idx)
    df.index.name = "date"
    return df


_DATA = _synth()


def _fetcher(config, since_ms):
    # Ignore pagination; the pipeline applies its own rolling window.
    return _DATA.copy()


def test_gate_logic():
    cfg = Config.from_env()
    edge = {"pearson_r": 0.10, "directional_acc": 0.55}
    # No current model + real edge -> promote.
    ok, _ = registry.gate(edge, None, cfg)
    assert ok
    # Negative r fails the baseline.
    ok, _ = registry.gate({"pearson_r": -0.01, "directional_acc": 0.55}, None, cfg)
    assert not ok
    # Worse than current -> rejected (regression guard).
    ok, _ = registry.gate({"pearson_r": 0.05, "directional_acc": 0.55},
                          {"pearson_r": 0.10}, cfg)
    assert not ok
    # Better than current -> promoted.
    ok, _ = registry.gate({"pearson_r": 0.12, "directional_acc": 0.55},
                          {"pearson_r": 0.10}, cfg)
    assert ok


def test_run_once_creates_portable_predict():
    cfg = Config.from_env()
    rec = pipeline.run_once(cfg, fetcher=_fetcher)
    assert rec["version"] in registry.list_versions(cfg)
    pkl = os.path.join(cfg.version_dir(rec["version"]), "predict.pkl")
    assert os.path.exists(pkl)
    assert os.path.exists(cfg.metrics_path)

    # Reload with stdlib pickle and call -> must be a float (portability).
    predict = export.load_predict(pkl)
    out = predict(_DATA.tail(120))
    assert isinstance(out, float)


def test_server_inference_returns_float(monkeypatch):
    cfg = Config.from_env()
    versions = registry.list_versions(cfg)
    assert versions, "run_once test must have produced a version"
    registry.promote(cfg, versions[-1])  # ensure a current model exists

    monkeypatch.setattr(fdata, "get_recent_candles",
                        lambda config, **kw: _DATA.tail(cfg.recent_candles))

    from fastapi.testclient import TestClient
    from forge import server
    client = TestClient(server.app)

    h = client.get("/health").json()
    assert h["has_model"] is True

    r = client.get("/inference/BTC")
    assert r.status_code == 200
    assert isinstance(r.json(), float)
