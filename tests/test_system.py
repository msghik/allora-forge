"""End-to-end tests for the Forge worker, fully offline (synthetic 5m OHLCV).

Covers competition metrics, the promotion gate, the daily cycle producing a
calibrated portable predict.pkl, and the inference server returning a float.
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
from forge import evaluate, export, pipeline, registry  # noqa: E402
from forge.config import Config  # noqa: E402


def _synth(n=288 * 60, seed=0):  # ~60 days of 5-minute candles
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    ret = rng.normal(0, 0.0015, n)
    close = 30000 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0003, n)))
    vol = np.abs(rng.normal(100, 20, n))
    df = pd.DataFrame({"open": open_, "high": high, "low": low,
                       "close": close, "volume": vol}, index=idx)
    df.index.name = "date"
    return df


_DATA = _synth()


def _fetcher(config, since_ms):
    return _DATA.copy()


def test_competition_metrics_keys():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.01, 3000)
    pred = 0.5 * y + rng.normal(0, 0.01, 3000)  # genuinely correlated
    m = evaluate.competition_metrics(y, pred, step=1)
    for k in ("pearson_r", "pearson_p", "directional_acc", "da_ci_low", "da_p",
              "log_aspect_ratio", "wrmse_impr", "zptae_impr", "n"):
        assert k in m
    assert m["pearson_r"] > 0
    wl = evaluate.whitelist_report(m)
    assert wl["total"] == len(evaluate.WHITELIST)


def test_gate_logic():
    cfg = Config.from_env()
    assert registry.gate({"pearson_r": 0.05, "zptae_impr": 0.10}, None, cfg)[0]
    assert not registry.gate({"pearson_r": -0.01, "zptae_impr": 0.50}, None, cfg)[0]
    assert not registry.gate({"pearson_r": 0.05, "zptae_impr": 0.05},
                             {"zptae_impr": 0.10}, cfg)[0]
    assert registry.gate({"pearson_r": 0.05, "zptae_impr": 0.12},
                         {"zptae_impr": 0.10}, cfg)[0]


def test_run_once_creates_portable_predict():
    cfg = Config.from_env()
    rec = pipeline.run_once(cfg, fetcher=_fetcher)
    assert rec["version"] in registry.list_versions(cfg)
    pkl = os.path.join(cfg.version_dir(rec["version"]), "predict.pkl")
    assert os.path.exists(pkl)
    assert os.path.exists(cfg.metrics_path)

    predict = export.load_predict(pkl)
    out = predict(_DATA.tail(200))
    assert isinstance(out, float)

    # metadata carries the whitelist readout and calibration scale
    meta = registry.get_current_metadata(cfg) or {}
    if meta:
        assert "whitelist" in meta and "scale" in meta


def test_server_inference_returns_float(monkeypatch):
    cfg = Config.from_env()
    versions = registry.list_versions(cfg)
    assert versions, "run_once test must have produced a version"
    registry.promote(cfg, versions[-1])

    monkeypatch.setattr(fdata, "get_recent_candles",
                        lambda config, **kw: _DATA.tail(cfg.recent_candles))

    from fastapi.testclient import TestClient
    from forge import server
    client = TestClient(server.app)

    assert client.get("/health").json()["has_model"] is True
    r = client.get("/inference/BTC")
    assert r.status_code == 200
    assert isinstance(r.json(), float)
