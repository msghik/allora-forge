"""End-to-end tests for the Forge worker, fully offline (synthetic 5m OHLCV).

Covers competition metrics, the promotion gate, cross-asset feature building, the
daily cycle producing a calibrated portable predict.pkl, and the inference server.
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

# Point config at a temp workspace BEFORE importing modules that read env at import.
_TMP = tempfile.mkdtemp(prefix="forge_test_")
os.environ["ALLORA_DATA_DIR"] = os.path.join(_TMP, "data")
os.environ["ALLORA_MODELS_DIR"] = os.path.join(_TMP, "models")

from forge import data as fdata  # noqa: E402
from forge import evaluate, export, features, pipeline, registry  # noqa: E402
from forge.config import Config  # noqa: E402


def _mk(seed, p0, n=288 * 60):  # ~60 days of 5-minute candles
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    ret = rng.normal(0, 0.0015, n)
    close = p0 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0003, n)))
    vol = np.abs(rng.normal(100, 20, n))
    df = pd.DataFrame({"open": open_, "high": high, "low": low,
                       "close": close, "volume": vol}, index=idx)
    df.index.name = "date"
    return df


_DATA = {"BTC/USDT": _mk(0, 30000), "ETH/USDT": _mk(1, 3000)}


def _fetcher(config, symbol, since_ms):
    return _DATA[symbol].copy()


def test_competition_metrics_keys():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.01, 3000)
    pred = 0.5 * y + rng.normal(0, 0.01, 3000)
    m = evaluate.competition_metrics(y, pred, step=1)
    for k in ("pearson_r", "pearson_p", "directional_acc", "da_ci_low", "da_p",
              "log_aspect_ratio", "wrmse_impr", "zptae_impr", "n"):
        assert k in m
    assert m["pearson_r"] > 0
    assert evaluate.whitelist_report(m)["total"] == len(evaluate.WHITELIST)


def test_cross_features_complete():
    cols = features.active_feature_cols("eth")
    assert "eth_log_return" in cols and "x_corr_48" in cols
    feats = features.build_features(_DATA["BTC/USDT"].iloc[:4000],
                                    ref_df=_DATA["ETH/USDT"].iloc[:4000],
                                    cross_prefix="eth")
    assert len(feats) > 0
    for col in cols:
        assert col in feats.columns
    assert not feats[cols].iloc[-1].isna().any()  # no NaN in the inference row


def test_gate_logic():
    cfg = Config.from_env()
    assert registry.gate({"pearson_r": 0.05, "whitelist_passed": 3, "zptae_impr": 0.1}, None, cfg)[0]
    assert not registry.gate({"pearson_r": -0.01, "whitelist_passed": 8}, None, cfg)[0]
    assert registry.gate({"pearson_r": 0.05, "whitelist_passed": 4, "zptae_impr": 0.0},
                         {"whitelist_passed": 3, "zptae_impr": 0.5}, cfg)[0]
    assert not registry.gate({"pearson_r": 0.05, "whitelist_passed": 2, "zptae_impr": 0.9},
                             {"whitelist_passed": 3, "zptae_impr": 0.1}, cfg)[0]


def test_run_once_cross_asset_portable():
    cfg = Config.from_env()
    assert cfg.cross_prefix == "eth"
    rec = pipeline.run_once(cfg, fetcher=_fetcher)
    assert rec["version"] in registry.list_versions(cfg)
    pkl = os.path.join(cfg.version_dir(rec["version"]), "predict.pkl")
    assert os.path.exists(pkl)

    predict = export.load_predict(pkl)
    out = predict(_DATA["BTC/USDT"].tail(500), _DATA["ETH/USDT"].tail(500))
    assert isinstance(out, float)
    # cross-asset model must be given the reference asset
    with pytest.raises(ValueError):
        predict(_DATA["BTC/USDT"].tail(500))

    # version metadata is always written (promotion depends on the noisy gate)
    import json
    with open(os.path.join(cfg.version_dir(rec["version"]), "metadata.json")) as f:
        meta = json.load(f)
    assert meta.get("cross_symbol") == "ETH/USDT"
    assert meta.get("n_features") == len(features.active_feature_cols("eth"))


def test_server_inference_returns_float(monkeypatch):
    cfg = Config.from_env()
    versions = registry.list_versions(cfg)
    assert versions, "run_once test must have produced a version"
    registry.promote(cfg, versions[-1])

    monkeypatch.setattr(fdata, "get_recent_candles",
                        lambda config, symbol=None, **kw: _DATA[symbol or config.symbol].tail(cfg.recent_candles))

    from fastapi.testclient import TestClient
    from forge import server
    client = TestClient(server.app)

    assert client.get("/health").json()["has_model"] is True
    r = client.get("/inference/BTC")
    assert r.status_code == 200
    assert isinstance(r.json(), float)
