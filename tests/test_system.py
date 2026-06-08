"""End-to-end tests for the Forge worker, fully offline (synthetic 5m candles).

Covers competition metrics, the promotion gate, order-flow + cross-asset + futures
feature building, the sign-aware candidate set, the daily cycle producing a
calibrated portable predict.pkl, and the inference server.
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

# Point config at a temp workspace and disable the (networked) futures fetch
# BEFORE importing modules that read env at import.
_TMP = tempfile.mkdtemp(prefix="forge_test_")
os.environ["ALLORA_DATA_DIR"] = os.path.join(_TMP, "data")
os.environ["ALLORA_MODELS_DIR"] = os.path.join(_TMP, "models")
os.environ["ALLORA_FUTURES_SYMBOL"] = ""        # no network in tests; futures tested directly

from forge import data as fdata  # noqa: E402
from forge import evaluate, export, features, pipeline, registry  # noqa: E402
from forge.config import Config  # noqa: E402


def _mk(seed, p0, n=288 * 60):  # ~60 days of 5-minute candles, with order-flow cols
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    ret = rng.normal(0, 0.0015, n)
    close = p0 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0003, n)))
    vol = np.abs(rng.normal(100, 20, n))
    trades = np.abs(rng.normal(500, 80, n))
    # taker-buy share leans with the bar's direction (gives order-flow signal)
    share = np.clip(0.5 + 4 * ret + rng.normal(0, 0.05, n), 0.05, 0.95)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                       "volume": vol, "quote_volume": vol * close,
                       "trades": trades, "taker_buy_base": vol * share}, index=idx)
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


def test_orderflow_and_cross_features_complete():
    cols = features.active_feature_cols("eth", use_orderflow=True)
    assert "eth_log_return" in cols and "x_corr_48" in cols
    assert "ofi_12" in cols and "taker_buy_ratio_z" in cols
    feats = features.build_features(_DATA["BTC/USDT"].iloc[:4000],
                                    ref_df=_DATA["ETH/USDT"].iloc[:4000],
                                    cross_prefix="eth", use_orderflow=True)
    assert len(feats) > 0
    for col in cols:
        assert col in feats.columns
    assert not feats[cols].iloc[-1].isna().any()  # no NaN in the inference row


def test_futures_features_neutral_and_real():
    base = features.build_features(_DATA["BTC/USDT"].iloc[:4000], use_orderflow=True)
    # neutral: no futures frame -> columns present, all zero, no NaN
    neutral = features.add_futures_features(base, None)
    for col in features.FUTURES_COLS:
        assert col in neutral.columns
    assert not neutral[features.FUTURES_COLS].isna().any().any()
    assert float(neutral["funding_rate"].abs().sum()) == 0.0

    # real: a synthetic funding/OI frame produces non-trivial features
    fidx = base.index
    fut = pd.DataFrame({
        "funding_rate": np.linspace(-1e-4, 2e-4, len(fidx)),
        "open_interest": np.linspace(1e6, 1.3e6, len(fidx)),
    }, index=fidx)
    real = features.add_futures_features(base, fut)
    assert not real[features.FUTURES_COLS].isna().any().any()
    assert float(real["oi_change_12"].abs().sum()) > 0.0


def test_onchain_features_neutral_and_real():
    base = features.build_features(_DATA["BTC/USDT"].iloc[:4000], use_orderflow=True)
    # neutral: no on-chain frame -> columns present, all zero, no NaN
    neutral = features.add_onchain_features(base, None)
    for col in features.ONCHAIN_COLS:
        assert col in neutral.columns
    assert not neutral[features.ONCHAIN_COLS].isna().any().any()
    assert float(neutral[features.ONCHAIN_COLS].abs().sum().sum()) == 0.0

    # real: a synthetic Dune frame (hourly) produces non-trivial features
    hidx = pd.date_range(base.index[0], base.index[-1], freq="1h")
    rng = np.random.default_rng(3)
    src = pd.DataFrame({
        "stable_supply": 1e11 + np.cumsum(rng.normal(0, 1e7, len(hidx))),
        "cex_netflow": rng.normal(0, 500, len(hidx)),
        "dex_volume": np.abs(rng.normal(1e8, 2e7, len(hidx))),
        "active_addr": np.abs(rng.normal(9e5, 5e4, len(hidx))),
    }, index=hidx).rename_axis("ts")
    real = features.add_onchain_features(base, src)
    assert not real[features.ONCHAIN_COLS].isna().any().any()
    assert float(real["stable_supply_chg24"].abs().sum()) > 0.0
    assert float(real["cex_netflow_z"].abs().sum()) > 0.0


def test_core_ohlcv_fallback_no_row_wipe():
    """Regression: when raw klines fall back to core OHLCV the order-flow columns
    are all-NaN; they must NOT wipe every row via a blanket dropna (the live
    'insufficient data: 0 rows' crash)."""
    btc = _DATA["BTC/USDT"].iloc[:6000].copy()
    eth = _DATA["ETH/USDT"].iloc[:6000].copy()
    for c in ("quote_volume", "trades", "taker_buy_base"):
        btc[c] = np.nan          # simulate ccxt core-OHLCV fallback (no order flow)
        eth[c] = np.nan
    fut = pd.DataFrame({"funding_rate": np.linspace(-1e-4, 2e-4, 200),
                        "open_interest": np.full(200, np.nan)},   # OI fetch failed
                       index=pd.date_range("2024-01-01", periods=200, freq="8h")).rename_axis("date")
    cols = features.active_feature_cols("eth", use_orderflow=True, use_futures=True)
    feats = features.build_features(btc, ref_df=eth, cross_prefix="eth", fut_df=fut,
                                    use_orderflow=True, use_futures=True)
    from forge import train
    X, y = train.build_target(feats, 12, cols)
    assert len(X) > 1000 and not X.isna().any().any()
    assert float(X[features.ORDERFLOW_COLS].abs().sum().sum()) == 0.0  # neutral, not NaN


def test_gate_logic():
    cfg = Config.from_env()
    assert registry.gate({"pearson_r": 0.05, "whitelist_passed": 3, "zptae_impr": 0.1}, None, cfg)[0]
    assert not registry.gate({"pearson_r": -0.01, "whitelist_passed": 8}, None, cfg)[0]
    assert registry.gate({"pearson_r": 0.05, "whitelist_passed": 4, "zptae_impr": 0.0},
                         {"whitelist_passed": 3, "zptae_impr": 0.5}, cfg)[0]
    assert not registry.gate({"pearson_r": 0.05, "whitelist_passed": 2, "zptae_impr": 0.9},
                             {"whitelist_passed": 3, "zptae_impr": 0.1}, cfg)[0]


def test_run_once_sign_aware_portable():
    cfg = Config.from_env()
    assert cfg.cross_prefix == "eth"
    assert cfg.use_orderflow and not cfg.use_futures   # futures disabled for the offline test
    rec = pipeline.run_once(cfg, fetcher=_fetcher)
    assert rec["version"] in registry.list_versions(cfg)
    pkl = os.path.join(cfg.version_dir(rec["version"]), "predict.pkl")
    assert os.path.exists(pkl)

    predict = export.load_predict(pkl)
    # 3-arg contract; futures arg may be None (model has futures disabled)
    out = predict(_DATA["BTC/USDT"].tail(500), _DATA["ETH/USDT"].tail(500), None)
    assert isinstance(out, float)
    assert isinstance(predict(_DATA["BTC/USDT"].tail(500), _DATA["ETH/USDT"].tail(500)), float)
    with pytest.raises(ValueError):           # cross model must get the reference asset
        predict(_DATA["BTC/USDT"].tail(500))

    import json
    with open(os.path.join(cfg.version_dir(rec["version"]), "metadata.json")) as f:
        meta = json.load(f)
    assert meta.get("cross_symbol") == "ETH/USDT"
    assert meta.get("use_orderflow") is True
    assert meta.get("n_features") == len(
        features.active_feature_cols("eth", use_orderflow=True))
    # the winner is one of the sign-aware / regressor candidates
    assert meta.get("model") in {"Ridge", "LightGBM", "LGBM-Clf"} \
        or str(meta.get("model", "")).startswith("Blend")


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
