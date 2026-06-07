"""Central configuration for the Forge worker.

Tuned for the Forge competition: **1-hour-ahead log-return**, polled every 5
minutes, scored with ZPTAE + the whitelist metric bundle. Defaults model on
5-minute candles with a 12-bar (=1h) horizon. Single asset (BTC) for now; the
code is symbol-parametrized so adding ETH is a config change.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # --- market / task ---
    exchange: str = "binanceus"
    symbol: str = "BTC/USDT"          # competition asset (BTC/USD); USDT pair is the liquid proxy
    timeframe: str = "5m"             # base candle resolution (matches 5-min poll cadence)
    horizon_steps: int = 12           # bars ahead = 1 hour at 5m (12 * 5m)

    # --- training ---
    train_window_days: int = 120      # rolling window of 5m bars (~34.5k rows)
    val_fraction: float = 0.2         # chronological holdout
    min_train_rows: int = 2000
    random_state: int = 42
    ridge_alphas: tuple = (0.1, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
    lgbm_params: dict = field(default_factory=lambda: dict(
        n_estimators=2000, learning_rate=0.03, num_leaves=31, max_depth=4,
        min_child_samples=200, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0, verbose=-1,
    ))

    # --- variance calibration (fixes the log-aspect-ratio criterion) ---
    # Scale predictions so std(pred) ~= ratio * std(true). 1.0 = full match
    # (log-aspect ~0, most "informative"); lower (toward ~0.35) trades log-aspect
    # margin for better WRMSE/WZPTAE-vs-zero. Must keep |log10(ratio)| < 0.5.
    calibration_target_ratio: float = 1.0

    # --- scoring (competition) ---
    zptae_power: float = 1.5          # power-tanh exponent (surrogate of Allora ZPTAE)
    eval_nonoverlap: bool = True      # evaluate whitelist metrics on non-overlapping 1h windows
    gate_tolerance: float = 0.0       # promote if new primary >= current - tolerance

    # --- data fetching ---
    fetch_page_limit: int = 1000
    recent_candles: int = 300         # candles pulled for a single live inference (~25h at 5m)

    # --- paths ---
    data_dir: str = "data"
    models_dir: str = "models"

    # --- scheduler ---
    retrain_hour_utc: int = 1         # retrain daily; inferences every 5 min use the latest model

    # --- serving / allora ---
    topic_token: str = "BTC"
    server_host: str = "0.0.0.0"
    server_port: int = 8000

    # --- ops ---
    alert_webhook: str = ""

    # ----- derived -----
    @property
    def data_file(self) -> str:
        # one store per symbol+timeframe, e.g. ohlcv_BTCUSDT_5m.csv
        sym = self.symbol.replace("/", "")
        return f"ohlcv_{sym}_{self.timeframe}.csv"

    @property
    def data_path(self) -> str:
        return os.path.join(self.data_dir, self.data_file)

    @property
    def current_dir(self) -> str:
        return os.path.join(self.models_dir, "current")

    @property
    def current_predict(self) -> str:
        return os.path.join(self.current_dir, "predict.pkl")

    @property
    def current_metadata(self) -> str:
        return os.path.join(self.current_dir, "metadata.json")

    @property
    def metrics_path(self) -> str:
        return os.path.join(self.models_dir, "metrics.jsonl")

    @property
    def predictions_path(self) -> str:
        return os.path.join(self.models_dir, "predictions.jsonl")

    @property
    def horizon_minutes(self) -> int:
        import pandas as pd
        per_bar = pd.Timedelta(self.timeframe).total_seconds() / 60.0
        return int(round(per_bar * self.horizon_steps))

    def version_dir(self, version: str) -> str:
        return os.path.join(self.models_dir, version)

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.models_dir, self.current_dir):
            os.makedirs(d, exist_ok=True)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_env(cls) -> "Config":
        c = cls()
        c.exchange = os.environ.get("ALLORA_EXCHANGE", c.exchange)
        c.symbol = os.environ.get("ALLORA_SYMBOL", c.symbol)
        c.timeframe = os.environ.get("ALLORA_TIMEFRAME", c.timeframe)
        c.horizon_steps = int(os.environ.get("ALLORA_HORIZON_STEPS", c.horizon_steps))
        c.train_window_days = int(os.environ.get("ALLORA_TRAIN_WINDOW_DAYS", c.train_window_days))
        c.calibration_target_ratio = float(
            os.environ.get("ALLORA_CALIBRATION_RATIO", c.calibration_target_ratio))
        c.data_dir = os.environ.get("ALLORA_DATA_DIR", c.data_dir)
        c.models_dir = os.environ.get("ALLORA_MODELS_DIR", c.models_dir)
        c.retrain_hour_utc = int(os.environ.get("ALLORA_RETRAIN_HOUR_UTC", c.retrain_hour_utc))
        c.topic_token = os.environ.get("ALLORA_TOPIC_TOKEN", c.topic_token)
        c.server_port = int(os.environ.get("ALLORA_SERVER_PORT", c.server_port))
        c.alert_webhook = os.environ.get("ALLORA_ALERT_WEBHOOK", c.alert_webhook)
        return c
