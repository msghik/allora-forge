"""Central configuration for the Forge worker.

All knobs live here with sensible defaults; a subset can be overridden via
environment variables (handy for Docker / GitHub Actions). Keeping config in one
typed object makes the daily pipeline and the inference server reproducible.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # --- market / task ---
    exchange: str = "binanceus"
    symbol: str = "BTC/USDT"          # Topic 69 = BTC/USD; USDT pair is the liquid proxy
    timeframe: str = "1h"
    horizon_hours: int = 24           # Topic 69 is a 1-day forecast

    # --- training ---
    train_window_days: int = 365      # rolling window -> adapts to regime shifts
    val_fraction: float = 0.2         # chronological holdout
    min_train_rows: int = 500         # refuse to train on too little data
    random_state: int = 42
    ridge_alphas: tuple = (0.1, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
    lgbm_params: dict = field(default_factory=lambda: dict(
        n_estimators=2000, learning_rate=0.03, num_leaves=31, max_depth=4,
        min_child_samples=80, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0, verbose=-1,
    ))

    # --- promotion gate ---
    baseline_min_da: float = 0.50     # candidate must beat a coin-flip on direction
    gate_tolerance: float = 0.0       # promote if new_r >= prod_r - tolerance

    # --- data fetching ---
    fetch_page_limit: int = 1000      # ccxt candles per page
    recent_candles: int = 300         # candles pulled for a single live inference

    # --- paths ---
    data_dir: str = "data"
    data_file: str = "ohlcv.csv"
    models_dir: str = "models"

    # --- scheduler ---
    retrain_hour_utc: int = 1         # daily retrain time (after the 00:00 UTC close)

    # --- serving / allora ---
    topic_token: str = "BTC"          # token in GET /inference/<token>
    server_host: str = "0.0.0.0"
    server_port: int = 8000

    # --- ops ---
    alert_webhook: str = ""           # optional: POST alerts here

    # ----- derived paths -----
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

    def version_dir(self, version: str) -> str:
        return os.path.join(self.models_dir, version)

    def ensure_dirs(self) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.current_dir, exist_ok=True)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config, overriding selected fields from env vars."""
        c = cls()
        c.exchange = os.environ.get("ALLORA_EXCHANGE", c.exchange)
        c.symbol = os.environ.get("ALLORA_SYMBOL", c.symbol)
        c.timeframe = os.environ.get("ALLORA_TIMEFRAME", c.timeframe)
        c.horizon_hours = int(os.environ.get("ALLORA_HORIZON_HOURS", c.horizon_hours))
        c.train_window_days = int(os.environ.get("ALLORA_TRAIN_WINDOW_DAYS", c.train_window_days))
        c.data_dir = os.environ.get("ALLORA_DATA_DIR", c.data_dir)
        c.models_dir = os.environ.get("ALLORA_MODELS_DIR", c.models_dir)
        c.retrain_hour_utc = int(os.environ.get("ALLORA_RETRAIN_HOUR_UTC", c.retrain_hour_utc))
        c.topic_token = os.environ.get("ALLORA_TOPIC_TOKEN", c.topic_token)
        c.server_port = int(os.environ.get("ALLORA_SERVER_PORT", c.server_port))
        c.alert_webhook = os.environ.get("ALLORA_ALERT_WEBHOOK", c.alert_webhook)
        return c
