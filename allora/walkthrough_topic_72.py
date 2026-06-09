"""
walkthrough_topic_72.py -- train + export a predict.pkl for Allora Forge TOPIC 72
(1h BTC/USD log-return prediction, updated every 5 minutes) with the Allora Forge
Builder Kit.

HOW TO USE
----------
1) Clone + install the Builder Kit and activate its venv:
     git clone https://github.com/allora-network/allora-forge-builder-kit.git
     cd allora-forge-builder-kit
     python3.11 -m venv .venv && source .venv/bin/activate
     pip install . && pip install -r requirements.txt
2) (optional) Free data key from developer.allora.network:
     export ALLORA_API_KEY="UP-...your-key..."
   No key? This script automatically falls back to data_source="binance".
3) Copy THIS file into the kit folder and run it there:
     cp /path/to/allora-forge/allora/walkthrough_topic_72.py .
     python walkthrough_topic_72.py
   -> writes predict.pkl in the current directory.
4) Deploy + monitor:
     TOPIC_ID=72 python deploy_worker.py
     python -m allora_forge_builder_kit.web_dashboard      # dashboard on :8787

WHAT IT DOES
------------
- Builds the kit's feature/target frame for BTC at 5-minute bars with a 12-bar
  (= 1 hour) ahead target -- exactly topic 72's horizon.
- Trains a *regularized* LightGBM (the settings our research validated for this
  task: shallow trees, strong min_child, low learning rate).
- Picks a least-squares-optimal shrink factor SCALE. The kit's grader rewards
  error-improvement-over-zero and has NO log-aspect constraint, so shrinking helps;
  directional accuracy and Pearson r are unchanged by positive scaling.
- Saves a self-contained predict(nonce)->float that returns the 1h LOG-RETURN
  (topic 72's target -- NOT a price).

Lines marked [verify] use the kit's API; adjust them if your kit version differs.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import cloudpickle
import numpy as np
import pandas as pd  # noqa: F401
from lightgbm import LGBMRegressor

from allora_forge_builder_kit import AlloraMLWorkflow  # [verify] import path

# ---------------- config for TOPIC 72 ----------------
TICKERS = ["btcusd"]
INTERVAL = "5m"
NUMBER_OF_INPUT_BARS = 48
TARGET_BARS = 12          # 12 * 5m = 1 hour ahead  <-- topic-72 horizon (topic_77 used 1)
DAYS = 365                # history to train on
VAL_FRACTION = 0.2

# Regularized LightGBM (validated by walk-forward ablation/tuning on this task).
LGBM_PARAMS = dict(n_estimators=600, learning_rate=0.02, max_depth=3,
                   num_leaves=31, min_child_samples=400, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.8, reg_alpha=0.5,
                   reg_lambda=1.0, random_state=42, verbose=-1)

api_key = os.environ.get("ALLORA_API_KEY", "").strip()
DATA_SOURCE = "allora" if api_key else "binance"
print(f"[setup] data_source={DATA_SOURCE}  (set ALLORA_API_KEY to use Allora data)")

# ---------------- 1. data + features from the kit ----------------
workflow = AlloraMLWorkflow(
    tickers=TICKERS,
    number_of_input_bars=NUMBER_OF_INPUT_BARS,
    target_bars=TARGET_BARS,
    interval=INTERVAL,
    data_source=DATA_SOURCE,
    api_key=api_key or None,
)

start_date = datetime.now(timezone.utc) - timedelta(days=DAYS)
try:                              # some kit versions backfill explicitly first
    workflow.backfill(days=DAYS)
except TypeError:
    try:
        workflow.backfill()
    except Exception:
        pass
except Exception:
    pass

try:
    df = workflow.get_full_feature_target_dataframe(start_date=start_date)
except TypeError:                 # [verify] older/newer signature
    df = workflow.get_full_feature_target_dataframe()

feature_cols = [c for c in df.columns if c.startswith("feature_")]
assert feature_cols, "no feature_* columns found -- check the kit's dataframe schema"
assert "target" in df.columns, "no 'target' column -- check the kit's dataframe schema"
df = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
print(f"[data] {len(df)} rows, {len(feature_cols)} features")

# ---------------- 2. chronological split (purged) + train ----------------
cut = int(len(df) * (1 - VAL_FRACTION))
purge = TARGET_BARS               # don't let train targets overlap the holdout window
X_tr, y_tr = df[feature_cols].iloc[:max(1, cut - purge)], df["target"].iloc[:max(1, cut - purge)]
X_va, y_va = df[feature_cols].iloc[cut:], df["target"].iloc[cut:]
model = LGBMRegressor(**LGBM_PARAMS).fit(X_tr, y_tr)

# ---------------- 3. holdout metrics + least-squares shrink ----------------
pred_va = model.predict(X_va)
yv = y_va.values
da = float(np.mean(np.sign(pred_va) == np.sign(yv)))
r = float(np.corrcoef(pred_va, yv)[0, 1]) if np.std(pred_va) > 0 else 0.0
denom = float(np.dot(pred_va, pred_va))          # scale minimizing RMSE vs truth
SCALE = float(np.clip(np.dot(pred_va, yv) / denom, 0.05, 1.0)) if denom > 0 else 1.0
print(f"[holdout] DA={da:.3f}  Pearson r={r:.3f}  SCALE={SCALE:.3f}  "
      f"(DA/r are scale-invariant; SCALE only lifts error-improvement metrics)")

try:                                              # informational: the kit's grade
    from allora_forge_builder_kit import PerformanceEvaluator
    print("[grade]", PerformanceEvaluator(workflow).evaluate(y_true=yv, y_pred=pred_va * SCALE))
except Exception as exc:                          # noqa: BLE001
    print(f"[grade] skipped ({exc})")

# ---------------- 4. refit on ALL data + predict closure ----------------
final_model = LGBMRegressor(**LGBM_PARAMS).fit(df[feature_cols], df["target"])


def predict(nonce: int = None) -> float:
    """Predicted 1h BTC/USD LOG-RETURN (topic 72's target). Self-fetching."""
    try:
        live = workflow.get_live_features(ticker=TICKERS[0])
    except TypeError:                             # [verify] signature fallback
        live = workflow.get_live_features(TICKERS[0])
    if live is None or len(live) == 0:
        raise ValueError("could not fetch live features")
    x = live[feature_cols].tail(1).values
    return float(final_model.predict(x)[0]) * SCALE


try:                                              # sanity check before saving
    print(f"[live] sample prediction = {predict():+.6f} log-return")
except Exception as exc:                          # noqa: BLE001
    print(f"[live] sample skipped ({exc}); will work once deployed")

with open("predict.pkl", "wb") as f:
    cloudpickle.dump(predict, f)
print("[done] wrote predict.pkl  ->  deploy:  TOPIC_ID=72 python deploy_worker.py")
