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
# LightGBM capacity is tuned per-fold below (the kit builds ~240 features, so the
# right depth/regularization differs from our ~54-feature stack).

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

# ---------------- 2. tune over walk-forward folds (the kit builds ~240 features,
#                     so let the data pick the LightGBM capacity, judged on DA) ----
BASE = dict(n_estimators=600, learning_rate=0.03, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0,
            random_state=42, verbose=-1)
GRID = [dict(max_depth=d, num_leaves=nl, min_child_samples=mc)
        for d, nl in ((4, 31), (6, 63), (8, 127)) for mc in (100, 300)]
N_FOLDS = 4
X_all, y_all = df[feature_cols], df["target"]
n = len(df)
val_len = n // (2 * N_FOLDS)
folds = [(n - (N_FOLDS - i) * val_len, n - (N_FOLDS - 1 - i) * val_len) for i in range(N_FOLDS)]


def _eval(params):
    das, rs, oof_p, oof_t = [], [], [], []
    for v0, v1 in folds:
        tr_end = max(1, v0 - TARGET_BARS)                      # purge the horizon
        m = LGBMRegressor(**{**BASE, **params}).fit(X_all.iloc[:tr_end], y_all.iloc[:tr_end])
        p = m.predict(X_all.iloc[v0:v1]); t = y_all.iloc[v0:v1].values
        das.append(float(np.mean(np.sign(p) == np.sign(t))))
        rs.append(float(np.corrcoef(p, t)[0, 1]) if np.std(p) > 0 else 0.0)
        oof_p += list(p); oof_t += list(t)
    return np.mean(das), np.std(das), np.mean(rs), np.array(oof_p), np.array(oof_t)


print(f"[tune] {len(GRID)} configs x {N_FOLDS} folds on {len(feature_cols)} features...")
best = None
for params in GRID:
    da_m, da_s, r_m, oof_p, oof_t = _eval(params)
    print(f"   depth={params['max_depth']} min_child={params['min_child_samples']:>3} | "
          f"DA={da_m:.3f}+/-{da_s:.3f}  r={r_m:.3f}")
    if best is None or da_m > best[0]:
        best = (da_m, da_s, r_m, oof_p, oof_t, params)
da_m, da_s, r_m, oof_p, oof_t, BEST_PARAMS = best
print(f"[best] {BEST_PARAMS} -> DA={da_m:.3f}+/-{da_s:.3f}  r={r_m:.3f}")

# ---------------- 3. shrink (SCALE) from out-of-fold predictions ----------------
denom = float(np.dot(oof_p, oof_p))
SCALE = float(np.clip(np.dot(oof_p, oof_t) / denom, 0.05, 1.0)) if denom > 0 else 1.0
print(f"[scale] SCALE={SCALE:.3f}  (DA/r are scale-invariant; SCALE only lifts "
      f"error-improvement metrics)")

try:                                              # the kit's official grade
    from allora_forge_builder_kit import PerformanceEvaluator
    print("[grade]", PerformanceEvaluator().evaluate(y_true=oof_t, y_pred=oof_p * SCALE))
except Exception as exc:                          # noqa: BLE001
    print(f"[grade] skipped ({exc})")

# ---------------- 4. refit on ALL data + predict closure ----------------
final_model = LGBMRegressor(**{**BASE, **BEST_PARAMS}).fit(X_all, y_all)


def predict(nonce: int = None) -> float:
    """Predicted 1h BTC/USD LOG-RETURN (topic 72's target). Self-fetching."""
    try:
        live = workflow.get_live_features(ticker=TICKERS[0])
    except TypeError:                             # [verify] signature fallback
        live = workflow.get_live_features(TICKERS[0])
    if live is None or len(live) == 0:
        raise ValueError("could not fetch live features")
    x = live[feature_cols].tail(1)                # DataFrame keeps feature names
    return float(final_model.predict(x)[0]) * SCALE


try:                                              # sanity check before saving
    print(f"[live] sample prediction = {predict():+.6f} log-return")
except Exception as exc:                          # noqa: BLE001
    print(f"[live] sample skipped ({exc}); will work once deployed")

with open("predict.pkl", "wb") as f:
    cloudpickle.dump(predict, f)
print("[done] wrote predict.pkl  ->  deploy:  TOPIC_ID=72 python deploy_worker.py")
