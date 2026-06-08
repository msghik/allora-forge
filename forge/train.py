"""Model training: target construction, purged chronological split, recency
weighting, and the candidate learners.

Candidates are intentionally diverse so the pipeline can pick whatever scores
best on the (directional-accuracy-heavy) whitelist:
  * **Ridge** / **LightGBM** regressors  -- magnitude (good for Pearson r),
  * a **LightGBM classifier** wrapped as a signed predictor -- optimizes the sign
    directly (good for directional accuracy),
  * **blends** of the two.
"""
from __future__ import annotations

import numpy as np

from .features import FEATURE_COLS

EPS = 1e-12


def build_target(df_features, horizon: int, feature_cols=None):
    r"""target_t = ln(Close_{t+H} / Close_t); drop the H trailing NaN rows."""
    feature_cols = feature_cols or FEATURE_COLS
    data = df_features.copy()
    data["target"] = np.log(data["close"].shift(-horizon) / data["close"])
    data = data.dropna(subset=["target"])
    return data[feature_cols].copy(), data["target"].copy()


def chrono_split(X, y, val_fraction: float, purge: int = 0):
    """Chronological split with a purge gap so the last train targets (which look
    ``horizon`` bars ahead) don't overlap the validation window."""
    cut = int(len(X) * (1.0 - val_fraction))
    tr_end = max(0, cut - max(0, purge))
    return X.iloc[:tr_end], X.iloc[cut:], y.iloc[:tr_end], y.iloc[cut:]


def recency_weights(index, half_life_days: float):
    """Exponential-decay sample weights (newest = 1.0). None disables weighting."""
    if not half_life_days or half_life_days <= 0:
        return None
    age_days = np.asarray((index.max() - index).total_seconds(), dtype=float) / 86400.0
    return np.power(0.5, age_days / float(half_life_days))


def _combine_weights(base, magnitude):
    """Multiply recency weights by a normalized |target| emphasis (decisive moves)."""
    mag = np.abs(np.asarray(magnitude, dtype=float))
    mag = mag / (mag.mean() + EPS)
    return mag if base is None else base * mag


# ----- Ridge -----
def make_ridge(alpha: float):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def fit_ridge(alpha: float, X, y, sample_weight=None):
    m = make_ridge(alpha)
    if sample_weight is not None:
        m.fit(X, y, ridge__sample_weight=sample_weight)
    else:
        m.fit(X, y)
    return m


def train_ridge(X_tr, y_tr, config, sample_weight=None):
    """Pick alpha by TimeSeriesSplit CV, then refit (optionally weighted)."""
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    n_splits = max(2, min(5, len(X_tr) // 100))
    gs = GridSearchCV(
        make_ridge(alpha=1.0), {"ridge__alpha": list(config.ridge_alphas)},
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="neg_mean_squared_error", n_jobs=-1,
    )
    gs.fit(X_tr, y_tr)
    alpha = float(gs.best_params_["ridge__alpha"])
    return fit_ridge(alpha, X_tr, y_tr, sample_weight), {"alpha": alpha}


# ----- LightGBM regressor -----
def make_lgbm(config, **overrides):
    from lightgbm import LGBMRegressor
    params = {**config.lgbm_params, "random_state": config.random_state,
              "n_jobs": -1, **overrides}
    return LGBMRegressor(**params)


def _best_iter(probe, fallback):
    return int(probe.best_iteration_ or fallback)


def train_lgbm(X_tr, y_tr, config, sample_weight=None):
    """Early-stop on an inner tail, then refit on the full train set (weighted)."""
    from lightgbm import early_stopping, log_evaluation
    inner_cut = int(len(X_tr) * 0.85)
    X_in, X_es = X_tr.iloc[:inner_cut], X_tr.iloc[inner_cut:]
    y_in, y_es = y_tr.iloc[:inner_cut], y_tr.iloc[inner_cut:]

    probe = make_lgbm(config)
    probe.fit(X_in, y_in, eval_set=[(X_es, y_es)], eval_metric="l2",
              callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
    n = _best_iter(probe, config.lgbm_params["n_estimators"])
    model = make_lgbm(config, n_estimators=n)
    model.fit(X_tr, y_tr, sample_weight=sample_weight)
    return model, {"n_estimators": n}


# ----- LightGBM classifier (directional) -----
def make_lgbm_classifier(config, **overrides):
    from lightgbm import LGBMClassifier
    params = {**config.lgbm_params, "random_state": config.random_state,
              "n_jobs": -1, **overrides}
    params.pop("objective", None)
    return LGBMClassifier(**params)


def train_lgbm_classifier(X_tr, y_tr, config, sample_weight=None):
    """Binary up/down classifier, early-stopped on log-loss then refit (weighted)."""
    from lightgbm import early_stopping, log_evaluation
    label = (y_tr.values > 0).astype(int)
    inner_cut = int(len(X_tr) * 0.85)
    X_in, X_es = X_tr.iloc[:inner_cut], X_tr.iloc[inner_cut:]
    y_in, y_es = label[:inner_cut], label[inner_cut:]

    probe = make_lgbm_classifier(config)
    probe.fit(X_in, y_in, eval_set=[(X_es, y_es)], eval_metric="binary_logloss",
              callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
    n = _best_iter(probe, config.lgbm_params["n_estimators"])
    model = make_lgbm_classifier(config, n_estimators=n)
    model.fit(X_tr, label, sample_weight=sample_weight)
    return model, {"n_estimators": n}


def fit_final(name: str, params: dict, X, y, config, sample_weight=None):
    """Refit a base learner on the full (windowed) dataset for deployment."""
    if name == "Ridge":
        return fit_ridge(params["alpha"], X, y, sample_weight)
    if name == "LightGBM":
        m = make_lgbm(config, n_estimators=params["n_estimators"])
        m.fit(X, y, sample_weight=sample_weight)
        return m
    raise ValueError(f"unknown model {name}")


def calibration_scale(y_true, y_pred, target_ratio: float) -> float:
    """Factor s such that std(s * y_pred) ~= target_ratio * std(y_true).

    Counters model shrinkage so predictions have realistic magnitude (the
    log-aspect-ratio whitelist criterion). Pearson r and directional accuracy are
    invariant to this positive scaling; WRMSE/ZPTAE and log-aspect are not.
    """
    sp = float(np.std(np.asarray(y_pred, dtype=float)))
    st = float(np.std(np.asarray(y_true, dtype=float)))
    if sp <= 0 or st <= 0:
        return 1.0
    return float(target_ratio * st / sp)
