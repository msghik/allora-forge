"""Model training: target construction, chronological split, and the two
regularized candidate models (Ridge + LightGBM)."""
from __future__ import annotations

import numpy as np

from .features import FEATURE_COLS


def build_target(df_features, horizon: int):
    r"""target_t = ln(Close_{t+H} / Close_t); drop the H trailing NaN rows."""
    data = df_features.copy()
    data["target"] = np.log(data["close"].shift(-horizon) / data["close"])
    data = data.dropna(subset=["target"])
    return data[FEATURE_COLS].copy(), data["target"].copy()


def chrono_split(X, y, val_fraction: float):
    cut = int(len(X) * (1.0 - val_fraction))
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


# ----- Ridge -----
def make_ridge(alpha: float):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def train_ridge(X_tr, y_tr, config):
    """Fit Ridge with alpha chosen by TimeSeriesSplit CV on the train set."""
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    pipe = make_ridge(alpha=1.0)
    n_splits = max(2, min(5, len(X_tr) // 100))
    gs = GridSearchCV(
        pipe, {"ridge__alpha": list(config.ridge_alphas)},
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="neg_mean_squared_error", n_jobs=-1,
    )
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_, {"alpha": float(gs.best_params_["ridge__alpha"])}


# ----- LightGBM -----
def make_lgbm(config, **overrides):
    from lightgbm import LGBMRegressor
    params = {**config.lgbm_params, "random_state": config.random_state,
              "n_jobs": -1, **overrides}
    return LGBMRegressor(**params)


def train_lgbm(X_tr, y_tr, config):
    """Fit LightGBM with early stopping on an inner tail of the train set, then
    refit on the full train set with the chosen number of trees."""
    from lightgbm import early_stopping, log_evaluation
    inner_cut = int(len(X_tr) * 0.85)
    X_in, X_es = X_tr.iloc[:inner_cut], X_tr.iloc[inner_cut:]
    y_in, y_es = y_tr.iloc[:inner_cut], y_tr.iloc[inner_cut:]

    probe = make_lgbm(config)
    probe.fit(X_in, y_in, eval_set=[(X_es, y_es)], eval_metric="l2",
              callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
    best_iter = int(probe.best_iteration_ or config.lgbm_params["n_estimators"])

    model = make_lgbm(config, n_estimators=best_iter)
    model.fit(X_tr, y_tr)
    return model, {"n_estimators": best_iter}


def fit_final(name: str, params: dict, X, y, config):
    """Refit the winning model on the full (windowed) dataset for deployment."""
    if name == "Ridge":
        model = make_ridge(alpha=params["alpha"])
    else:
        model = make_lgbm(config, n_estimators=params["n_estimators"])
    model.fit(X, y)
    return model


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
