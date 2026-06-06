"""Out-of-sample evaluation metrics for the 24h log-return forecast."""
from __future__ import annotations

import warnings

import numpy as np


def evaluate(name, y_true, y_pred) -> dict:
    """Pearson r, directional accuracy, MAE and RMSE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from scipy.stats import pearsonr
        if np.std(y_pred) == 0 or np.std(y_true) == 0:
            r, p = 0.0, 1.0
        else:
            r, p = pearsonr(y_true, y_pred)
    da = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    return {
        "model": name,
        "pearson_r": float(r),
        "p_value": float(p),
        "directional_acc": da,
        "mae": mae,
        "rmse": rmse,
        "n": int(len(y_true)),
    }


def baseline_directional_acc(y_true) -> float:
    """Directional accuracy of an always-up baseline (share of positive moves)."""
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean(y_true > 0))
