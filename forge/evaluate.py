"""Competition-aligned evaluation for 1h log-return prediction.

Reports the Forge whitelist bundle: Pearson r (+p), directional accuracy (+95%
CI and p-value), log-aspect ratio, WRMSE and a ZPTAE *surrogate*, plus their
improvement over predicting zero. Metrics are computed on **non-overlapping 1h
windows** (how Allora measures ground truth), even though training uses the
denser, overlapping 5-min-spaced samples.

NOTE: Allora's exact ZPTAE is not public; `zptae` here is a documented surrogate
(z-normalized power-tanh absolute error) used for model *selection* only. Pearson,
DA and log-aspect ratio are exact.
"""
from __future__ import annotations

import warnings

import numpy as np


def nonoverlap(y_true, y_pred, step: int):
    """Subsample to non-overlapping windows (every `step`-th sample)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if step and step > 1:
        return y_true[::step], y_pred[::step]
    return y_true, y_pred


def _pearson(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0, 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from scipy.stats import pearsonr
        r, p = pearsonr(y_true, y_pred)
    return float(r), float(p)


def _directional(y_true, y_pred):
    """Directional accuracy with a Wilson 95% CI and a one-sided binomial p-value
    against a 0.5 coin flip."""
    correct = (np.sign(y_pred) == np.sign(y_true))
    n = int(len(correct))
    k = int(correct.sum())
    da = k / n if n else 0.0
    # Wilson score interval (95%).
    z = 1.959963984540054
    if n:
        denom = 1 + z**2 / n
        centre = (da + z**2 / (2 * n)) / denom
        half = (z * np.sqrt(da * (1 - da) / n + z**2 / (4 * n**2))) / denom
        lo, hi = centre - half, centre + half
    else:
        lo = hi = 0.0
    try:
        from scipy.stats import binomtest
        p = float(binomtest(k, n, 0.5, alternative="greater").pvalue) if n else 1.0
    except Exception:  # very old scipy
        from scipy.stats import binom_test
        p = float(binom_test(k, n, 0.5, alternative="greater")) if n else 1.0
    return da, float(lo), float(hi), p


def _power_tanh(z, power):
    """Smooth power-tanh: ~|z|^power near 0, power-law tail for large |z|."""
    a = np.abs(z)
    return np.tanh(a) * np.power(a, power - 1.0)


def _zptae(err, ref_std, power):
    z = err / ref_std if ref_std > 0 else err
    return float(np.mean(_power_tanh(z, power)))


def competition_metrics(y_true, y_pred, step: int = 1, power: float = 1.5,
                        ref_std: float | None = None) -> dict:
    """All whitelist metrics, computed on non-overlapping windows."""
    yt, yp = nonoverlap(y_true, y_pred, step)
    n = int(len(yt))
    if ref_std is None:
        ref_std = float(np.std(yt)) or 1.0

    r, r_p = _pearson(yt, yp)
    da, da_lo, da_hi, da_p = _directional(yt, yp)
    std_true = float(np.std(yt))
    std_pred = float(np.std(yp))
    log_aspect = float(np.log10(std_pred / std_true)) if std_true > 0 and std_pred > 0 else float("-inf")

    err = yp - yt
    rmse = float(np.sqrt(np.mean(err**2)))
    rmse_zero = float(np.sqrt(np.mean(yt**2)))
    # weighted (z-normalized) versions
    wrmse = float(np.sqrt(np.mean((err / ref_std) ** 2)))
    wrmse_zero = float(np.sqrt(np.mean((yt / ref_std) ** 2)))
    zptae = _zptae(err, ref_std, power)
    zptae_zero = _zptae(-yt, ref_std, power)

    def impr(model, base):
        return float(1.0 - model / base) if base > 0 else 0.0

    return {
        "n": n,
        "pearson_r": r, "pearson_p": r_p,
        "directional_acc": da, "da_ci_low": da_lo, "da_ci_high": da_hi, "da_p": da_p,
        "log_aspect_ratio": log_aspect,
        "std_pred": std_pred, "std_true": std_true,
        "rmse": rmse, "wrmse": wrmse, "wrmse_impr": impr(wrmse, wrmse_zero),
        "zptae": zptae, "zptae_impr": impr(zptae, zptae_zero),
    }


# Whitelist thresholds (for a quick pass/fail readout; "a clear majority" is the goal).
WHITELIST = {
    "directional_acc": (">", 0.55),
    "da_ci_low": (">", 0.52),
    "da_p": ("<", 0.05),
    "pearson_r": (">", 0.05),
    "pearson_p": ("<", 0.05),
    "wrmse_impr": (">", 0.10),
    "zptae_impr": (">", 0.20),
    "log_aspect_ratio": ("abs<", 0.5),
}


def whitelist_report(m: dict) -> dict:
    """Map each whitelist criterion to pass/fail for the given metrics."""
    out = {}
    for key, (op, thr) in WHITELIST.items():
        v = m.get(key)
        if v is None:
            out[key] = None
        elif op == ">":
            out[key] = bool(v > thr)
        elif op == "<":
            out[key] = bool(v < thr)
        elif op == "abs<":
            out[key] = bool(abs(v) < thr)
    out["passed"] = sum(1 for x in out.values() if x is True)
    out["total"] = len(WHITELIST)
    return out
