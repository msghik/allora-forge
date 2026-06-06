"""Package a trained model into a self-contained ``predict.pkl``.

The exported callable runs the EXACT same ``add_features`` pipeline, takes the
latest engineered row, and returns one float (the 24h log-return prediction).

We register :mod:`forge.features` for pickle-by-value so the feature code travels
*inside* predict.pkl. The artifact then reloads with plain ``pickle.load`` in any
environment that has numpy/pandas + the model's library (scikit-learn or
lightgbm) -- no need for the ``forge`` package to be importable there.
"""
from __future__ import annotations

import pickle

from . import features
from .features import add_features


def make_predict(model, feature_cols):
    """Build the single callable the Forge / worker node expects."""

    def predict(df):
        import pandas as pd  # noqa: F401 -- self-sufficient at inference time

        d = df.copy()
        # Ensure a DatetimeIndex (add_features uses index.hour).
        if not isinstance(d.index, pd.DatetimeIndex):
            for col in ("date", "timestamp"):
                if col in d.columns:
                    unit = "ms" if col == "timestamp" else None
                    d = d.set_index(pd.to_datetime(d[col], unit=unit))
                    break
        feats = add_features(d)
        if len(feats) == 0:
            raise ValueError("Not enough candle history to compute features "
                             "(need ~50+ candles).")
        return float(model.predict(feats[feature_cols].iloc[[-1]])[0])

    return predict


def export_predict(model, feature_cols, out_path: str) -> str:
    """Cloudpickle the predict callable to ``out_path`` (feature code by value)."""
    import cloudpickle

    cloudpickle.register_pickle_by_value(features)
    try:
        predict = make_predict(model, feature_cols)
        with open(out_path, "wb") as f:
            cloudpickle.dump(predict, f)
    finally:
        cloudpickle.unregister_pickle_by_value(features)
    return out_path


def load_predict(path: str):
    """Load a predict.pkl with the standard library (proves portability)."""
    with open(path, "rb") as f:
        return pickle.load(f)
