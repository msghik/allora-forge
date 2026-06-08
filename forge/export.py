"""Package a trained model into a self-contained ``predict.pkl``.

The exported callable runs the EXACT same feature pipeline, takes the latest
engineered row, and returns one float (the 1h log-return prediction). When the
model uses cross-asset features it accepts a reference DataFrame too:
``predict(df, ref_df)``.

We register :mod:`forge.features` for pickle-by-value so all feature code travels
inside predict.pkl; it reloads with plain ``pickle.load`` given numpy/pandas +
the model's library.
"""
from __future__ import annotations

import pickle

from . import features
from .features import build_features


def make_predict(model, feature_cols, scale: float = 1.0, cross_prefix=None):
    """Build the single callable the worker node / Forge expects."""

    def predict(df, ref_df=None):
        import pandas as pd  # noqa: F401 -- self-sufficient at inference time

        def _dtindex(x):
            if isinstance(x.index, pd.DatetimeIndex):
                return x
            for col in ("date", "timestamp"):
                if col in x.columns:
                    unit = "ms" if col == "timestamp" else None
                    return x.set_index(pd.to_datetime(x[col], unit=unit))
            return x

        d = _dtindex(df.copy())
        r = _dtindex(ref_df.copy()) if ref_df is not None else None
        if cross_prefix and r is None:
            raise ValueError("this model needs a reference asset; call predict(df, ref_df)")

        feats = build_features(d, ref_df=r, cross_prefix=cross_prefix)
        if len(feats) == 0:
            raise ValueError("Not enough candle history to compute features "
                             "(need ~300+ candles).")
        return float(model.predict(feats[feature_cols].iloc[[-1]])[0] * scale)

    return predict


def export_predict(model, feature_cols, out_path: str, scale: float = 1.0,
                   cross_prefix=None) -> str:
    """Cloudpickle the predict callable to ``out_path`` (feature code by value)."""
    import cloudpickle

    cloudpickle.register_pickle_by_value(features)
    try:
        predict = make_predict(model, feature_cols, scale=scale, cross_prefix=cross_prefix)
        with open(out_path, "wb") as f:
            cloudpickle.dump(predict, f)
    finally:
        cloudpickle.unregister_pickle_by_value(features)
    return out_path


def load_predict(path: str):
    """Load a predict.pkl with the standard library (proves portability)."""
    with open(path, "rb") as f:
        return pickle.load(f)
