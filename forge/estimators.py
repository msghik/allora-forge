"""Composite predictors that turn base learners into sign-aware return models.

Each wraps a probability classifier and/or a regressor behind the uniform
``.predict(X) -> np.ndarray`` interface the pipeline and exporter expect, so they
slot in as ordinary candidates and travel inside ``predict.pkl`` (registered for
pickle-by-value at export). The competition metric is directional-accuracy heavy,
so a classifier that minimizes log-loss on the *sign* often beats an MSE
regressor on DA; the blend keeps magnitude information for Pearson r.

All outputs are on an arbitrary scale -- the pipeline's variance-calibration step
sets the final magnitude (the log-aspect-ratio criterion), and DA / Pearson r are
invariant to positive scaling.
"""
from __future__ import annotations

import numpy as np


class SignMagnitudePredictor:
    """Classifier -> signed return: ``(2*P(up) - 1) * magnitude``.

    Targets directional accuracy directly (the classifier minimizes log-loss on
    the up/down label). ``magnitude`` is cosmetic -- calibration rescales anyway.
    """

    def __init__(self, clf, magnitude: float = 1.0):
        self.clf = clf
        self.magnitude = float(magnitude)

    def predict(self, X):
        p = self.clf.predict_proba(X)[:, 1]
        return (2.0 * p - 1.0) * self.magnitude


class BlendPredictor:
    """Convex blend of a regressor and a classifier-driven signed predictor.

    Each leg is standardized to unit std (measured on the fit set) so ``weight``
    is a meaningful mix: the regressor contributes magnitude (helps Pearson r) and
    the classifier contributes calibrated sign (helps directional accuracy).
    """

    def __init__(self, reg, clf, weight: float = 0.5,
                 reg_std: float = 1.0, sgn_std: float = 1.0):
        self.reg = reg
        self.clf = clf
        self.weight = float(weight)
        self.reg_std = float(reg_std) or 1.0
        self.sgn_std = float(sgn_std) or 1.0

    def predict(self, X):
        r = np.asarray(self.reg.predict(X), dtype=float) / self.reg_std
        p = self.clf.predict_proba(X)[:, 1]
        s = (2.0 * p - 1.0) / self.sgn_std
        return self.weight * r + (1.0 - self.weight) * s
