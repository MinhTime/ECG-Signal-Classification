"""Feature groups. 'base' = original paper features; others are added one per version."""
from __future__ import annotations

import numpy as np

from . import feature_extraction as fe


def base(X: np.ndarray, **_) -> np.ndarray:
    """Paper features: R-peak stats, QRS proxy, skewness, kurtosis, FFT magnitude (190 dims)."""
    return fe.extract_features_batch(X)


def rr(X: np.ndarray, rr: np.ndarray, **_) -> np.ndarray:
    """RR-interval features: pre-RR, post-RR, local mean RR, pre/local ratio (4 dims)."""
    return rr


def morph(X: np.ndarray, pool: int = 5, **_) -> np.ndarray:
    """Time-domain morphology: the beat waveform mean-pooled by `pool` (360/5 = 72 dims)."""
    n, w = X.shape
    return X[:, : w - w % pool].reshape(n, -1, pool).mean(axis=2)


def rr_norm(X: np.ndarray, rr: np.ndarray, rec: np.ndarray, **_) -> np.ndarray:
    """Patient-normalised RR: pre, post, local RR divided by the record's mean RR (3 dims)."""
    out = np.empty((len(rr), 3))
    for r in np.unique(rec):
        m = rec == r
        out[m] = rr[m, :3] / rr[m, 0].mean()
    return out


GROUPS = {"base": base, "rr": rr, "morph": morph, "rr_norm": rr_norm}
