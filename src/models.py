"""Classifiers."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


def build(name: str, seed: int = 42):
    if name == "rf":  # paper hyper-parameters
        return RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1, max_features="sqrt",
                                      random_state=seed, n_jobs=-1)
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              random_state=seed, n_jobs=-1, verbose=-1)
    raise ValueError(name)
