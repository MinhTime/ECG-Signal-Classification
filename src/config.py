"""Experiment versions. Each version changes exactly ONE factor vs the version it is based on."""

VERSIONS = {
    # Reproduction of the original repo / paper (Subba & Chingtham, 2024)
    "V0": dict(based_on=None, features=["base"], model="rf", balance="upsample",
               change="baseline: paper features + RF + up-sampling to 20k"),
    "V1": dict(based_on="V0", features=["base", "rr"], model="rf", balance="upsample",
               change="+ RR-interval features (pre, post, local mean, ratio)"),
    "V2": dict(based_on="V1", features=["base", "rr", "morph"], model="rf", balance="upsample",
               change="+ time-domain morphology (beat waveform pooled to 72 points)"),
    "V3": dict(based_on="V2", features=["base", "rr", "morph"], model="lgbm", balance="upsample",
               change="classifier RF -> LightGBM (gradient boosting)"),
    "V4": dict(based_on="V3", features=["base", "rr", "morph"], model="lgbm", balance="smote",
               change="balancing: random up-sampling -> SMOTE"),
    "V5": dict(based_on="V4", features=["base", "rr", "morph", "rr_norm"], model="lgbm",
               balance="smote", change="+ patient-normalised RR (RR / record mean RR)"),
}
