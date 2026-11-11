"""Load MIT-BIH beats, AAMI labels, RR intervals, and train/test splits."""
from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

log = logging.getLogger(__name__)

AAMI_CLASSES = ("N", "SVEB", "VEB", "FB", "Q")
SYMBOL_TO_AAMI = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "SVEB", "a": "SVEB", "J": "SVEB", "S": "SVEB",
    "V": "VEB", "E": "VEB",
    "F": "FB",
    "/": "Q", "f": "Q", "Q": "Q",
}
CLASS_TO_INT = {c: i for i, c in enumerate(AAMI_CLASSES)}
FS = 360

# de Chazal et al. (2004) inter-patient split. Paced records 102, 104, 107, 217 are excluded.
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207,
       208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221,
       222, 228, 231, 232, 233, 234]


@dataclass
class Beats:
    X: np.ndarray    # (n, window) beat segments
    y: np.ndarray    # (n,) int labels
    rr: np.ndarray   # (n, 4) RR features in seconds: pre, post, local mean, pre/local
    rec: np.ndarray  # (n,) record number


def find_data_dir(hint: str | None = None) -> str:
    """Return a folder holding the WFDB records (searches /kaggle/input if no hint)."""
    if hint and glob.glob(os.path.join(hint, "*.hea")):
        return hint
    roots = [hint] if hint else ["/kaggle/input", "./data", "."]
    for root in roots:
        hits = glob.glob(os.path.join(root, "**", "100.hea"), recursive=True)
        if hits:
            return os.path.dirname(hits[0])
    raise FileNotFoundError("MIT-BIH records (100.hea ...) not found")


def _rr_features(r_pos: np.ndarray) -> np.ndarray:
    """RR features for each beat from the R-peak sample positions of one record."""
    rr = np.diff(r_pos) / FS
    pre = np.concatenate([[rr[0]], rr])
    post = np.concatenate([rr, [rr[-1]]])
    # local mean of the previous 10 pre-RR intervals (causal, includes the current one)
    csum = np.cumsum(np.concatenate([[0.0], pre]))
    idx = np.arange(len(pre))
    lo = np.maximum(0, idx - 9)
    local = (csum[idx + 1] - csum[lo]) / (idx + 1 - lo)
    return np.stack([pre, post, local, pre / local], axis=1)


def load_beats(data_dir: str, window: int = 360, preprocessor=None) -> Beats:
    import wfdb

    names = sorted(os.path.splitext(f)[0] for f in os.listdir(data_dir)
                   if f.endswith(".hea") and f[:3].isdigit())
    half = window // 2
    Xs, ys, rrs, recs = [], [], [], []
    for name in names:
        path = os.path.join(data_dir, name)
        rec = wfdb.rdrecord(path)
        ann = wfdb.rdann(path, "atr")
        sig = rec.p_signal[:, 0].astype(np.float64)
        if preprocessor is not None:
            sig = preprocessor(sig)
        keep = [(s, SYMBOL_TO_AAMI[c]) for s, c in zip(ann.sample, ann.symbol) if c in SYMBOL_TO_AAMI]
        pos = np.array([s for s, _ in keep])
        rr = _rr_features(pos)
        for (s, cls), r in zip(keep, rr):
            a, b = s - half, s - half + window
            if a < 0 or b > len(sig):
                continue
            Xs.append(sig[a:b]); ys.append(CLASS_TO_INT[cls]); rrs.append(r); recs.append(int(name))
    beats = Beats(np.asarray(Xs), np.asarray(ys), np.asarray(rrs), np.asarray(recs))
    log.info("Loaded %d beats from %d records", len(beats.y), len(names))
    return beats


def split_indices(beats: Beats, mode: str = "random", seed: int = 42, test_size: float = 0.2):
    """'random' = paper protocol (stratified 80/20 over beats). 'interpatient' = DS1/DS2."""
    idx = np.arange(len(beats.y))
    if mode == "random":
        return train_test_split(idx, test_size=test_size, random_state=seed, stratify=beats.y)
    if mode == "interpatient":
        return idx[np.isin(beats.rec, DS1)], idx[np.isin(beats.rec, DS2)]
    raise ValueError(mode)


def upsample(F: np.ndarray, y: np.ndarray, target: int = 20_000, seed: int = 42):
    """Paper balancing: each minority class up-sampled with replacement to `target`."""
    classes, counts = np.unique(y, return_counts=True)
    major = classes[np.argmax(counts)]
    Fp, yp = [], []
    for c in classes:
        Fc = F[y == c]
        if c != major and len(Fc) < target:
            Fc = resample(Fc, replace=True, n_samples=target, random_state=seed)
        Fp.append(Fc); yp.append(np.full(len(Fc), c))
    F2, y2 = np.vstack(Fp), np.concatenate(yp)
    perm = np.random.default_rng(seed).permutation(len(y2))
    return F2[perm], y2[perm]


def smote(F: np.ndarray, y: np.ndarray, target: int = 20_000, seed: int = 42):
    """SMOTE: synthesise minority samples (k=5 neighbours) up to `target` per minority class."""
    from imblearn.over_sampling import SMOTE

    classes, counts = np.unique(y, return_counts=True)
    major = classes[np.argmax(counts)]
    strategy = {int(c): max(int(n), target) for c, n in zip(classes, counts) if c != major}
    return SMOTE(sampling_strategy=strategy, k_neighbors=5, random_state=seed).fit_resample(F, y)
