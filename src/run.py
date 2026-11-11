"""Run one or more versions:  python -m src.run --versions V0 V1 --split random --out results"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np

from . import dataset as ds, features, metrics, models, preprocess as pp
from .config import VERSIONS

log = logging.getLogger("run")


def prep(sig):
    return pp.preprocess_signal(sig, level=9, cutoff_low=1, cutoff_high=1, median_widths=(71, 215))


def run_version(name, beats, feats_cache, tr, te, split, out_root, seed=42):
    cfg = VERSIONS[name]
    for g in cfg["features"]:
        if g not in feats_cache:
            feats_cache[g] = features.GROUPS[g](beats.X, rr=beats.rr, rec=beats.rec)
    F = np.hstack([feats_cache[g] for g in cfg["features"]])
    Ftr, ytr = F[tr], beats.y[tr]
    if cfg["balance"] == "upsample":
        Ftr, ytr = ds.upsample(Ftr, ytr, seed=seed)
    elif cfg["balance"] == "smote":
        Ftr, ytr = ds.smote(Ftr, ytr, seed=seed)
    t0 = time.time()
    model = models.build(cfg["model"], seed=seed)
    model.fit(Ftr, ytr)
    pred = model.predict(F[te])
    m = metrics.evaluate(beats.y[te], pred, list(ds.AAMI_CLASSES))
    m.update(version=name, split=split, n_features=int(F.shape[1]), n_train=int(len(ytr)),
             n_test=int(len(te)), train_seconds=round(time.time() - t0, 1), config=cfg)
    out = os.path.join(out_root, name, split)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    txt = f"{name} [{split}] {cfg['change']}\n" + metrics.to_text(m, list(ds.AAMI_CLASSES))
    with open(os.path.join(out, "report.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt + "\n", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+", default=list(VERSIONS))
    ap.add_argument("--split", nargs="+", default=["random"], choices=["random", "interpatient"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    beats = ds.load_beats(ds.find_data_dir(a.data_dir), preprocessor=prep)
    cache = {}
    rows = []
    for split in a.split:
        tr, te = ds.split_indices(beats, split, seed=a.seed)
        for v in a.versions:
            m = run_version(v, beats, cache, tr, te, split, a.out, seed=a.seed)
            pc = m["per_class"]
            rows.append([v, split, m["accuracy"], m["macro_f1"], pc["SVEB"]["sensitivity"],
                         pc["SVEB"]["f1"], pc["VEB"]["f1"], pc["FB"]["sensitivity"], pc["FB"]["f1"]])
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "summary_" + "_".join(a.split) + ".csv")
    with open(path, "w") as f:
        f.write("version,split,accuracy,macro_f1,sveb_se,sveb_f1,veb_f1,fb_se,fb_f1\n")
        for r in rows:
            f.write(",".join([r[0], r[1]] + [f"{x:.4f}" for x in r[2:]]) + "\n")
    print(open(path).read())


if __name__ == "__main__":
    main()
