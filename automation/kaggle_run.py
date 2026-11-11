#!/usr/bin/env python3
"""Push version notebooks to Kaggle (CPU, private) and collect results.

  python automation/kaggle_run.py push V0 V1      # push notebooks/v0_kaggle.ipynb ...
  python automation/kaggle_run.py status V0 V1
  python automation/kaggle_run.py collect V0 V1   # download outputs into results/<V>/

Needs KAGGLE_USERNAME / KAGGLE_KEY in the environment (never commit them).
"""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = os.environ["KAGGLE_USERNAME"]
DATASET = "abdallahwagih/mit-bih-arrhythmia-database"  # WFDB files of PhysioNet mitdb 1.0.0


def kaggle(*a):
    r = subprocess.run(["kaggle", *a], capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def slug(v):
    return f"ecg-{v.lower()}"


def push(v):
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(ROOT / "notebooks" / f"{v.lower()}_kaggle.ipynb", Path(d) / f"{slug(v)}.ipynb")
        meta = {"id": f"{USER}/{slug(v)}", "title": slug(v).replace("-", " "),
                "code_file": f"{slug(v)}.ipynb", "language": "python", "kernel_type": "notebook",
                "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
                "dataset_sources": [DATASET], "kernel_sources": [], "competition_sources": [],
                "model_sources": []}
        (Path(d) / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
        return kaggle("kernels", "push", "-p", d)


def collect(v):
    with tempfile.TemporaryDirectory() as d:
        out = kaggle("kernels", "output", f"{USER}/{slug(v)}", "-p", d)
        src = Path(d) / "results" / v
        if not src.exists():
            return f"no results yet: {out[-300:]}"
        dst = ROOT / "results" / v
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
        for log in Path(d).glob("*.log"):
            shutil.copy(log, dst / "kaggle.log")
        return f"saved {dst}"


if __name__ == "__main__":
    cmd, versions = sys.argv[1], sys.argv[2:]
    for v in versions:
        if cmd == "push":
            print(v, push(v))
        elif cmd == "status":
            print(v, kaggle("kernels", "status", f"{USER}/{slug(v)}"))
        elif cmd == "collect":
            print(v, collect(v))
