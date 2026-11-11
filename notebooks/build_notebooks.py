"""Build one self-contained Kaggle notebook per version: python notebooks/build_notebooks.py"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ["__init__.py", "config.py", "dataset.py", "preprocess.py", "feature_extraction.py",
       "features.py", "models.py", "metrics.py", "run.py"]

WHY = {
    "V0": """**V0 - Baseline (reproduction).** Original repo / paper pipeline (Subba & Chingtham, IEEE Access 2024):
bior4.4 wavelet denoising + median baseline removal, 190 features (R-peak stats, QRS proxy, skewness,
kurtosis, FFT magnitude), Random Forest (100 trees), minority classes up-sampled to 20,000.
Paper reports 98.0% accuracy. Every later version changes exactly one factor.""",
    "V1": """**V1 = V0 + RR-interval features (one change).**
- *Prior result (V0):* SVEB sensitivity only 0.70; 164 of 556 SVEB beats predicted as N.
- *Theory:* an SVEB is a *premature* beat whose QRS shape is almost normal. The difference is timing
  (short RR before, long pause after), which a single-beat FFT cannot see.
- *Paper:* de Chazal et al., IEEE TBME 2004; Mondejar-Guerra et al., BSPC 2019 - RR intervals are the
  most discriminative features for SVEB.""",
    "V2": """**V2 = V1 + time-domain morphology (one change).**
- *Prior result (V1):* FB sensitivity 0.64 (47 FB -> N), 68 VEB -> N.
- *Theory:* FFT magnitude throws away phase, so it loses *where* and *how wide* the QRS/T waves are.
  Fusion and ventricular beats differ from N exactly in QRS width/shape. We add the beat waveform
  itself, mean-pooled 360 -> 72 points.
- *Paper:* de Chazal et al. 2004 (morphology samples); Mondejar-Guerra et al. 2019 (raw + wavelet morphology).""",
    "V3": """**V3 = V2 with classifier RF -> LightGBM (one change).**
- *Prior result (V2):* 75 SVEB -> N and FB sensitivity 0.68 remain; features now carry the information
  (each feature group helped), so the bottleneck is the classifier.
- *Theory:* RF averages independent deep trees; gradient boosting adds trees that fix the previous
  errors, which helps hard, rare cases and usually wins on tabular features.
- *Paper:* Ke et al., LightGBM, NeurIPS 2017; Grinsztajn et al., NeurIPS 2022 (tree boosting on tabular data).""",
    "V4": """**V4 = V3 with balancing up-sampling -> SMOTE (one change).**
- *Prior result (V3):* FB is the weakest class (sensitivity 0.75). FB has only 642 training beats,
  copied ~31x to reach 20,000, so the model sees the same points again and again (overfits them).
- *Theory:* SMOTE creates new points between minority neighbours instead of copies, widening the
  minority decision region.
- *Paper:* Chawla et al., JAIR 2002 (SMOTE); proposed in Sec. VIII of our report.""",
    "V5": """**V5 = V4 + patient-normalised RR features (one change).**
- *Prior result (V4):* 99.4% accuracy on the random (intra-patient) split, but on the inter-patient
  split (train DS1, test DS2 - unseen patients) SVEB sensitivity is only 0.43.
- *Theory:* absolute RR depends on each person's heart rate; "premature" is *relative* to the
  person's normal rhythm. Dividing RR by the record's mean RR removes the heart-rate offset.
- *Paper:* Lin & Yang, Math. Probl. Eng. 2014 (normalized RR); Mondejar-Guerra et al. 2019.""",
}


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c.update(outputs=[], execution_count=None)
    return c


def build(v):
    cells = [cell("markdown", f"# ECG classification - {v}\n\n" + WHY[v])]
    cells.append(cell("code", "!pip -q install wfdb peakutils PyWavelets imbalanced-learn lightgbm"))
    for f in SRC:
        code = (ROOT / "src" / f).read_text()
        cells.append(cell("code", f"%%writefile src/{f}\n{code}" if f != "__init__.py"
                          else "import os; os.makedirs('src', exist_ok=True)\n"
                               f"open('src/__init__.py','w').write({code!r})"))
    cells.append(cell("code", f"!python -m src.run --versions {v} --split random interpatient --out results"))
    cells.append(cell("code", "import glob\nfor p in sorted(glob.glob('results/**/report.txt', recursive=True)):\n"
                              "    print(open(p).read())"))
    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
          "language": "python"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out = ROOT / "notebooks" / f"{v.lower()}_kaggle.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    return out


if __name__ == "__main__":
    for v in WHY:
        print(build(v))
