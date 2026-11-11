"""Paper figures from results/<V>/<split>/metrics.json:  python paper/figures/make_figures.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
V = ["V0", "V1", "V2", "V3", "V4", "V5"]
CLS = ["N", "SVEB", "VEB", "FB", "Q"]
plt.rcParams.update({"font.size": 8, "font.family": "serif"})


def m(v, s):
    return json.loads((ROOT / "results" / v / s / "metrics.json").read_text())


# Fig. progression: macro-F1, SVEB Se, FB Se per version for both splits
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
for ax, s, title in zip(axes, ["random", "interpatient"], ["(a) Random beat split (paper protocol)",
                                                            "(b) Inter-patient split (DS1 → DS2)"]):
    x = np.arange(len(V))
    series = [("Macro-F1", [m(v, s)["macro_f1"] for v in V], "o-"),
              ("SVEB Se", [m(v, s)["per_class"]["SVEB"]["sensitivity"] for v in V], "s--"),
              ("FB Se", [m(v, s)["per_class"]["FB"]["sensitivity"] for v in V], "^:")]
    for label, y, st in series:
        ax.plot(x, np.array(y) * 100, st, label=label, lw=1.2, ms=4)
    ax.set_xticks(x); ax.set_xticklabels(V); ax.set_title(title, fontsize=8)
    ax.set_ylim(0, 100); ax.grid(alpha=0.3)
axes[0].set_ylabel("%"); axes[0].legend(loc="lower right", fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "progression.pdf")

# Fig. confusion matrices V0 vs V4 (random split), row-normalised
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
for ax, v in zip(axes, ["V0", "V4"]):
    cm = np.array(m(v, "random")["confusion"], dtype=float)
    norm = cm / cm.sum(1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xticklabels(CLS); ax.set_yticklabels(CLS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{v} ({'baseline' if v == 'V0' else 'best, random split'})", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "confusion.pdf")
print("ok")
