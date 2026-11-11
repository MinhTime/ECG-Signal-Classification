"""Summary table (results/summary.md, results/summary.csv) and figure (figures/results.png)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ["V0", "V1", "V2", "V3", "V4", "V5"]
COLS = [("accuracy", "Acc"), ("macro_f1", "Macro-F1"), ("SVEB", "SVEB Se"), ("FB", "FB Se"), ("VEB", "VEB Se")]


def load(v, split):
    p = ROOT / "results" / v / split / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def val(m, key):
    return m[key] if key in m else m["per_class"][key]["sensitivity"]


rows_md, rows_csv = [], ["version,split,change,accuracy,macro_f1,sveb_se,fb_se,veb_se"]
for split in ["random", "interpatient"]:
    rows_md += [f"\n**{split} split**\n", "| Version | Change | " + " | ".join(c for _, c in COLS) + " |",
                "|---|---|" + "---|" * len(COLS)]
    for v in VERSIONS:
        m = load(v, split)
        if not m:
            continue
        vals = [val(m, k) for k, _ in COLS]
        rows_md.append(f"| {v} | {m['config']['change']} | " + " | ".join(f"{x*100:.2f}" for x in vals) + " |")
        rows_csv.append(f"{v},{split},\"{m['config']['change']}\"," + ",".join(f"{x:.4f}" for x in vals))
(ROOT / "results" / "summary.md").write_text("\n".join(rows_md).strip() + "\n")
(ROOT / "results" / "summary.csv").write_text("\n".join(rows_csv) + "\n")
print("\n".join(rows_md))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, split in zip(axes, ["random", "interpatient"]):
    ms = [(v, load(v, split)) for v in VERSIONS if load(v, split)]
    x = range(len(ms))
    for i, (key, label) in enumerate([("macro_f1", "Macro-F1"), ("SVEB", "SVEB Se"), ("FB", "FB Se")]):
        ax.bar([j + (i - 1) * 0.27 for j in x], [val(m, key) for _, m in ms], width=0.27, label=label)
    ax.set_xticks(list(x)); ax.set_xticklabels([v for v, _ in ms]); ax.set_title(f"{split} split")
    ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
axes[0].legend(loc="lower right")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "results.png", dpi=150)
