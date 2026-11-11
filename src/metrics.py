"""Per-class and overall metrics (one-vs-rest from the confusion matrix)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def evaluate(y_true, y_pred, class_names):
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    per = {}
    for i, c in enumerate(class_names):
        tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp; tn = total - tp - fn - fp
        se = tp / (tp + fn) if tp + fn else 0.0
        sp = tn / (tn + fp) if tn + fp else 0.0
        ppv = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * se * ppv / (se + ppv) if se + ppv else 0.0
        per[c] = {"support": int(cm[i].sum()), "sensitivity": float(se), "specificity": float(sp),
                  "ppv": float(ppv), "f1": float(f1)}
    present = [c for c in class_names if per[c]["support"] > 0]
    return {
        "accuracy": float(np.trace(cm) / total),
        "macro_f1": float(np.mean([per[c]["f1"] for c in present])),
        "macro_sensitivity": float(np.mean([per[c]["sensitivity"] for c in present])),
        "macro_ppv": float(np.mean([per[c]["ppv"] for c in present])),
        "per_class": per,
        "confusion": cm.tolist(),
    }


def to_text(m, class_names) -> str:
    lines = [f"accuracy {m['accuracy']:.4f} | macro-F1 {m['macro_f1']:.4f} | "
             f"macro-Se {m['macro_sensitivity']:.4f} | macro-PPV {m['macro_ppv']:.4f}", "",
             f"{'class':<6}{'n':>7}{'Se':>8}{'Sp':>8}{'PPV':>8}{'F1':>8}"]
    for c in class_names:
        p = m["per_class"][c]
        lines.append(f"{c:<6}{p['support']:>7}{p['sensitivity']:>8.4f}{p['specificity']:>8.4f}"
                     f"{p['ppv']:>8.4f}{p['f1']:>8.4f}")
    lines += ["", "confusion (rows=true, cols=pred): " + " ".join(class_names)]
    lines += [f"{c:<6}" + "".join(f"{v:>7}" for v in row) for c, row in zip(class_names, m["confusion"])]
    return "\n".join(lines)
