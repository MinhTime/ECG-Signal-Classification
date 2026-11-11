"""Create paper/ECG_Paper_v2.docx = original report + new Section IX (improvements V0-V5).

python paper/make_paper_v2.py <original ECG_Paper-1.docx>
"""
import copy
import json
import sys
from pathlib import Path

import docx
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
V = ["V0", "V1", "V2", "V3", "V4", "V5"]


def m(v, split):
    return json.loads((ROOT / "results" / v / split / "metrics.json").read_text())


def se(x, c):
    return x["per_class"][c]["sensitivity"] * 100


R = {v: m(v, "random") for v in V}
SHORT = {"V0": "Baseline (paper)", "V1": "+ RR intervals", "V2": "+ morphology",
         "V3": "RF → LightGBM", "V4": "Up-sampling → SMOTE", "V5": "+ normalised RR"}
I = {v: m(v, "interpatient") for v in V}

d = docx.Document(sys.argv[1])
P = d.paragraphs
anchor = next(p for p in P if p.text.startswith("IX. Conclusion"))
h1 = anchor.style
h2 = next(p for p in P if p.text.startswith("A. Data Loading")).style
normal = next(p for p in P if p.text.startswith("Electrocardiography")).style
cap = next(p for p in P if p.text.startswith("TABLE III")).style
tmpl_tbl = d.tables[0]


TEMPL = {
    "h1": next(p for p in P if p.text.startswith("VIII.")),
    "h2": next(p for p in P if p.text.startswith("A. Data Loading")),
    "normal": next(p for p in P if p.text.startswith("Electrocardiography")),
    "cap": next(p for p in P if p.text.startswith("TABLE III")),
}


def para(text, kind="normal", bold_prefix=None):
    """Clone a paragraph of the original so the new text keeps its exact formatting."""
    src = TEMPL[kind]
    new = copy.deepcopy(src._p)
    anchor._p.addprevious(new)
    p = docx.text.paragraph.Paragraph(new, src._parent)
    runs = p.runs
    for r in runs[1:]:
        r._r.getparent().remove(r._r)
    runs[0].text = text
    if bold_prefix:
        b = copy.deepcopy(runs[0]._r)
        runs[0]._r.addprevious(b)
        br = docx.text.run.Run(b, p)
        br.text = bold_prefix
        br.bold = True
    return p


def table(header, rows):
    t = d.add_table(rows=1, cols=len(header))
    t.style = tmpl_tbl.style
    for c, h in zip(t.rows[0].cells, header):
        c.text = h
        for r in c.paragraphs[0].runs:
            r.bold = True
    for row in rows:
        cells = t.add_row().cells
        for c, val in zip(cells, row):
            c.text = val
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
    anchor._p.addprevious(t._tbl)


r0, r4, r5 = R["V0"], R["V4"], R["V5"]
para("IX. Improving the Pipeline: Controlled Experiments (V0–V5)", "h1")
para("Starting from the reimplemented pipeline, we ran a sequence of controlled experiments. Each version "
     "changes exactly one factor relative to the previous one, and each change is motivated by (i) the "
     "errors of the previous version, (ii) a signal-processing or learning argument, and (iii) prior "
     "literature. All runs use the same code (src/), the same seed (42) and were executed as Kaggle CPU "
     "notebooks (one notebook per version). Because accuracy is dominated by the N class (83% of beats), "
     "we use macro-F1 and the sensitivity (Se) of the minority classes SVEB and FB as the main metrics.")
para("A. Versions", "h2")
why = [
    ("V0 (baseline). ", f"The pipeline of Sections IV–V, run end to end. Measured accuracy is "
     f"{r0['accuracy']*100:.2f}% (close to the 98% target), but SVEB Se is only {se(r0,'SVEB'):.1f}% and "
     f"FB Se {se(r0,'FB'):.1f}%."),
    ("V1: + RR-interval features. ", "An SVEB is a premature beat whose QRS shape is almost normal; the "
     "difference lies in timing (a short RR before the beat and a compensatory pause after). The 360-sample "
     "FFT of a single beat cannot represent this. We add pre-RR, post-RR, the local mean of the previous 10 "
     "RR intervals and the pre/local ratio, following de Chazal et al. [10] and Mondéjar-Guerra et al. [11]."),
    ("V2: + time-domain morphology. ", "FFT magnitude discards phase, so it loses where and how wide the QRS "
     "and T waves are, which is exactly how fusion and ventricular beats differ from N. We add the beat "
     "waveform mean-pooled from 360 to 72 points [10], [11]."),
    ("V3: Random Forest → LightGBM. ", "With informative features in place, the remaining SVEB/FB errors "
     "point to the classifier. Gradient boosting adds trees that correct previous errors, which helps rare, "
     "hard cases and generally outperforms bagging on tabular features [12], [13]."),
    ("V4: random up-sampling → SMOTE. ", "FB has only 642 training beats; copying them ~31 times makes the "
     "model memorise the same points. SMOTE [14] interpolates new minority samples instead, as proposed in "
     "Section VIII."),
    ("V5: + patient-normalised RR. ", "Absolute RR depends on each person's heart rate; prematurity is "
     "relative to that person's rhythm. Dividing pre-, post- and local RR by the record's mean RR removes "
     "the heart-rate offset [11], [15]. This targets generalisation to unseen patients (Section IX-C)."),
]
for b, t in why:
    para(t, bold_prefix=b)

para("B. Results on the Paper Protocol (random 80/20 beat split)", "h2")
para("TABLE V. Results on the random (intra-patient) split, 21,888 test beats. Values in %.", "cap")
rows = []
for v in V:
    x = R[v]
    rows.append([v, SHORT[v], f"{x['accuracy']*100:.2f}",
                 f"{x['macro_f1']*100:.2f}", f"{se(x,'SVEB'):.2f}", f"{se(x,'VEB'):.2f}", f"{se(x,'FB'):.2f}"])
table(["Ver.", "Change", "Acc", "Macro-F1", "SVEB Se", "VEB Se", "FB Se"], rows)
best = max(V, key=lambda v: R[v]["macro_f1"])
para(f"Every one-factor change from V0 to V4 improved the macro-F1, from {r0['macro_f1']*100:.2f}% to "
     f"{r4['macro_f1']*100:.2f}%, and the accuracy from {r0['accuracy']*100:.2f}% to {r4['accuracy']*100:.2f}%, "
     f"above the 98% reported in [1]. The largest single gains came from RR features (SVEB Se "
     f"+{se(R['V1'],'SVEB')-se(r0,'SVEB'):.1f} points) and from LightGBM (SVEB Se "
     f"+{se(R['V3'],'SVEB')-se(R['V2'],'SVEB'):.1f} points). SMOTE mainly helped FB "
     f"({se(R['V3'],'FB'):.1f}% → {se(r4,'FB'):.1f}%). V5 is on par with V4 on this split "
     f"(macro-F1 {r5['macro_f1']*100:.2f}%), which is expected: in a random beat split the same patients "
     f"appear in training and test, so patient normalisation adds little. The best model on this split is {best}.")

para("C. Honest Check: Inter-patient Split (DS1 → DS2)", "h2")
para("The random beat split used in [1] places beats of the same patient, and even overlapping windows of "
     "neighbouring beats, in both training and test sets. This is known to inflate results [10], [16]. We "
     "therefore also evaluate every version with the standard inter-patient split of de Chazal et al. [10]: "
     "train on 22 records (DS1), test on 22 different patients (DS2); paced records are excluded.")
para("TABLE VI. Results on the inter-patient split (DS2, unseen patients). Values in %.", "cap")
rows = []
for v in V:
    x = I[v]
    rows.append([v, f"{x['accuracy']*100:.2f}", f"{x['macro_f1']*100:.2f}", f"{se(x,'SVEB'):.2f}",
                 f"{x['per_class']['SVEB']['ppv']*100:.2f}", f"{se(x,'VEB'):.2f}", f"{se(x,'FB'):.2f}"])
table(["Ver.", "Acc", "Macro-F1", "SVEB Se", "SVEB PPV", "VEB Se", "FB Se"], rows)
i0, i5 = I["V0"], I["V5"]
para(f"On unseen patients the original pipeline almost never detects SVEB (Se {se(i0,'SVEB'):.1f}%) or FB "
     f"(Se {se(i0,'FB'):.1f}%), although its accuracy still looks high ({i0['accuracy']*100:.1f}%) because "
     f"most beats are N. The improvements carry over: macro-F1 rises from {i0['macro_f1']*100:.1f}% (V0) to "
     f"{i5['macro_f1']*100:.1f}% (V5), and SVEB Se from {se(i0,'SVEB'):.1f}% to {se(i5,'SVEB'):.1f}%. "
     f"Patient-normalised RR (V5) is the change that matters most here (SVEB Se {se(I['V4'],'SVEB'):.1f}% → "
     f"{se(i5,'SVEB'):.1f}%), while it was neutral on the random split. One change did not transfer: the raw "
     f"morphology of V2 lowered inter-patient macro-F1 ({I['V1']['macro_f1']*100:.1f}% → {I['V2']['macro_f1']*100:.1f}%), "
     "because waveform shape is partly patient-specific. FB remains hard for all versions, and the Q class has "
     "only 7 beats in DS2, which pulls macro-F1 down. "
     "We therefore recommend reporting the inter-patient protocol alongside the random split in future work.")

# renumber conclusion and extend it
anchor.text = "X. Conclusion and Future Work"
concl = P[P.index(anchor) + 1] if anchor in P else None
for p in d.paragraphs:
    if p.text.startswith("This paper presented a modular"):
        p.add_run(f" Building on this baseline, five controlled one-factor changes (RR intervals, time-domain "
                  f"morphology, LightGBM, SMOTE and patient-normalised RR) raised the accuracy on the paper's "
                  f"protocol from {r0['accuracy']*100:.2f}% to {r4['accuracy']*100:.2f}% and the macro-F1 from "
                  f"{r0['macro_f1']*100:.2f}% to {r4['macro_f1']*100:.2f}%. On the stricter inter-patient split, "
                  f"macro-F1 improved from {i0['macro_f1']*100:.1f}% to {i5['macro_f1']*100:.1f}%, showing that "
                  "the gains are not only due to patient overlap, while also exposing how optimistic the random "
                  "split is.")
    if p.text.startswith("Cardiovascular disease remains"):
        p.add_run(f" Beyond reproduction, we run five controlled one-factor improvements; the best model "
                  f"(RR + morphology features, LightGBM, SMOTE) reaches {r4['accuracy']*100:.2f}% accuracy and "
                  f"{r4['macro_f1']*100:.2f}% macro-F1 (baseline {r0['accuracy']*100:.2f}% / "
                  f"{r0['macro_f1']*100:.2f}%), and we show with an inter-patient evaluation that the random "
                  "beat split strongly overestimates performance on new patients.")

refs = [
    "[10] P. de Chazal, M. O'Dwyer, and R. B. Reilly, “Automatic classification of heartbeats using ECG "
    "morphology and heartbeat interval features,” IEEE Trans. Biomed. Eng., vol. 51, no. 7, pp. 1196–1206, 2004.",
    "[11] V. Mondéjar-Guerra, J. Novo, J. Rouco, M. G. Penedo, and M. Ortega, “Heartbeat classification "
    "fusing temporal and morphological information of ECGs via ensemble of classifiers,” Biomed. Signal "
    "Process. Control, vol. 47, pp. 41–48, 2019.",
    "[12] G. Ke et al., “LightGBM: A highly efficient gradient boosting decision tree,” in Proc. NeurIPS, 2017.",
    "[13] L. Grinsztajn, E. Oyallon, and G. Varoquaux, “Why do tree-based models still outperform deep "
    "learning on typical tabular data?,” in Proc. NeurIPS Datasets and Benchmarks, 2022.",
    "[14] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, “SMOTE: Synthetic minority "
    "over-sampling technique,” J. Artif. Intell. Res., vol. 16, pp. 321–357, 2002.",
    "[15] C.-C. Lin and C.-M. Yang, “Heartbeat classification using normalized RR intervals and "
    "morphological features,” Math. Probl. Eng., vol. 2014, Art. no. 712474, 2014.",
    "[16] E. J. da S. Luz, W. R. Schwartz, G. Cámara-Chávez, and D. Menotti, “ECG-based heartbeat "
    "classification for arrhythmia detection: A survey,” Comput. Methods Programs Biomed., vol. 127, "
    "pp. 144–164, 2016.",
]
last = [p for p in d.paragraphs if p.text.startswith("[9]")][0]
for ref in reversed(refs):
    new = copy.deepcopy(last._p)
    last._p.addnext(new)
    np_ = docx.text.paragraph.Paragraph(new, last._parent)
    for r in np_.runs[1:]:
        r._r.getparent().remove(r._r)
    np_.runs[0].text = ref

out = ROOT / "paper" / "ECG_Paper_v2.docx"
d.save(out)
print(out)
