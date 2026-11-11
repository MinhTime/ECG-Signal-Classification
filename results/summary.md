**random split**

| Version | Change | Acc | Macro-F1 | SVEB Se | FB Se | VEB Se |
|---|---|---|---|---|---|---|
| V0 | baseline: paper features + RF + up-sampling to 20k | 97.57 | 87.60 | 69.96 | 55.62 | 90.53 |
| V1 | + RR-interval features (pre, post, local mean, ratio) | 98.47 | 91.23 | 80.22 | 64.38 | 94.61 |
| V2 | + time-domain morphology (beat waveform pooled to 72 points) | 99.01 | 93.10 | 85.25 | 68.12 | 96.68 |
| V3 | classifier RF -> LightGBM (gradient boosting) | 99.36 | 94.85 | 93.35 | 75.00 | 98.41 |
| V4 | balancing: random up-sampling -> SMOTE | 99.42 | 95.42 | 95.14 | 78.75 | 98.41 |
| V5 | + patient-normalised RR (RR / record mean RR) | 99.39 | 95.31 | 94.42 | 78.75 | 98.48 |

**interpatient split**

| Version | Change | Acc | Macro-F1 | SVEB Se | FB Se | VEB Se |
|---|---|---|---|---|---|---|
| V0 | baseline: paper features + RF + up-sampling to 20k | 90.88 | 33.75 | 0.22 | 0.00 | 84.50 |
| V1 | + RR-interval features (pre, post, local mean, ratio) | 94.44 | 38.35 | 1.63 | 0.00 | 92.17 |
| V2 | + time-domain morphology (beat waveform pooled to 72 points) | 92.66 | 36.23 | 2.29 | 0.00 | 91.30 |
| V3 | classifier RF -> LightGBM (gradient boosting) | 90.43 | 42.00 | 25.64 | 5.15 | 96.52 |
| V4 | balancing: random up-sampling -> SMOTE | 90.70 | 46.27 | 42.79 | 14.69 | 95.96 |
| V5 | + patient-normalised RR (RR / record mean RR) | 92.29 | 50.15 | 61.89 | 14.43 | 95.84 |
