# CT dose model report — CCTA

Trained 2026-09-22T17:21:00 on 50 exams (48 unique).

## Data audit

- kVp levels present: 120, 140
- Exact duplicate rows, 1-based row numbers (kept together in CV): [28, 29, 42, 43]
- CTDIvol/mAs at 120 kV: median 0.0605 mGy/mAs (range 0.0560–0.0607, n=46)
- CTDIvol/mAs at 140 kV: median 0.0894 mGy/mAs (range 0.0893–0.0894, n=4)
- Rows with a different dose-per-mAs (≈-7%, likely different collimation/pitch/scan mode), 1-based row numbers: [2, 9, 22, 41, 45]
- DLP/CTDIvol (implied irradiated length): {'mean': 23.24, 'std': 2.19, 'min': 18.82, 'max': 28.95} cm; correlation with the 'scan length' field = -0.228, correlation of 'scan length' with BMI = 0.895
  - ⚠ The 'scan length / SCAN SIZE' field does not behave like the irradiated z-length (DLP/CTDIvol does not scale with it and it tracks BMI). It may be the patient size / FOV. Please verify.

## Physics law fitted to your scanner

- k (dose per mAs at 120 kV): 0.0605
- kV exponent n: 2.533

## CTDIvol: repeated grouped 5-fold CV (mean ± sd over repeats)

| model | MAPE % | MAE | RMSE | R² | max APE % |
|---|---|---|---|---|---|
| Physics law (mAs·kV^n) **(selected)** | 0.92 ± 0.03 | 0.31 | 0.79 | 0.9981 | 8.1 |
| Physics + BMI term | 0.93 ± 0.03 | 0.32 | 0.79 | 0.9981 | 8.1 |
| Physics + GP residual | 1.32 ± 0.10 | 0.46 | 0.80 | 0.9980 | 8.2 |
| Physics + NN residual | 1.38 ± 0.02 | 0.56 | 0.81 | 0.9980 | 7.5 |
| Log-log ridge (poly2) | 2.07 ± 0.81 | 1.21 | 2.98 | 0.9603 | 15.9 |
| Deep MLP ensemble | 2.26 ± 0.75 | 1.19 | 2.85 | 0.9642 | 15.7 |
| Monotone gradient boosting | 7.25 ± 0.52 | 4.03 | 7.29 | 0.8379 | 32.7 |

90% conformal prediction interval: prediction × [0.927, 1.003]

## DLP: repeated grouped 5-fold CV (mean ± sd over repeats)

| model | MAPE % | MAE | RMSE | R² | max APE % |
|---|---|---|---|---|---|
| Physics law (CTDI × const length) **(selected)** | 8.17 ± 0.17 | 89.53 | 117.91 | 0.9225 | 23.5 |
| Physics + length | 8.55 ± 0.38 | 95.86 | 122.41 | 0.9164 | 23.6 |
| Physics + GP residual | 9.63 ± 0.80 | 115.34 | 161.22 | 0.8535 | 29.3 |
| Physics + NN residual | 9.66 ± 0.68 | 117.08 | 163.91 | 0.8488 | 29.0 |
| Physics + length + BMI | 9.70 ± 0.67 | 117.41 | 163.81 | 0.8490 | 28.9 |
| Log-log ridge (poly2) | 10.07 ± 1.71 | 125.31 | 206.95 | 0.7341 | 45.8 |
| Physics + GBM residual | 10.44 ± 0.98 | 120.83 | 162.97 | 0.8500 | 31.9 |
| Deep MLP ensemble | 12.39 ± 1.20 | 158.40 | 264.34 | 0.5861 | 56.8 |
| Monotone gradient boosting | 14.35 ± 0.75 | 168.43 | 233.67 | 0.6945 | 44.2 |

90% conformal prediction interval: prediction × [0.820, 1.248]