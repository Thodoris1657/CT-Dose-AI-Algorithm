# CT dose model report — CCTA

Trained 2026-11-05T02:22:15 on 87 exams (85 unique).

## Data audit

- kVp levels present: 120, 140
- Exact duplicate rows, 1-based row numbers (kept together in CV): [28, 29, 42, 43]
- CTDIvol/mAs at 120 kV: median 0.0605 mGy/mAs (range 0.0559–0.0607, n=80)
- CTDIvol/mAs at 140 kV: median 0.0894 mGy/mAs (range 0.0891–0.0894, n=7)
- Rows with a different dose-per-mAs (≈-7%, likely different collimation/pitch/scan mode), 1-based row numbers: [2, 9, 22, 41, 45, 57, 87]
- DLP/CTDIvol (implied irradiated length): {'mean': 22.7, 'std': 2.55, 'min': 17.55, 'max': 28.95} cm; correlation with the 'scan length' field = -0.09, correlation of 'scan length' with BMI = 0.885
  - ⚠ The 'scan length / SCAN SIZE' field does not behave like the irradiated z-length (DLP/CTDIvol does not scale with it and it tracks BMI). It may be the patient size / FOV. Please verify.

## Physics law fitted to your scanner

- k (dose per mAs at 120 kV): 0.06048
- kV exponent n: 2.531

## Pre-scan stage: predicting what the scanner will do, from BMI alone


### mAs (selected: Neural network)

| model | MAPE % | R² |
|---|---|---|
| Neural network **(selected)** | 20.46 ± 0.30 | 0.429 |
| Log-quadratic in BMI | 21.07 ± 0.46 | 0.369 |
| Gaussian process | 21.10 ± 0.51 | 0.375 |
| Random forest | 21.25 ± 0.75 | 0.358 |
| Gradient boosting | 21.41 ± 0.76 | 0.338 |
| k-nearest neighbours | 23.61 ± 0.83 | 0.199 |
| Log-linear in BMI | 23.71 ± 0.92 | -0.120 |
| Median (no BMI) | 31.54 ± 0.59 | -0.054 |

### scan_length_cm (selected: Log-quadratic in BMI)

| model | MAPE % | R² |
|---|---|---|
| Log-quadratic in BMI **(selected)** | 4.98 ± 0.14 | 0.784 |
| Gaussian process | 5.04 ± 0.14 | 0.781 |
| Random forest | 5.14 ± 0.09 | 0.756 |
| Gradient boosting | 5.52 ± 0.12 | 0.709 |
| Log-linear in BMI | 5.79 ± 0.18 | 0.632 |
| Neural network | 5.89 ± 0.17 | 0.686 |
| k-nearest neighbours | 5.93 ± 0.12 | 0.659 |
| Median (no BMI) | 10.81 ± 0.19 | -0.049 |

## CTDIvol: repeated grouped 5-fold CV (mean ± sd over repeats)

| model | MAPE % | MAE | RMSE | R² | max APE % |
|---|---|---|---|---|---|
| Physics law (mAs·kV^n) **(selected)** | 0.77 ± 0.00 | 0.25 | 0.67 | 0.9988 | 8.2 |
| Physics + BMI term | 0.77 ± 0.00 | 0.26 | 0.67 | 0.9988 | 8.2 |
| Physics + GP residual | 0.97 ± 0.06 | 0.34 | 0.64 | 0.9989 | 8.1 |
| Log-log ridge (poly2) | 1.08 ± 0.06 | 0.45 | 0.72 | 0.9986 | 8.2 |
| Physics + NN residual | 1.15 ± 0.01 | 0.46 | 0.70 | 0.9986 | 7.7 |
| Deep MLP ensemble | 1.33 ± 0.24 | 0.54 | 1.12 | 0.9963 | 10.2 |
| Monotone gradient boosting | 4.55 ± 0.63 | 2.47 | 4.22 | 0.9498 | 25.8 |

90% conformal prediction interval: prediction × [0.928, 1.003]

## DLP: repeated grouped 5-fold CV (mean ± sd over repeats)

| model | MAPE % | MAE | RMSE | R² | max APE % |
|---|---|---|---|---|---|
| Physics + GBM residual | 9.42 ± 0.56 | 103.35 | 135.97 | 0.9110 | 30.4 |
| Physics + NN residual | 9.78 ± 0.26 | 110.21 | 149.45 | 0.8926 | 25.7 |
| Physics + length + BMI | 9.80 ± 0.27 | 110.31 | 149.49 | 0.8926 | 25.9 |
| Physics law (CTDI × const length) **(selected)** | 9.83 ± 0.15 | 108.38 | 141.93 | 0.9033 | 29.3 |
| Physics + GP residual | 9.93 ± 0.58 | 111.77 | 150.04 | 0.8917 | 27.3 |
| Log-log ridge (poly2) | 10.10 ± 0.72 | 116.09 | 170.38 | 0.8575 | 37.2 |
| Physics + length | 10.14 ± 0.29 | 111.63 | 144.92 | 0.8992 | 29.6 |
| Monotone gradient boosting | 11.70 ± 1.09 | 126.79 | 169.46 | 0.8608 | 45.1 |
| Deep MLP ensemble | 13.13 ± 1.11 | 154.90 | 225.49 | 0.7488 | 52.8 |

90% conformal prediction interval: prediction × [0.827, 1.229]