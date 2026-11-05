# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [semantic versioning](https://semver.org/).

## [1.1.0] — 2026-11-05

### Added
- **Data-quality screen** (`ctdose/quality.py`): physics-based validation of every data file before
  training. Checks that `CTDIvol / mAs` matches the scanner constant for that tube voltage and that
  `DLP / CTDIvol` is a plausible irradiated length. Rows are labelled `ok`, `review` or `error`;
  rows flagged `error` are excluded from training and reported, never deleted. Override with
  `python train.py --keep-flagged`.
- Median absolute percentage error (`MedAPE_%`) alongside MAPE, so a single outlier cannot dominate
  the reported accuracy.
- Data-quality panel in the application (Expert mode → *Model details*) and a provenance note on the
  prediction tab.
- Five tests covering the quality screen, the stability of the fitted constants and generalisation to
  unseen examinations.

### Changed
- **Dataset: 50 → 89 coronary CT examinations** (87 used for training). Source files moved to
  `data/raw/`; only the top level of `data/` is treated as training data.
- Retrained all models. The fitted dose law is essentially unchanged — `k` 0.06050 → 0.06048,
  kV exponent 2.533 → 2.531 — despite 78% more data.
- CTDIvol cross-validated error 0.92% → 0.77%. DLP 8.2% → 9.8%: the larger sample contains more
  variety in irradiated length, so the earlier figure was optimistic.
- Pre-scan mAs model: the neural network (20.5%) now outperforms the random forest that won on 50
  examinations.
- README updated with the new results, the prospective test and the data-quality policy.

### Fixed
- `train.py` no longer picks up backup and source files left in `data/`, which silently duplicated
  training rows.

### Validation
- Prospective test: the model fitted on the original 50 examinations predicted the 37 clean new ones
  at **0.58% mean error** (median 0.15%) for CTDIvol, against 1.89% for a deep MLP ensemble and
  4.27% for gradient boosting.
- With 87 clean examinations, the irradiated length is confirmed unpredictable from the four input
  columns (cross-validated R² = 0.05), so the residual DLP error is irreducible until the true
  z-coverage is exported.

## [1.0.0] — 2026-09-29

### Added
- Physics-informed prediction of CTDIvol and DLP, benchmarked against eight alternatives under
  repeated grouped cross-validation, with conformal prediction intervals.
- Dose-reduction engine: iso-CNR search over kV, mAs and scan length with BMI-based safety limits.
- Phantom Bayesian optimisation and a safe contextual bandit for learning from radiologist scores.
- Pre-scan prediction of dose from BMI alone.
- Streamlit application with Simple and Expert modes; LaTeX report, presentation and cheatsheet.
