# CT Dose AI

Predicts **CTDIvol** and **DLP** from **BMI, kVp, mAs and scan length**, and recommends which settings to change
(kVp / mAs / scan length) to lower dose. BMI is treated as fixed. Stage 1 covers coronary CT angiography (CCTA,
50 exams). Other exam types can be added without changing any code.

## Quick start (Windows)

1. Install Python 3.10 or newer from python.org (tick "Add Python to PATH").
2. Double-click **`setup_windows.bat`**. This is a one-time step. It creates `.venv`, installs the requirements and
   trains the models.
3. Double-click **`run_app.bat`**. The app opens in your browser at http://localhost:8501.

Manual commands: `pip install -r requirements.txt`, then `python app.py` or `streamlit run app.py`.
`python app.py` relaunches itself through Streamlit and opens the browser. Other commands: `python train.py`
and `python -m pytest -q`.
On first start, or when your scikit-learn version differs from the one that saved the models, the app retrains
automatically. This takes about a minute.

## What the app does

| Tab | Purpose |
|---|---|
| **Predict** | CTDIvol and DLP with 90% prediction intervals, a comparison with a DLP reference level, and dose-vs-mAs curves for each kV. Warns when inputs fall outside the training data. |
| **Pre-scan (BMI only)** | Dose estimate *before* the scan, when only BMI is known. A second model predicts what the tube-current modulation will choose, and the physics law turns that into CTDIvol and DLP, with honestly wider intervals. |
| **Reduce dose** | The recommendation engine. Tries kV changes with mAs compensation that keeps the estimated iodine CNR, compares against peers with the same BMI in your own data, and shows the effect of shortening the scan. Constraints: minimum kV by BMI, tube mAs limit, available kV values. Also includes a manual what-if panel. |
| **Phantom optimisation (BO)** | Bayesian optimisation. You scan the phantom at the setting it proposes, enter the measured CNR, and it proposes the next scan. It finds the lowest CTDIvol that still meets the CNR target, usually within 8–20 scans. The calibrated γ feeds *Reduce dose*. |
| **Clinical learning (bandit)** | Safe contextual bandit. It proposes a protocol per patient, a radiologist approves it or overrides it, and scores the scan from 1 to 4 afterwards. It learns the lowest acceptable dose per BMI and lowers the dose step by step within safety limits. Scored exams with console doses also become training data. |
| **Batch** | Upload a CSV, Excel or PDF file to predict every row. If the file includes actual doses, it also shows the errors. |
| **Model & data** | Cross-validation results for every model, the physics law fitted to your scanner, out-of-fold prediction plots and the data audit. |
| **Train / add exams** | Upload data for a new exam type (such as brain CT or chest CT). Each exam gets its own model. |

## Modelling approach

Dose quantities are positive and multiply together, so every model is fitted in **log space**. The training script
`train.py` builds and compares these models:

* **Physics-informed law**, fitted robustly with a Huber loss:
  `CTDIvol = k · mAs · (kV/120)^n` and `DLP = CTDIvol × L`. The mAs exponent is fixed at 1 by physics.
* **Physics + ML residual**: Gaussian process, neural network or gradient boosting learns what the physics law misses.
* Statistical and ML baselines: log-log ridge with interactions, gradient boosting with monotone constraints, and a
  bagged ensemble of deep MLPs.
* **Validation**: repeated grouped 5-fold CV (10 repeats). Duplicate rows always stay in the same fold, so they cannot
  inflate the score.
* **Selection**: the 1-standard-error rule. It picks the simplest model whose score is within one SE of the best.
* **Uncertainty**: conformal prediction intervals calibrated on out-of-fold residuals. Their coverage does not depend
  on distribution assumptions.
* **Applicability domain**: the app flags inputs outside the training ranges, including kV values not seen in the data.

### Results on the CCTA data (50 exams, 48 unique)

| Target | Best model (selected) | CV MAPE | R² | Deep MLP ensemble |
|---|---|---|---|---|
| CTDIvol | Physics law, k = 0.0605 mGy/mAs, n = 2.53 | **0.9%** | 0.998 | 2.3% |
| DLP | Physics law × constant length | **8.2%** | 0.92 | 12.4% |

With 48 unique exams, the physics-informed model beats every black-box model, including deep learning. This is the
expected result: CTDIvol is essentially a scanner constant times mAs times a kV factor. More complex models would
become worthwhile with thousands of exams and extra inputs such as heart rate, pitch, collimation, gating mode and
iterative-reconstruction level. The same pipeline trains them automatically when that data exists.

## The pre-scan stage — where flexible models do win

The dose law is physics, so no black-box model beats it. But the question *"what will the scanner choose for this
patient?"* has no formula, and there the model competition comes out the other way (repeated cross-validation,
50 exams, BMI as the only input):

| Model | mAs error | Scan-length error |
|---|---|---|
| Median, ignoring BMI (baseline) | 27.9% | 9.6% |
| Log-linear in BMI | 23.4% | 5.6% |
| Gaussian process | 21.2% | **5.3%** (selected) |
| **Random forest** | **19.5%** (selected) | 5.4% |
| Neural network | 22.8% | 6.1% |

Chaining the two stages gives a dose estimate from BMI alone (BMI → mAs → physics → CTDIvol, DLP). The interval
combines both stages' out-of-fold residuals, so it is around ±35% instead of the ±7% you get once the mAs is
known — wide, but honest, and useful for consenting, planning and triage.

## Learning loops (image quality)

The dose predictor answers "what dose will this protocol give?". The two learning loops answer "how low can the
dose go and still give a diagnostic image?". They share one physics core: the fitted scanner law and the iodine CNR
model.

**Phantom Bayesian optimisation** (`ctdose/phantom_bo.py`)
- Surrogate for log CNR: `b0 − γ·log(kV/120) + δ·log CTDIvol`, a Bayesian linear model with physics priors
  (γ ≈ 1.35, δ ≈ 0.5), plus a Gaussian-process residual that captures effects such as photon starvation.
- Acquisition: constrained expected improvement, `(best feasible dose − dose) × P(CNR ≥ target)`.
- Simulated benchmark (10 synthetic phantoms): 8–9 scans to converge. The recommendation came within about 8% of
  the true optimum dose, and the CNR target was met in 10 of 10 runs.

**Safe contextual bandit** (`ctdose/bandit.py`)
- Model: Bayesian logistic regression of P(score ≥ 3) on BMI, log CTDIvol and log kV, with physics-informed priors.
  All patients share one model, so it learns from every exam.
- Decision: Thompson sampling picks the lowest-dose protocol whose sampled P(acceptable) ≥ target (default 0.97).
- Safety: the BMI minimum-kV rule applies. The pessimistic (10th-percentile) P(acceptable) must be ≥ 0.85, and a
  proposal can go at most 15% below the lowest dose already proven acceptable for a similar BMI. A radiologist
  approves every proposal.
- Simulated benchmark (300 synthetic patients, synthetic reader): about 30% lower dose than current practice in the
  first 50 patients, rising to 45–50% later. The non-acceptable rate stayed at 3–5%, in line with the 97% target.
- Logs are stored in `data/clinical/<exam>_bandit_log.csv`. Scored rows whose CTDIvol and DLP were read from the
  console are added to the training data at the next `python train.py`.

The simulations show that the algorithms behave correctly. They do not prove clinical benefit: real gains depend
on your phantom, your scanner and your readers. Any clinical use needs approval from the medical physicist and
the responsible radiologist.

## Findings to discuss with the data provider

1. **The "SCAN SIZE CM" field does not seem to be the irradiated scan length.** DLP/CTDIvol, which is the real
   irradiated length, averages about 23 cm and has *no* positive correlation with that field (r = −0.23). The field
   does track BMI (r = 0.89). It may be the patient diameter or the FOV. If the true z-coverage is recorded, adding it
   would substantially improve the DLP model. The DLP error (about 8%) comes almost entirely from this missing
   information.
2. **Five exams (rows 2, 9, 22, 41, 45) have about 7% lower CTDIvol per mAs.** This usually indicates a different
   collimation, pitch or scan mode. Recording that parameter would remove most of the remaining CTDIvol error.
3. **Rows 28/43 and 29/42 are exact duplicates.** They are handled in CV, but it is worth checking whether this is a
   copy error.
4. **Dose level.** Median DLP is about 1080 mGy·cm, while published CCTA references are roughly 170–400 mGy·cm for
   prospective gating and roughly 870–1060 for retrospective gating. The protocol is probably retrospective gating at
   120 kV for almost every BMI. The largest savings come from (a) 100 kV (or 80 kV in slim patients), which gives about
   40% less dose at the same iodine CNR, and (b) prospective ECG triggering where heart rate allows. Option (b) is
   outside the four input parameters.

## Adding other exams

Put a CSV or Excel file in `data/` with these columns (Greek or English headers both work):
`exam, BMI, kVp (or KV), mAs, scan length, CTDIvol (or CTDI), DLP`. Then run `python train.py` or use the
**Train** tab. Every value in the `exam` column gets its own model and report.

## Caveats

This is a decision-support tool, not a medical device. Recommendations at kV values not present in the training data
(such as 80 or 100 kV here) rely on the fitted physics kV law. The CNR model is a first-order approximation
(iodine contrast ∝ kV^-γ, noise ∝ 1/√CTDIvol). Verify any protocol change with a phantom and a medical physicist
before using it clinically.

References: Stocker et al., *PROTECTION VI*, Eur Heart J 2018 (DLP reference 400 mGy·cm; kV by BMI);
national CCTA DRLs summarised in Radiation Doses in Cardiovascular CT, PMC10141413.

---

## Σύνοψη (για την ομάδα)

* **CTDIvol**: προβλέπεται με σφάλμα ~1% από τον φυσικό νόμο CTDIvol = 0,0605 · mAs · (kV/120)^2,53, που
  προσαρμόστηκε στον τομογράφο σας. Κανένα μοντέλο ML ή deep learning δεν τον ξεπερνά με 50 εξετάσεις.
* **DLP**: σφάλμα ~8%. Το πεδίο «SCAN SIZE CM» δεν φαίνεται να είναι το πραγματικό μήκος σάρωσης (συσχετίζεται με
  το BMI, όχι με το DLP/CTDI). Αν μας δώσετε το πραγματικό μήκος z, το σφάλμα θα πέσει σημαντικά.
* **Νέο**: Bayesian optimisation σε ομοίωμα (phantom), που βρίσκει τα χαμηλότερα kV/mAs με αποδεκτό CNR σε
  ~8–20 σαρώσεις, και ασφαλές contextual bandit με βαθμολογίες ακτινολόγων (1–4), που μαθαίνει σταδιακά τη
  χαμηλότερη αποδεκτή δόση ανά BMI. Κάθε πρόταση την εγκρίνει ακτινολόγος.
* **Μείωση δόσης**: στα 100 kV (ή 80 kV για BMI < 25) η δόση πέφτει ~40–65% με ίδιο εκτιμώμενο CNR ιωδίου,
  αν το mAs προσαρμοστεί όπως προτείνει η εφαρμογή. Επίσης, κάθε 1 cm λιγότερο μήκος σάρωσης μειώνει το DLP κατά ~3,5%.
