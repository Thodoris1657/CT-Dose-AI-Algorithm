# CT Dose AI

Predicts **CTDIvol** and **DLP** from **BMI, kVp, mAs and scan length**, and recommends which settings to change
(kVp / mAs / scan length) to lower dose. BMI is treated as fixed. Stage 1 covers coronary CT angiography (CCTA,
**89 exams**). Other exam types can be added without changing any code.

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

The app opens in **Simple mode**: one input box at the top, four tabs, and plain-language explanations on screen.
Switch to **Expert mode** in the sidebar to add the phantom loop, the clinical-learning loop, batch processing,
the model card and training.

**Simple mode**

| Tab | Purpose |
|---|---|
| **1 · Dose of this scan** | Dose per slice (CTDIvol) and total dose (DLP), each with a 90% range, an approximate mSv figure, and a green/amber/red comparison with your reference level. |
| **2 · How to lower it** | One recommended change in a card — new kV, new mAs, the saving — with the full option list, the reasoning and a what-if panel behind expanders. |
| **3 · Before the scan** | The same estimate from BMI alone, before the scanner has chosen the mAs. |
| **Help** | How to use it, a glossary of every term on screen, and the common questions. |

**Expert mode adds**

| Tab | Purpose |
|---|---|
| **Phantom testing** | Bayesian optimisation. You scan a phantom at the setting it proposes, enter the measured CNR, and it proposes the next one. It finds the lowest CTDIvol that still meets the CNR target, usually within 8–20 scans, and calibrates the image-quality model used by tab 2. |
| **Learning from doctors** | Safe contextual bandit. One proposal per patient, a radiologist approves or overrides it, then scores the images 1–4. It learns the lowest acceptable dose per BMI, in small evidenced steps. |
| **Many scans at once** | Upload a CSV, Excel or PDF file and predict every row. If the file includes the measured doses, it also reports the error. |
| **Model details** | Cross-validation results for every candidate model, the fitted dose law, predicted-vs-measured plots and the data audit. |
| **Add data** | Upload data for another examination type. Each one gets its own model. |

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

### Results on the CCTA data (89 exams, 87 used for training)

| Target | Best model (selected) | CV MAPE | median APE | R² | Deep MLP ensemble |
|---|---|---|---|---|---|
| CTDIvol | Physics law, k = 0.0605 mGy/mAs, n = 2.53 | **0.8%** | 0.2% | 0.999 | 1.3% |
| DLP | Physics law × constant length | **9.8%** | 9.0% | 0.903 | 13.1% |

**The strongest evidence is prospective.** The model fitted on the first 50 exams was used to predict the 37 new
ones it had never seen:

| Model | CTDIvol error on unseen exams | DLP |
|---|---|---|
| **Physics law** | **0.58%** (median 0.15%) | 12.7% |
| Log-log ridge | 1.01% | 11.6% |
| Deep MLP ensemble | 1.89% | 14.2% |
| Monotone gradient boosting | 4.27% | 12.5% |

The fitted constants barely moved when the dataset grew by 78%: k went from 0.06050 to 0.06048 and the kV
exponent from 2.533 to 2.531. That stability is the real argument for the physics model — it is measuring
something about the scanner, not fitting the sample.

DLP is the opposite story: its error rose from 8.2% (50 exams) to 9.8%, because the larger sample contains more
variety in the irradiated length. The earlier figure was optimistic, and this one is more honest. With 87 exams
the irradiated length is provably not recoverable from the four inputs (cross-validated R² of a random forest:
0.05), so that error is irreducible until the true z-coverage is exported.

## Data quality screen

Dose numbers are typed off a console, so the pipeline screens every file on physics grounds before training
(`ctdose/quality.py`):

* **CTDIvol / mAs** must match the scanner constant for that kV.
* **DLP / CTDIvol** must be a plausible irradiated length.

Each row is labelled `ok`, `review` (a repeatable deviation — a second scan mode) or `error` (physically
inconsistent in isolation). Rows flagged `error` are excluded from training and reported; nothing is ever deleted
from the data file. Use `python train.py --keep-flagged` to train on everything.

On the current 89 exams:

| | Rows | What it is |
|---|---|---|
| Used for training | 87 | |
| Flagged `review` | 7 | rows 2, 9, 22, 41, 45, 57, 89 all deliver 7% less dose per mAs — one second scan mode, not noise |
| Held out as `error` | 2 | row 65: DLP implies 66 cm of coverage against 23 cm everywhere else; row 85: dose per mAs is 27% off the constant |

Excluding those two rows takes CTDIvol from 1.18% to 0.77% and DLP R² from 0.744 to 0.903, which is how much
damage two mistyped numbers do in a dataset this size.

## The pre-scan stage — where flexible models do win

The dose law is physics, so no black-box model beats it. But the question *"what will the scanner choose for this
patient?"* has no formula, and there the model competition comes out the other way (repeated cross-validation,
50 exams, BMI as the only input):

| Model | mAs error | Scan-length error |
|---|---|---|
| Median, ignoring BMI (baseline) | 30.6% | 10.5% |
| Log-linear in BMI | 23.6% | 5.8% |
| Log-quadratic in BMI | 21.2% | **5.0%** (selected) |
| Gaussian process | 21.3% | 5.3% |
| Random forest | 23.6% | 5.5% |
| **Neural network** | **20.8%** (selected) | 5.3% |

With 50 exams a random forest won the mAs task; with 89 the neural network does. This is the one place in the
project where a flexible model keeps winning, and the ranking is still shifting — exactly what you expect while
the sample is small.

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

With 89 exams the first two findings are no longer tentative — they are measured.

1. **The "SCAN SIZE CM" field is not the irradiated scan length.** Across 87 clean exams it correlates with BMI
   at r = +0.89 and with the true implied length (DLP/CTDIvol) at r = −0.07. A random forest given BMI, mAs and
   that column predicts the implied length with R² = 0.05, i.e. not at all. The field is almost certainly patient
   size or field of view. Exporting the real z-coverage — it is in every scanner's dose report — is the single
   change that would take DLP from ~10% error to the ~1% level of CTDIvol.
2. **Seven exams use a second scan mode.** Rows 2, 9, 22, 41, 45, 57 and 89 all deliver 7% less CTDIvol per mAs,
   with almost no spread between them. That is a protocol, not noise: different collimation, pitch or scan mode.
   Recording that one field would let the model separate the two modes instead of averaging them.
3. **Two rows look like transcription errors** and are held out of training automatically: row 65 (DLP 2933
   implies 66 cm of coverage — a factor-of-ten typo, or several acquisitions summed into one row) and row 85
   (CTDIvol 29.5 at 672 mAs is 27% off the scanner constant; 40.5 would fit exactly). Please check both against
   the console.
4. **Is the protocol retrospectively gated?** Median DLP is 1044 mGy·cm against a 400 mGy·cm reference for CCTA.
   If prospective triggering is feasible at the patient's heart rate it saves more than any kV or mAs change —
   but it is outside the four parameters supplied, so the tool cannot propose it.
5. **Which additional fields can be exported routinely?** In priority order: true scan length; gating mode;
   collimation and pitch; heart rate; reconstruction kernel and iterative level; and a 1–4 diagnostic-quality
   score.

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
