"""Tests for phantom Bayesian optimisation and the clinical bandit (python -m pytest -q)."""
from pathlib import Path

import numpy as np
import pandas as pd

from ctdose.bandit import BanditConfig, DoseBandit, run_bandit_simulation
from ctdose.phantom_bo import Campaign, CNRSurrogate, dose_fn_from_predictor, run_simulation, suggest
from ctdose.predictor import DosePredictor

MODEL = Path(__file__).resolve().parents[1] / "models" / "CCTA.joblib"


def pred():
    return DosePredictor.load(MODEL)


def test_surrogate_recovers_gamma():
    rng = np.random.RandomState(1)
    kv = rng.choice([80, 100, 120], 40)
    ctdi = rng.uniform(5, 60, 40)
    cnr = np.exp(0.7 - 1.5 * np.log(kv / 120) + 0.5 * np.log(ctdi) + rng.normal(0, 0.02, 40))
    s = CNRSurrogate().fit(kv, ctdi, cnr)
    assert abs(s.gamma - 1.5) < 0.1 and abs(s.delta - 0.5) < 0.05


def test_bo_first_proposal_is_reference_scan():
    c = Campaign(name="t", ref_kVp=120, ref_mAs=700)
    r = suggest(c, dose_fn_from_predictor(pred()))
    assert r.next_scan["kVp"] == 120 and r.next_scan["mAs"] == 700


def test_bo_converges_near_true_optimum_and_meets_target():
    f = dose_fn_from_predictor(pred())
    for seed in range(3):
        r = run_simulation(f, seed=seed)
        assert r["n_scans"] <= 30
        assert r["true_cnr_at_rec"] >= 0.98 * r["target"]              # constraint (almost) met
        assert r["recommended"]["CTDIvol"] <= 1.25 * r["true_optimum"][2]  # within 25% of true optimum
        assert r["recommended"]["CTDIvol"] < 0.6 * r["ref_dose"]


def test_bandit_initial_proposal_is_conservative():
    b = DoseBandit(pred(), BanditConfig())
    p = b.propose(27.0)
    assert p["kVp"] >= 100                      # BMI rule
    ref = b.arms(27.0).query("is_reference").CTDIvol.iloc[0]
    assert p["CTDIvol"] >= 0.84 * ref            # no more than one safety step below the reference arm


def test_bandit_never_below_bmi_kv_rule():
    b = DoseBandit(pred(), BanditConfig())
    for bmi in (18, 24, 28, 33, 45):
        A = b.arms(bmi)
        assert (A[A.allowed].kVp >= (80 if bmi < 25 else 100 if bmi < 30 else 120)).all()


def test_bandit_learns_and_stays_safe():
    r = run_bandit_simulation(pred(), 200, seed=0)
    late = r.iloc[100:]
    assert (1 - late.CTDIvol / late.ctdi_ref).mean() > 0.25      # meaningful dose reduction
    assert (late.score < 3).mean() < 0.10                        # non-acceptable rate stays low


def test_bandit_log_roundtrip(tmp_path):
    b = DoseBandit(pred(), BanditConfig())
    b.record(exam="CCTA", BMI=27, kVp=100, mAs=600, CTDIvol=22.0, DLP=500, scan_length_cm=30, score=4)
    b.save_log("CCTA", tmp_path)
    d = DoseBandit.load_log("CCTA", tmp_path)
    assert len(d) == 1 and bool(d.acceptable.iloc[0])


# ---------------------------------------------------------------- pre-scan stage
def test_prescan_beats_the_no_BMI_baseline():
    """BMI must carry real signal about what the scanner will choose."""
    p = pred()
    s = p.prescan.cv_summary["mAs"]
    base = float(s.loc[s.model == "Median (no BMI)", "MAPE_%"].iloc[0])
    best = float(s["MAPE_%"].min())
    assert best < base - 3          # at least 3 percentage points better than ignoring BMI
    sel = p.prescan.selected["mAs"]
    assert sel != "Median (no BMI)"


def test_prescan_predictions_increase_with_BMI():
    p = pred()
    vals = [p.predict_prescan(b)["CTDIvol"] for b in (20, 27, 35, 45)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_prescan_interval_is_wider_than_the_fixed_protocol_interval():
    """Not knowing the mAs must cost accuracy — the interval has to reflect that."""
    p = pred()
    r = p.predict_prescan(28.0)
    fixed = p.predict_one(28.0, r["kVp"], r["mAs"], r["scan_length_cm"])
    w_pre = r["CTDIvol_hi"] / r["CTDIvol_lo"]
    w_fixed = fixed["CTDIvol_hi"] / fixed["CTDIvol_lo"]
    assert w_pre > 2 * w_fixed
    assert r["CTDIvol_lo"] < r["CTDIvol"] < r["CTDIvol_hi"]


def test_prescan_respects_a_known_scan_length():
    p = pred()
    r = p.predict_prescan(28.0, kVp=120, scan_length_cm=25.0)
    assert r["scan_length_cm"] == 25.0 and r["length_predicted"] is False
