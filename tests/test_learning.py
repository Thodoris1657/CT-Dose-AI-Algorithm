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
