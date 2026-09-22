"""Run:  python -m pytest -q"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ctdose.data import FEATURES, audit, load_table, row_groups
from ctdose.models import PhysicsModel, candidates
from ctdose.optimizer import OptimizerConfig, recommend
from ctdose.predictor import DosePredictor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cardiac_ct.csv"
MODEL = ROOT / "models" / "CCTA.joblib"


@pytest.fixture(scope="module")
def df():
    return load_table(DATA)


def test_data_schema(df):
    assert len(df) == 50 and df[FEATURES].notna().all().all()
    assert audit(df)["n_unique"] == 48


def test_duplicates_share_group(df):
    g = row_groups(df)
    assert g[27] == g[42] and g[28] == g[41]


def test_physics_recovers_synthetic_law():
    rng = np.random.RandomState(0)
    X = np.column_stack([rng.uniform(18, 40, 200), rng.choice([100, 120, 140], 200),
                         rng.uniform(200, 1400, 200), rng.uniform(20, 40, 200)])
    y = 0.06 * X[:, 2] * (X[:, 1] / 120) ** 2.7 * np.exp(rng.normal(0, 0.01, 200))
    m = PhysicsModel().fit(X, y)
    assert abs(m.kv_exponent_ - 2.7) < 0.05
    assert abs(np.exp(m.intercept_) - 0.06) < 0.002


def test_predictions_monotone_in_mas_and_kv():
    p = DosePredictor.load(MODEL)
    base = pd.DataFrame(dict(BMI=[28] * 3, kVp=[120] * 3, mAs=[500, 800, 1100], scan_length_cm=[30] * 3))
    out = p.predict(base)
    assert np.all(np.diff(out.CTDIvol) > 0) and np.all(np.diff(out.DLP) > 0)
    hi_kv = p.predict(base.assign(kVp=140))
    assert np.all(hi_kv.CTDIvol > out.CTDIvol)


def test_intervals_contain_prediction():
    p = DosePredictor.load(MODEL)
    r = p.predict_one(30, 120, 800, 30)
    for t in ("CTDIvol", "DLP"):
        assert r[t + "_lo"] <= r[t] <= r[t + "_hi"]


def test_optimizer_respects_constraints():
    p = DosePredictor.load(MODEL)
    res = recommend(p, 27.0, 120, 800, 30, OptimizerConfig())
    b = res["best"]
    assert b["DLP"] <= res["current"]["DLP"]
    assert b["kVp"] >= res["kv_min"]
    assert b["CNR_rel"] >= 0.97
    # obese patient: must not propose kV below 120
    res2 = recommend(p, 38.0, 120, 1200, 34, OptimizerConfig())
    assert res2["best"]["kVp"] >= 120
