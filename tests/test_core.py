"""Run:  python -m pytest -q"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ctdose.data import FEATURES, TARGETS, audit, load_table, row_groups
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
    assert len(df) == 89 and df[FEATURES].notna().all().all()
    assert audit(df)["n_unique"] == 87


def test_duplicates_share_group(df):
    """Identical rows must land in the same CV fold, whatever their position."""
    g = row_groups(df)
    key = df[FEATURES + TARGETS].round(4).apply(tuple, axis=1)
    dup = key[key.duplicated(keep=False)]
    assert len(dup) >= 2
    for _, idx in dup.groupby(dup).groups.items():
        idx = list(idx)
        assert len({g[i] for i in idx}) == 1


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


# ---------------------------------------------------------------- data quality screen
def test_quality_screen_finds_the_known_bad_rows(df):
    """Row 65 (DLP implies 66 cm) and row 85 (-27% dose per mAs) are transcription errors."""
    from ctdose import quality
    rep = quality.report(df)
    errors = {f["row"] for f in rep["flagged"] if f["flag"] == "error"}
    assert {65, 85} <= errors
    assert rep["n_error"] == len(errors)


def test_quality_screen_keeps_the_systematic_cluster(df):
    """The -7% dose-per-mAs rows are a second scan mode, not errors: review, never excluded."""
    from ctdose import quality
    rep = quality.report(df)
    assert rep["n_review"] >= 5
    assert rep["systematic_cluster"]["n"] >= 5
    keep, held = quality.split(df)
    assert len(keep) == len(df) - rep["n_error"]
    assert set(held.flag) == {"error"}


def test_scan_length_still_not_predictable(df):
    """If this ever starts passing differently, the DLP model should be revisited."""
    from ctdose import quality
    res = quality.length_is_predictable(df)
    assert max(v["R2"] for v in res.values() if isinstance(v, dict)) < 0.25


def test_physics_law_is_stable_under_more_data(df):
    """The scanner constants must not wander when the dataset grows."""
    import numpy as np
    from ctdose import quality
    from ctdose.models import PhysicsModel
    clean, _ = quality.split(df)
    first, rest = clean.iloc[:45], clean.iloc[45:]
    a = PhysicsModel().fit(first[FEATURES].to_numpy(float), first.CTDIvol.to_numpy(float))
    b = PhysicsModel().fit(rest[FEATURES].to_numpy(float), rest.CTDIvol.to_numpy(float))
    assert abs(np.exp(a.intercept_) - np.exp(b.intercept_)) / np.exp(a.intercept_) < 0.02


def test_model_generalises_to_unseen_exams(df):
    """Train on the first half, predict the second: CTDIvol must stay under 2%."""
    import numpy as np
    from sklearn.base import clone
    from ctdose import quality
    from ctdose.models import candidates
    clean, _ = quality.split(df)
    tr, te = clean.iloc[:45], clean.iloc[45:]
    m = clone(candidates("CTDIvol")["Physics law (mAs·kV^n)"]).fit(
        tr[FEATURES].to_numpy(float), tr.CTDIvol.to_numpy(float))
    p = m.predict(te[FEATURES].to_numpy(float))
    assert np.mean(np.abs(p - te.CTDIvol.to_numpy(float)) / te.CTDIvol.to_numpy(float)) < 0.02
