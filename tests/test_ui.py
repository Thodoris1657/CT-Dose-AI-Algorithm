"""Smoke tests for the user-interface layer.

The Shiny app itself is exercised by hand, but the figure functions it calls are pure and
cheap to test: if one of them raises, the screen it belongs to goes blank in production.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from ctdose import figures as fg
from ctdose.bandit import BanditConfig, DoseBandit, features as bandit_features
from ctdose.optimizer import OptimizerConfig, dose_curves, recommend
from ctdose.predictor import DosePredictor

MODEL = Path(__file__).resolve().parents[1] / "models" / "CCTA.joblib"


@pytest.fixture(scope="module")
def pred():
    return DosePredictor.load(MODEL)


def test_dose_bar_is_a_figure(pred):
    r = pred.predict_one(27.0, 120, 750, 30)
    f = fg.dose_bar(r["DLP"], r["DLP_lo"], r["DLP_hi"], 400)
    assert isinstance(f, go.Figure) and len(f.data) >= 1


def test_dose_vs_mas_marks_the_current_scan(pred):
    kvs = [80, 100, 120]
    curves = dose_curves(pred, 27.0, 30.0, kvs=kvs, mas_range=(50, 1500))
    r = pred.predict_one(27.0, 120, 750, 30)
    f = fg.dose_vs_mas(curves, kvs, 750, r["DLP"], r["DLP_lo"], r["DLP_hi"], "DLP", 400)
    assert isinstance(f, go.Figure)
    assert any(t.name == "this scan" for t in f.data)


def test_tradeoff_puts_the_suggestion_left_of_the_current_scan(pred):
    """The whole point of screen 2: same quality, less dose."""
    res = recommend(pred, 27.0, 120, 750, 30.0, OptimizerConfig(kv_options=(80, 100, 120, 140)))
    cur, best = res["current"], res["best"]
    assert best is not None
    f = fg.quality_vs_dose(120, cur["CTDIvol"], cur["DLP"] / cur["CTDIvol"], [80, 100, 120], 1.35, 100, best)
    assert isinstance(f, go.Figure)
    assert best["DLP"] < cur["DLP"]
    assert best["CNR_rel"] >= 0.97


def test_acceptability_curve_renders_without_any_scored_exams(pred):
    b = DoseBandit(pred, BanditConfig(), log=DoseBandit.load_log("does_not_exist"))
    r = pred.predict_one(27.0, 120, 750, 30)
    f = fg.acceptability(b.model, bandit_features, 27.0, b.ctdi_ref(27.0), r["CTDIvol"],
                         r["DLP"] / r["CTDIvol"], [100, 120], 0.97, None, 0)
    assert isinstance(f, go.Figure) and len(f.data) >= 2


def test_prescan_curve(pred):
    if getattr(pred, "prescan", None) is None:
        pytest.skip("model has no pre-scan stage")
    r = pred.predict_prescan(27.0)
    f = fg.prescan_curve(pred.prescan.curve(), pred.train_df, 27.0, r["mAs"], "mAs", "t", "mAs")
    assert isinstance(f, go.Figure)


def test_predicted_vs_measured(pred):
    y = pred.train_df.CTDIvol.to_numpy()
    p = pred.oof["CTDIvol"][pred.selected["CTDIvol"]].mean(axis=0)
    assert isinstance(fg.predicted_vs_measured(y, p, "CTDIvol"), go.Figure)


def test_options_bar(pred):
    res = recommend(pred, 27.0, 120, 750, 30.0, OptimizerConfig())
    c = res["candidates"].sort_values("DLP")
    assert isinstance(fg.options_bar(c, res["current"]["DLP"], 400), go.Figure)


def test_app_module_imports_and_defines_the_screens():
    """Catches syntax errors and missing outputs before anyone starts the server."""
    import app
    assert app.app is not None
    assert app.MODELS, "the app must find at least one trained model"
