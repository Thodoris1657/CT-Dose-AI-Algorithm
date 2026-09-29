"""Pre-scan prediction: what will the scanner do, before the patient is on the table?

The dose model answers "given kV, mAs and length, what dose?" — that part is physics.
This module answers the genuinely statistical questions that come *before* it:

    BMI  ->  mAs      (what the automatic tube-current modulation will choose)
    BMI  ->  scan length

Neither has a closed-form law: they are the joint result of the ATCM algorithm, the
operator's planning and the patient's anatomy. There is no formula to fall back on, so
this is where flexible models earn their place — and in cross-validation they do beat
the linear baseline here, unlike in the dose model.

Chaining the two stages gives a *pre-scan* dose estimate from BMI alone:

    BMI -> mAŝ -> (physics) -> CTDIvol, DLP

with the uncertainty of both stages combined by convolving their out-of-fold residual
distributions (they are independent: the ATCM error does not know about the dose-law
error), so the interval is honest rather than optimistic.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .evaluation import ConformalCalibrator, cross_validate

PRESCAN_TARGETS = ["mAs", "scan_length_cm"]


class LogTarget(BaseEstimator, RegressorMixin):
    """Fit any sklearn regressor on log(y); predict back in native units."""

    def __init__(self, base=None):
        self.base = base

    def fit(self, X, y):
        self.m_ = clone(self.base).fit(np.asarray(X, float), np.log(np.asarray(y, float)))
        return self

    def predict(self, X):
        return np.exp(self.m_.predict(np.asarray(X, float)))


class ConstantModel(BaseEstimator, RegressorMixin):
    """Baseline: ignore BMI, always predict the median. If nothing beats this, BMI carries no signal."""

    def fit(self, X, y):
        self.v_ = float(np.median(np.asarray(y, float)))
        return self

    def predict(self, X):
        return np.full(len(np.asarray(X, float)), self.v_)


def _gp():
    k = ConstantKernel(0.1, (1e-3, 10)) * RBF(1.0, (0.1, 50)) + WhiteKernel(0.05, (1e-4, 1))
    return make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=k, n_restarts_optimizer=2, random_state=0))


def candidates() -> dict:
    """Model zoo for the pre-scan targets (same competition rules as the dose models)."""
    return {
        "Median (no BMI)": ConstantModel(),
        "Log-linear in BMI": LogTarget(LinearRegression()),
        "Log-quadratic in BMI": LogTarget(make_pipeline(
            PolynomialFeatures(2, include_bias=False), StandardScaler(), LinearRegression())),
        "Gaussian process": LogTarget(_gp()),
        "k-nearest neighbours": LogTarget(KNeighborsRegressor(n_neighbors=7, weights="distance")),
        "Random forest": LogTarget(RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0)),
        "Gradient boosting": LogTarget(HistGradientBoostingRegressor(
            max_depth=2, learning_rate=0.06, max_iter=250, min_samples_leaf=5, l2_regularization=1.0, random_state=0)),
        "Neural network": LogTarget(make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(32, 16), alpha=0.5, solver="lbfgs", max_iter=800, random_state=0))),
    }


COMPLEXITY = {"Median (no BMI)": 0, "Log-linear in BMI": 1, "Log-quadratic in BMI": 2, "k-nearest neighbours": 3,
              "Gaussian process": 3, "Gradient boosting": 4, "Random forest": 4, "Neural network": 5}


class PrescanPredictor:
    """Trained BMI -> {mAs, scan length} models, with conformal residuals for chaining."""

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.models: dict = {}
        self.selected: dict = {}
        self.cv_summary: dict = {}
        self.calibrators: dict = {}
        self.residuals: dict = {}      # target -> pooled out-of-fold log residuals
        self.bmi_range: tuple = (0.0, 0.0)

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame, groups=None, k: int = 5, repeats: int = 10, verbose: bool = True):
        X = df[["BMI"]].to_numpy(float)
        groups = np.arange(len(df)) if groups is None else groups
        self.bmi_range = (float(df.BMI.min()), float(df.BMI.max()))
        for t in PRESCAN_TARGETS:
            y = df[t].to_numpy(float)
            if verbose:
                print(f"  --- pre-scan {t} (from BMI alone)")
            cands = candidates()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                summ, oof = cross_validate(cands, X, y, groups, k=k, repeats=repeats, verbose=verbose)
            # 1-SE rule with the pre-scan complexity ordering
            best = summ.iloc[0]
            ok = summ[summ["MAPE_%"] <= best["MAPE_%"] + best["MAPE_%_sd"]].copy()
            ok["cx"] = ok.model.map(COMPLEXITY).fillna(9)
            name = ok.sort_values(["cx", "MAPE_%"]).iloc[0]["model"]
            if verbose:
                print(f"  selected: {name}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.models[t] = clone(cands[name]).fit(X, y)
            self.selected[t] = name
            summ["selected"] = summ.model == name
            self.cv_summary[t] = summ
            self.calibrators[t] = ConformalCalibrator(self.alpha).fit(y, oof[name])
            r = (np.log(y)[None, :] - np.log(oof[name])).ravel()
            self.residuals[t] = r[np.isfinite(r)]
        return self

    # ------------------------------------------------------------------ predict
    def predict(self, bmi) -> dict:
        b = np.atleast_2d(np.asarray(bmi, float).reshape(-1, 1))
        out = {}
        for t, m in self.models.items():
            p = m.predict(b)
            lo, hi = self.calibrators[t].interval(p)
            out[t] = float(p[0]) if len(p) == 1 else p
            out[t + "_lo"], out[t + "_hi"] = (float(lo[0]), float(hi[0])) if len(p) == 1 else (lo, hi)
        return out

    def in_domain(self, bmi: float) -> bool:
        lo, hi = self.bmi_range
        span = hi - lo
        return bool(lo - 0.1 * span <= bmi <= hi + 0.1 * span)

    # ------------------------------------------------------------------ chaining
    def chained_interval(self, point: float, dose_residuals: np.ndarray, mas_scaling: float = 1.0) -> tuple:
        """Combine the mAs-stage and dose-stage uncertainties.

        Dose is linear in mAs, so in log space the two errors simply add. The interval is
        obtained by convolving the two empirical residual distributions (Monte-Carlo), which
        makes no normality assumption.
        """
        rng = np.random.RandomState(0)
        a = rng.choice(self.residuals["mAs"], 20000) * mas_scaling
        b = rng.choice(np.asarray(dose_residuals, float), 20000)
        tot = a + b
        lo, hi = np.quantile(tot, [self.alpha / 2, 1 - self.alpha / 2])
        return float(point * np.exp(lo)), float(point * np.exp(hi))

    # ------------------------------------------------------------------ reporting
    def summary_table(self, target: str) -> pd.DataFrame:
        s = self.cv_summary[target]
        return s[["model", "MAPE_%", "MAPE_%_sd", "MAE", "RMSE", "R2", "selected"]]

    def curve(self, n: int = 80) -> pd.DataFrame:
        """Predicted mAs / scan length across the BMI range, with intervals (for plotting)."""
        lo, hi = self.bmi_range
        b = np.linspace(lo - 1, hi + 1, n)
        rows = {"BMI": b}
        for t, m in self.models.items():
            p = m.predict(b.reshape(-1, 1))
            l, h = self.calibrators[t].interval(p)
            rows[t], rows[t + "_lo"], rows[t + "_hi"] = p, l, h
        return pd.DataFrame(rows)
