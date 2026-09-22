"""Candidate models for CT dose prediction.

Every model works in LOG space (dose quantities are strictly positive and
multiplicative in their drivers), and exposes sklearn-style fit/predict in
native units (mGy, mGy*cm).

Physics prior (CT dose physics):
    CTDIvol  = k_scanner * mAs * (kVp/120)^n        n ≈ 2.5–3.0
    DLP      = CTDIvol * L_irradiated
so in log space:
    log CTDIvol = a + 1·log mAs + n·log(kVp/120)
    log DLP     = a + 1·log mAs + n·log(kVp/120) + β_L·log L + β_B·BMI_z

Model families
--------------
PhysicsModel          fitted physics law (robust Huber regression), few params
HybridModel           physics prior + learned residual (GP / MLP / GBM)
RidgeLogModel         purely statistical log-log ridge regression (+ interactions)
GBMModel              monotone-constrained histogram gradient boosting
MLPEnsemble           bagged deep MLP ensemble (standardised inputs, log target)
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import HuberRegressor, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .data import FEATURES

KV_REF = 120.0
KV_EXPONENT_PRIOR = 2.6   # literature ~2.5-3.0; used only if data has a single kV level
BMI_REF = 27.0


def log_features(X) -> np.ndarray:
    """[BMI_z, log mAs, log(kV/120), log L] from raw X (cols = FEATURES)."""
    X = np.asarray(X, float)
    bmi, kv, mas, L = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return np.column_stack([(bmi - BMI_REF) / 7.0, np.log(mas), np.log(kv / KV_REF), np.log(L)])


# ----------------------------------------------------------------------------
# Physics
# ----------------------------------------------------------------------------
class PhysicsModel(BaseEstimator, RegressorMixin):
    """log y = a + log mAs + n·log(kV/120) [+ β_L log L] [+ β_B BMI_z]

    mAs exponent is FIXED at 1 (dose is linear in tube charge) – this is what
    makes the model extrapolate safely to mAs values never seen in training.
    Fitted with Huber loss so a minority protocol cluster does not bias it.
    """

    def __init__(self, use_length: bool = False, use_bmi: bool = False, kv_exponent: float | None = None):
        self.use_length = use_length
        self.use_bmi = use_bmi
        self.kv_exponent = kv_exponent

    def _design(self, F):
        cols = []
        fit_kv = self.kv_exponent is None and self.kv_free_
        if fit_kv:
            cols.append(F[:, 2])
        if self.use_length:
            cols.append(F[:, 3])
        if self.use_bmi:
            cols.append(F[:, 0])
        return np.column_stack(cols) if cols else np.zeros((len(F), 0))

    def _offset(self, F):
        off = F[:, 1].copy()  # + log mAs (fixed coefficient 1)
        if self.kv_exponent is not None or not self.kv_free_:
            n = self.kv_exponent if self.kv_exponent is not None else KV_EXPONENT_PRIOR
            off += n * F[:, 2]
        return off

    def fit(self, X, y):
        F = log_features(X)
        self.kv_free_ = len(np.unique(np.round(np.asarray(X)[:, 1]))) > 1
        z = np.log(y) - self._offset(F)
        D = self._design(F)
        if D.shape[1] == 0:
            self.intercept_ = float(np.median(z))
            self.coef_ = np.zeros(0)
        else:
            h = HuberRegressor(epsilon=1.35, alpha=1e-6, max_iter=2000).fit(D, z)
            self.intercept_, self.coef_ = float(h.intercept_), h.coef_
        return self

    def predict_log(self, X):
        F = log_features(X)
        return self.intercept_ + self._offset(F) + (self._design(F) @ self.coef_ if len(self.coef_) else 0)

    def predict(self, X):
        return np.exp(self.predict_log(X))

    # human-readable parameters
    def params(self) -> dict:
        p = {"k (dose per mAs at 120 kV)": float(np.exp(self.intercept_))}
        i = 0
        if self.kv_exponent is None and self.kv_free_:
            p["kV exponent n"] = float(self.coef_[i]); i += 1
        else:
            p["kV exponent n (fixed)"] = self.kv_exponent or KV_EXPONENT_PRIOR
        if self.use_length:
            p["length elasticity β_L"] = float(self.coef_[i]); i += 1
        if self.use_bmi:
            p["BMI effect β_B (per 7 BMI units, log)"] = float(self.coef_[i]); i += 1
        return p

    @property
    def kv_exponent_(self) -> float:
        if self.kv_exponent is not None:
            return self.kv_exponent
        return float(self.coef_[0]) if self.kv_free_ else KV_EXPONENT_PRIOR


# ----------------------------------------------------------------------------
# Residual learners
# ----------------------------------------------------------------------------
def make_gp():
    k = ConstantKernel(0.05, (1e-4, 10)) * RBF(length_scale=[1.0] * 4, length_scale_bounds=(0.05, 100)) \
        + WhiteKernel(0.005, (1e-6, 1))
    return make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=k, normalize_y=False, n_restarts_optimizer=2, random_state=0))


class MLPEnsemble(BaseEstimator, RegressorMixin):
    """Bagged ensemble of deep MLPs (log-target). Deep learning baseline."""

    def __init__(self, hidden=(32, 32, 16), n_members: int = 5, alpha: float = 1e-2, log_target: bool = True):
        self.hidden = hidden
        self.n_members = n_members
        self.alpha = alpha
        self.log_target = log_target

    def fit(self, X, y):
        F = log_features(X)
        t = np.log(y) if self.log_target else np.asarray(y, float)
        self.ym_, self.ys_ = t.mean(), t.std() + 1e-9
        self.members_ = []
        rng = np.random.RandomState(0)
        for s in range(self.n_members):
            idx = rng.choice(len(F), len(F), replace=True)
            m = make_pipeline(StandardScaler(), MLPRegressor(
                hidden_layer_sizes=self.hidden, activation="relu", solver="lbfgs", alpha=self.alpha,
                max_iter=600, random_state=s))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(F[idx], (t[idx] - self.ym_) / self.ys_)
            self.members_.append(m)
        return self

    def predict_log(self, X):
        F = log_features(X)
        p = np.mean([m.predict(F) for m in self.members_], axis=0) * self.ys_ + self.ym_
        return p if self.log_target else np.log(np.clip(p, 1e-6, None))

    def predict(self, X):
        return np.exp(self.predict_log(X))


class HybridModel(BaseEstimator, RegressorMixin):
    """Physics prior + ML residual correction (physics-informed ML).

    residual learner in {"gp", "mlp", "gbm"} is trained on log(y) - physics(X).
    If there is no learnable structure the residual model collapses to ~0 and
    the hybrid reduces to the physics law — it cannot do worse by construction
    except via variance, which CV measures.
    """

    def __init__(self, physics: PhysicsModel | None = None, residual: str = "gp"):
        self.physics = physics
        self.residual = residual

    def _res_model(self):
        if self.residual == "gp":
            return make_gp()
        if self.residual == "mlp":
            return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1.0,
                                                                solver="lbfgs", max_iter=3000, random_state=0))
        if self.residual == "gbm":
            return HistGradientBoostingRegressor(max_depth=2, learning_rate=0.05, max_iter=150,
                                                 min_samples_leaf=6, l2_regularization=1.0, random_state=0)
        raise ValueError(self.residual)

    def fit(self, X, y):
        self.phys_ = clone(self.physics or PhysicsModel()).fit(X, y)
        r = np.log(y) - self.phys_.predict_log(X)
        self.res_ = self._res_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res_.fit(log_features(X), r)
        return self

    def predict_log(self, X):
        return self.phys_.predict_log(X) + self.res_.predict(log_features(X))

    def predict(self, X):
        return np.exp(self.predict_log(X))


class RidgeLogModel(BaseEstimator, RegressorMixin):
    """Pure statistical log-log ridge with pairwise interactions (no physics)."""

    def __init__(self, degree: int = 2):
        self.degree = degree

    def fit(self, X, y):
        self.m_ = make_pipeline(PolynomialFeatures(self.degree, include_bias=False), StandardScaler(),
                                RidgeCV(alphas=np.logspace(-4, 3, 30)))
        self.m_.fit(log_features(X), np.log(y))
        return self

    def predict_log(self, X):
        return self.m_.predict(log_features(X))

    def predict(self, X):
        return np.exp(self.predict_log(X))


class GBMModel(BaseEstimator, RegressorMixin):
    """Gradient boosting with monotone constraints (+mAs, +kV) in log space."""

    def fit(self, X, y):
        self.m_ = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=400, min_samples_leaf=4,
                                                l2_regularization=0.5, monotonic_cst=[0, 1, 1, 0], random_state=0)
        self.m_.fit(log_features(X), np.log(y))
        return self

    def predict_log(self, X):
        return self.m_.predict(log_features(X))

    def predict(self, X):
        return np.exp(self.predict_log(X))


# ----------------------------------------------------------------------------
# Candidate zoo (ordered from simplest -> most complex; used by 1-SE rule)
# ----------------------------------------------------------------------------
def candidates(target: str) -> dict[str, BaseEstimator]:
    if target == "CTDIvol":
        phys = PhysicsModel()
        return {
            "Physics law (mAs·kV^n)": phys,
            "Physics + BMI term": PhysicsModel(use_bmi=True),
            "Physics + GP residual": HybridModel(phys, "gp"),
            "Physics + NN residual": HybridModel(phys, "mlp"),
            "Log-log ridge (poly2)": RidgeLogModel(),
            "Monotone gradient boosting": GBMModel(),
            "Deep MLP ensemble": MLPEnsemble(),
        }
    phys = PhysicsModel(use_length=True, use_bmi=True)
    return {
        "Physics law (CTDI × const length)": PhysicsModel(),
        "Physics + length": PhysicsModel(use_length=True),
        "Physics + length + BMI": phys,
        "Physics + GP residual": HybridModel(phys, "gp"),
        "Physics + NN residual": HybridModel(phys, "mlp"),
        "Physics + GBM residual": HybridModel(phys, "gbm"),
        "Log-log ridge (poly2)": RidgeLogModel(),
        "Monotone gradient boosting": GBMModel(),
        "Deep MLP ensemble": MLPEnsemble(),
    }


COMPLEXITY = {  # rank for 1-SE rule
    "Physics law (mAs·kV^n)": 0, "Physics law (CTDI × const length)": 0,
    "Physics + BMI term": 1, "Physics + length": 1, "Physics + length + BMI": 1,
    "Log-log ridge (poly2)": 2,
    "Physics + GP residual": 3, "Physics + GBM residual": 3, "Physics + NN residual": 4,
    "Monotone gradient boosting": 4, "Deep MLP ensemble": 5,
}

__all__ = ["FEATURES", "PhysicsModel", "HybridModel", "RidgeLogModel", "GBMModel", "MLPEnsemble", "candidates",
           "COMPLEXITY", "log_features"]
