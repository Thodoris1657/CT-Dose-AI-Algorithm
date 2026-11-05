"""Trained-model bundle: prediction with calibrated intervals + applicability domain."""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

from .data import FEATURES
from .evaluation import ConformalCalibrator
from .models import KV_REF, PhysicsModel

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def slug(exam: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", exam).strip("_") or "exam"


@dataclass
class DosePredictor:
    exam: str
    n_train: int
    models: dict                       # target -> fitted estimator
    selected: dict                     # target -> model name
    calibrators: dict                  # target -> ConformalCalibrator
    physics_ctdi: PhysicsModel         # used for inversion (mAs for a target CTDI)
    cv_summary: dict                   # target -> DataFrame
    oof: dict                          # target -> {model: (repeats, n)}
    train_df: pd.DataFrame
    audit: dict
    ranges: dict
    peer_models: dict = field(default_factory=dict)  # quantile -> QuantileRegressor (CTDI@120 vs BMI)
    prescan: object = None            # PrescanPredictor: BMI -> mAs, scan length (pre-scan planning)
    quality: dict = field(default_factory=dict)   # data-quality screen: flagged rows and the policy applied
    alpha: float = 0.10
    sklearn_version: str = ""
    trained_at: str = field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))

    # ------------------------------------------------------------------ predict
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[FEATURES].to_numpy(float)
        out = df[FEATURES].copy().reset_index(drop=True)
        for t, m in self.models.items():
            p = m.predict(X)
            lo, hi = self.calibrators[t].interval(p)
            out[t], out[t + "_lo"], out[t + "_hi"] = p, lo, hi
        out["in_domain"], out["domain_note"] = zip(*[self.domain_check(r) for r in X])
        return out

    def predict_one(self, BMI, kVp, mAs, scan_length_cm) -> dict:
        r = self.predict(pd.DataFrame([dict(BMI=BMI, kVp=kVp, mAs=mAs, scan_length_cm=scan_length_cm)]))
        return r.iloc[0].to_dict()

    def domain_check(self, x) -> tuple[bool, str]:
        notes = []
        for v, f in zip(x, FEATURES):
            lo, hi = self.ranges[f]
            span = hi - lo if hi > lo else abs(hi) * 0.1 + 1
            if v < lo - 0.1 * span or v > hi + 0.1 * span:
                notes.append(f"{f}={v:g} outside training range [{lo:g}, {hi:g}]")
        kv_levels = self.audit.get("kVp_levels", [])
        if x[1] not in kv_levels:
            notes.append(f"kVp={x[1]:g} not seen in training ({', '.join(f'{k:g}' for k in kv_levels)}); "
                         f"relies on physics kV-law (n={self.physics_ctdi.kv_exponent_:.2f})")
        return (len(notes) == 0, "; ".join(notes))

    # ------------------------------------------------------------ physics bits
    def mas_for_ctdi(self, ctdi_target: float, kv: float, bmi: float, L: float) -> float:
        """Invert the SELECTED CTDI model: mAs such that CTDI(kv, mAs) = target."""
        m = self.models["CTDIvol"]
        # physics initial guess
        k = np.exp(self.physics_ctdi.intercept_)
        guess = ctdi_target / (k * (kv / KV_REF) ** self.physics_ctdi.kv_exponent_)
        grid = np.geomspace(max(guess / 3, 1), guess * 3, 400)
        X = np.column_stack([np.full_like(grid, bmi), np.full_like(grid, kv), grid, np.full_like(grid, L)])
        c = m.predict(X)
        if np.all(np.diff(c) > 0):
            return float(np.interp(ctdi_target, c, grid))
        return float(guess)

    def log_residuals(self, target: str) -> np.ndarray:
        """Pooled out-of-fold log residuals of the selected model (used for chaining)."""
        y = self.train_df[target].to_numpy(float)
        P = np.atleast_2d(self.oof[target][self.selected[target]])
        r = (np.log(y)[None, :] - np.log(P)).ravel()
        return r[np.isfinite(r)]

    def predict_prescan(self, BMI: float, kVp: float | None = None, scan_length_cm: float | None = None) -> dict:
        """Dose estimate BEFORE the scan: only BMI is known.

        Stage 1 (statistical): BMI -> mAs the tube-current modulation will choose, and the
        scan length, unless the length is already known and passed in.
        Stage 2 (physics): the dose law turns those into CTDIvol and DLP.
        The interval combines both stages' out-of-fold residuals, so it is wider - honestly so -
        than the interval you get once the protocol is fixed.
        """
        if self.prescan is None:
            raise RuntimeError("This model was trained without the pre-scan stage; re-run train.py.")
        ps = self.prescan.predict(BMI)
        mas = ps["mAs"]
        length_known = scan_length_cm is not None
        L = float(scan_length_cm) if length_known else ps["scan_length_cm"]
        kv = float(kVp) if kVp is not None else float(self.train_df.kVp.mode().iloc[0])
        dose = self.predict_one(BMI, kv, mas, L)
        out = dict(BMI=BMI, kVp=kv, mAs=mas, mAs_lo=ps["mAs_lo"], mAs_hi=ps["mAs_hi"],
                   scan_length_cm=L, length_predicted=not length_known,
                   in_domain=self.prescan.in_domain(BMI) and dose["in_domain"],
                   domain_note=dose["domain_note"])
        if not length_known:
            out["scan_length_lo"], out["scan_length_hi"] = ps["scan_length_cm_lo"], ps["scan_length_cm_hi"]
        for t in ("CTDIvol", "DLP"):
            out[t] = dose[t]
            lo, hi = self.prescan.chained_interval(dose[t], self.log_residuals(t))
            if t == "DLP" and not length_known:      # DLP also inherits the scan-length uncertainty
                rl = self.prescan.residuals["scan_length_cm"]
                rng = np.random.RandomState(1)
                extra = rng.choice(rl, 20000)
                a = rng.choice(self.prescan.residuals["mAs"], 20000)
                b = rng.choice(self.log_residuals(t), 20000)
                q = np.quantile(a + b + extra, [self.prescan.alpha / 2, 1 - self.prescan.alpha / 2])
                lo, hi = float(dose[t] * np.exp(q[0])), float(dose[t] * np.exp(q[1]))
            out[t + "_lo"], out[t + "_hi"] = lo, hi
        return out

    def peer_ctdi(self, bmi: float) -> dict:
        """CTDIvol (normalised to 120 kV) achieved by peers of the same BMI in the local data."""
        return {q: float(np.exp(m.predict([[bmi]])[0])) for q, m in self.peer_models.items()}

    # ---------------------------------------------------------------- persist
    def save(self, directory: Path | None = None) -> Path:
        directory = Path(directory or MODELS_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        p = directory / f"{slug(self.exam)}.joblib"
        joblib.dump(self, p, compress=3)
        return p

    @staticmethod
    def load(path) -> "DosePredictor":
        return joblib.load(path)


def available_models(directory: Path | None = None) -> dict[str, Path]:
    directory = Path(directory or MODELS_DIR)
    return {p.stem: p for p in sorted(directory.glob("*.joblib"))}


def fit_peer_models(df: pd.DataFrame, kv_exponent: float) -> dict:
    """Quantile regression of log CTDIvol@120kV-equivalent on BMI (local 'achievable dose')."""
    y = np.log(df.CTDIvol.to_numpy() / (df.kVp.to_numpy() / KV_REF) ** kv_exponent)
    X = df[["BMI"]].to_numpy()
    out = {}
    for q in (0.25, 0.5, 0.75):
        out[q] = QuantileRegressor(quantile=q, alpha=0.0, solver="highs").fit(X, y)
    return out
