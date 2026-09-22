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
