"""Model evaluation: repeated grouped K-fold CV, metrics, model selection,
and conformal (distribution-free) prediction intervals."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone

from .models import COMPLEXITY


def grouped_folds(groups: np.ndarray, k: int, seed: int):
    """K folds over unique groups (duplicates never straddle train/test)."""
    rng = np.random.RandomState(seed)
    ug = np.unique(groups)
    rng.shuffle(ug)
    for fold in np.array_split(ug, k):
        te = np.isin(groups, fold)
        yield np.where(~te)[0], np.where(te)[0]


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    ss = ((y - y.mean()) ** 2).sum()
    return {
        "MAPE_%": float(np.mean(np.abs(err) / y) * 100),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2": float(1 - (err ** 2).sum() / ss) if ss > 0 else float("nan"),
        "MaxAPE_%": float(np.max(np.abs(err) / y) * 100),
    }


def cross_validate(models: dict, X, y, groups, k: int = 5, repeats: int = 10, verbose=True):
    """Repeated grouped K-fold. Returns (summary DataFrame, oof dict name -> (repeats, n) preds)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    rows, oof = [], {}
    for name, est in models.items():
        t0 = time.time()
        P = np.full((repeats, len(y)), np.nan)
        per_rep = []
        for r in range(repeats):
            for tr, te in grouped_folds(groups, k, seed=r):
                m = clone(est).fit(X[tr], y[tr])
                P[r, te] = m.predict(X[te])
            per_rep.append(metrics(y, P[r]))
        oof[name] = P
        df = pd.DataFrame(per_rep)
        row = {"model": name}
        for c in df.columns:
            row[c] = df[c].mean()
            row[c + "_sd"] = df[c].std(ddof=1)
        row["fit_s"] = time.time() - t0
        rows.append(row)
        if verbose:
            print(f"    {name:38s} MAPE {row['MAPE_%']:6.2f}% ±{row['MAPE_%_sd']:.2f}  R² {row['R2']:.4f}  ({row['fit_s']:.1f}s)")
    return pd.DataFrame(rows).sort_values("MAPE_%").reset_index(drop=True), oof


def select_one_se(summary: pd.DataFrame) -> str:
    """1-standard-error rule: simplest model within 1 SE of the best CV MAPE."""
    best = summary.iloc[0]
    thr = best["MAPE_%"] + best["MAPE_%_sd"]
    ok = summary[summary["MAPE_%"] <= thr].copy()
    ok["cx"] = ok["model"].map(COMPLEXITY).fillna(9)
    return ok.sort_values(["cx", "MAPE_%"]).iloc[0]["model"]


class ConformalCalibrator:
    """Split/CV+ style conformal calibration on log-ratio residuals.

    Uses out-of-fold residuals r = log(y) - log(ŷ) and returns multiplicative
    bounds  [ŷ·exp(q_lo), ŷ·exp(q_hi)]  with finite-sample coverage ≥ 1-α
    (under exchangeability).
    """

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha

    def fit(self, y, oof_pred):
        y = np.asarray(y, float)
        P = np.atleast_2d(oof_pred)
        r = (np.log(y)[None, :] - np.log(P)).ravel()
        r = r[np.isfinite(r)]
        n = len(y)
        a = self.alpha
        lo_level = max(0.0, np.floor((n + 1) * (a / 2)) / n)
        hi_level = min(1.0, np.ceil((n + 1) * (1 - a / 2)) / n)
        self.q_lo_ = float(np.quantile(r, lo_level))
        self.q_hi_ = float(np.quantile(r, hi_level))
        self.abs_q_ = float(np.quantile(np.abs(r), min(1.0, np.ceil((n + 1) * (1 - a)) / n)))
        return self

    def interval(self, pred):
        pred = np.asarray(pred, float)
        return pred * np.exp(self.q_lo_), pred * np.exp(self.q_hi_)

    def coverage(self, y, oof_pred) -> float:
        lo, hi = self.interval(np.atleast_2d(oof_pred))
        y = np.asarray(y, float)[None, :]
        return float(np.mean((y >= lo) & (y <= hi)))
