"""Physics-based data-quality screening.

Dose data arrives by hand-typing numbers off a console, so transcription errors are
normal. They matter: a single row with a ten-fold DLP shifts the fitted effective
length for every patient. This module finds such rows *using physics rather than
statistics*, which is what makes it reliable on 50 rows as well as on 5000.

Two residuals carry all the information:

    yield  = CTDIvol / mAs      -> a scanner constant at a given kV.
                                   Deviations are real protocol differences
                                   (collimation, pitch, scan mode) or typos.
    length = DLP / CTDIvol      -> the irradiated length in cm.
                                   Must be physiologically plausible for the exam.

Each row is labelled:

    ok      - consistent with the rest of the data
    review  - a systematic, repeatable deviation: several rows share it, so it is
              almost certainly a different scan mode that nobody recorded.
              Kept for training; reported to the user.
    error   - physically implausible in isolation: a yield far off the scanner
              constant, or an implied length that no cardiac scan can have.
              Excluded from training by default, never deleted from the file.

The thresholds are deliberately loose: the goal is to catch the handful of rows a
physicist would also query, not to trim the distribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FEATURES, TARGETS

# a cardiac acquisition cannot plausibly irradiate less than ~8 cm or more than ~60 cm
L_ABS_MIN, L_ABS_MAX = 8.0, 60.0
YIELD_REVIEW = 0.04      # 4%  off the kV median -> worth reporting
YIELD_ERROR = 0.15       # 15% off the kV median -> not the same machine setting
L_REL_SPAN = 2.0         # implied length more than 2x (or less than 1/2) the median


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(np.median(np.abs(x - np.median(x))) * 1.4826) or float("nan")


def screen(df: pd.DataFrame) -> pd.DataFrame:
    """Return the input with quality columns added: yield, implied length, flag, reason."""
    d = df.copy().reset_index(drop=True)
    d["row"] = np.arange(1, len(d) + 1)          # 1-based, matches the spreadsheet
    ok_rows = d[FEATURES + TARGETS].notna().all(axis=1)

    d["ctdi_per_mas"] = d.CTDIvol / d.mAs
    d["implied_length_cm"] = d.DLP / d.CTDIvol
    med_yield = d.groupby("kVp").ctdi_per_mas.transform("median")
    d["yield_dev"] = d.ctdi_per_mas / med_yield - 1
    med_L = float(d.loc[ok_rows, "implied_length_cm"].median())
    mad_L = _mad(d.loc[ok_rows, "implied_length_cm"])
    d["length_z"] = (d.implied_length_cm - med_L) / (mad_L if mad_L == mad_L and mad_L > 0 else 1.0)

    flags, reasons = [], []
    for r in d.itertuples():
        f, why = "ok", []
        if not ok_rows[r.Index]:
            flags.append("error"); reasons.append("missing values"); continue
        if abs(r.yield_dev) > YIELD_ERROR:
            f = "error"
            why.append(f"CTDIvol/mAs is {r.yield_dev*100:+.0f}% off the {r.kVp:g} kV constant "
                       f"— one of CTDIvol or mAs looks mistyped")
        elif abs(r.yield_dev) > YIELD_REVIEW:
            f = "review"
            why.append(f"CTDIvol/mAs is {r.yield_dev*100:+.0f}% off the {r.kVp:g} kV constant "
                       f"— probably a different collimation, pitch or scan mode")
        if not (L_ABS_MIN <= r.implied_length_cm <= L_ABS_MAX) or \
           not (med_L / L_REL_SPAN <= r.implied_length_cm <= med_L * L_REL_SPAN):
            f = "error"
            why.append(f"DLP/CTDIvol implies {r.implied_length_cm:.0f} cm of coverage "
                       f"(the rest of the data sits at {med_L:.0f} cm) — check the DLP, "
                       f"or whether this exam combined several acquisitions")
        flags.append(f)
        reasons.append("; ".join(why))
    d["flag"], d["reason"] = flags, reasons
    return d


def split(df: pd.DataFrame, exclude: tuple = ("error",)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rows used for training, rows held out) — held-out rows are reported, never deleted."""
    s = screen(df)
    keep = ~s.flag.isin(exclude)
    return df.reset_index(drop=True)[keep.to_numpy()].reset_index(drop=True), s[~keep.to_numpy()]


def report(df: pd.DataFrame) -> dict:
    """A compact, JSON-friendly summary for the model card and the app."""
    s = screen(df)
    med_L = float(s.implied_length_cm.median())
    out = {
        "n_rows": int(len(s)),
        "n_ok": int((s.flag == "ok").sum()),
        "n_review": int((s.flag == "review").sum()),
        "n_error": int((s.flag == "error").sum()),
        "implied_length_median_cm": round(med_L, 2),
        "implied_length_sd_cm": round(float(s.implied_length_cm.std()), 2),
        "ctdi_per_mas_by_kV": {f"{k:g}": round(float(v), 5)
                               for k, v in s.groupby("kVp").ctdi_per_mas.median().items()},
        "flagged": [],
    }
    for r in s[s.flag != "ok"].itertuples():
        out["flagged"].append({
            "row": int(r.row), "flag": r.flag, "BMI": float(r.BMI), "kVp": float(r.kVp),
            "mAs": float(r.mAs), "CTDIvol": float(r.CTDIvol), "DLP": float(r.DLP),
            "yield_dev_pct": round(float(r.yield_dev) * 100, 1),
            "implied_length_cm": round(float(r.implied_length_cm), 1),
            "reason": r.reason,
        })
    # a repeated deviation is a protocol, not a mistake — say so explicitly
    rev = s[s.flag == "review"]
    if len(rev) >= 3 and rev.yield_dev.std() < 0.02:
        out["systematic_cluster"] = {
            "n": int(len(rev)),
            "mean_yield_dev_pct": round(float(rev.yield_dev.mean()) * 100, 1),
            "rows": rev.row.astype(int).tolist(),
            "note": ("These rows share one dose-per-mAs value, so they are a second scan mode rather than "
                     "noise. Recording collimation / pitch / scan mode would let the model separate them."),
        }
    return out


def length_is_predictable(df: pd.DataFrame, seed: int = 0) -> dict:
    """Can the irradiated length be predicted from the four inputs? (It decides the DLP ceiling.)

    Compares a constant against a linear model and a random forest under 5-fold CV.
    A negative or near-zero R² means the remaining DLP error is irreducible with these columns.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_predict

    d, _ = split(df)
    L = (d.DLP / d.CTDIvol).to_numpy(float)
    X = d[["BMI", "mAs", "scan_length_cm"]].to_numpy(float)
    y = np.log(L)
    res = {}
    for name, m in (("constant", None), ("linear", LinearRegression()),
                    ("random forest", RandomForestRegressor(300, min_samples_leaf=4, random_state=seed))):
        p = np.full_like(y, np.median(y)) if m is None else \
            cross_val_predict(m, X, y, cv=KFold(5, shuffle=True, random_state=seed))
        res[name] = {"MAPE_%": round(float(np.mean(np.abs(np.exp(p) - L) / L) * 100), 2),
                     "R2": round(float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()), 3)}
    res["verdict"] = ("the scan length is not recoverable from these columns — the remaining DLP error is "
                      "irreducible until the true z-coverage is exported"
                      if max(v["R2"] for v in res.values() if isinstance(v, dict)) < 0.2 else
                      "part of the scan length is predictable; the DLP model now uses it")
    return res
