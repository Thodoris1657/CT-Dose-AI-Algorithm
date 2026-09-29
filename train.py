"""Train, benchmark and persist CT-dose models for every exam type found in ./data.

Usage
-----
    python train.py                       # all files in ./data (csv / xlsx / pdf)
    python train.py --data data/new.xlsx  # a specific file
    python train.py --repeats 20          # more CV repeats (more stable estimates)

For each exam type:
  1. data audit (duplicates, physics consistency, scan-length semantics)
  2. repeated grouped K-fold CV of every candidate model, for CTDIvol and DLP
  3. model selection with the 1-SE rule (simplest model within 1 SE of the best)
  4. refit on all data + conformal calibration of 90 % prediction intervals
  5. local "achievable dose" quantile models (CTDI vs BMI)
  6. save models/<exam>.joblib and reports/<exam>/ (report.md, cv_metrics.csv)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone

from ctdose.data import FEATURES, TARGETS, audit, load_table, row_groups
from ctdose.evaluation import ConformalCalibrator, cross_validate, select_one_se
from ctdose.models import PhysicsModel, candidates
from ctdose.prescan import PrescanPredictor
from ctdose.predictor import DosePredictor, fit_peer_models, slug

ROOT = Path(__file__).resolve().parent


def load_clinical_logs() -> pd.DataFrame:
    """Scored exams from the clinical-learning loop become extra training rows
    (only rows whose CTDIvol / DLP were read from the scanner console)."""
    frames = []
    for p in sorted((ROOT / "data" / "clinical").glob("*_bandit_log.csv")):
        d = pd.read_csv(p, encoding="utf-8-sig")
        if "dose_source" in d:
            d = d[d.dose_source.astype(str).str.lower() == "scanner"]
        d = d.dropna(subset=["BMI", "kVp", "mAs", "scan_length_cm", "CTDIvol", "DLP"])
        if len(d):
            frames.append(d[["exam", "BMI", "kVp", "mAs", "scan_length_cm", "CTDIvol", "DLP"]])
            print(f"  loaded {p.name}: {len(d)} scored clinical exams with scanner doses")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_all(paths: list[Path], include_clinical: bool = True) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            frames.append(load_table(p))
            print(f"  loaded {p.name}: {len(frames[-1])} rows")
        except Exception as e:  # noqa: BLE001
            print(f"  ! skipped {p.name}: {e}")
    if include_clinical:
        clin = load_clinical_logs()
        if len(clin):
            frames.append(clin)
    df = pd.concat(frames, ignore_index=True).dropna(subset=FEATURES + TARGETS)
    return df.reset_index(drop=True)


def train_exam(df: pd.DataFrame, exam: str, repeats: int, k: int, alpha: float, out_models: Path, out_reports: Path):
    print(f"\n=== {exam}: {len(df)} rows ===")
    aud = audit(df)
    print("  audit:", json.dumps({k_: aud[k_] for k_ in ("n_unique", "kVp_levels", "duplicate_rows", "yield_outliers",
                                                        "corr_Leff_vs_scan_length")}, ensure_ascii=False))
    X = df[FEATURES].to_numpy(float)
    groups = row_groups(df)

    models, selected, cals, summaries, oofs = {}, {}, {}, {}, {}
    for t in TARGETS:
        y = df[t].to_numpy(float)
        print(f"  --- {t}")
        cands = candidates(t)
        summ, oof = cross_validate(cands, X, y, groups, k=k, repeats=repeats)
        name = select_one_se(summ)
        print(f"  selected (1-SE rule): {name}")
        models[t] = clone(cands[name]).fit(X, y)
        cal = ConformalCalibrator(alpha).fit(y, oof[name])
        cals[t] = cal
        summ["selected"] = summ.model == name
        summaries[t], oofs[t], selected[t] = summ, oof, name
        print(f"  {int((1-alpha)*100)}% interval: ×[{np.exp(cal.q_lo_):.3f}, {np.exp(cal.q_hi_):.3f}]"
              f"  (OOF coverage {cal.coverage(y, oof[name])*100:.1f}%)")

    print("  --- pre-scan stage (BMI -> mAs, scan length)")
    prescan = PrescanPredictor(alpha).fit(df, groups, k=k, repeats=max(3, repeats // 2))

    phys = PhysicsModel().fit(X, df.CTDIvol.to_numpy())
    ranges = {f: (float(df[f].min()), float(df[f].max())) for f in FEATURES}
    pred = DosePredictor(exam=exam, n_train=len(df), models=models, selected=selected, calibrators=cals,
                         physics_ctdi=phys, cv_summary=summaries, oof=oofs, train_df=df.reset_index(drop=True),
                         audit=aud, ranges=ranges, peer_models=fit_peer_models(df, phys.kv_exponent_), alpha=alpha,
                         sklearn_version=sklearn.__version__, prescan=prescan)
    path = pred.save(out_models)
    print(f"  saved {path.relative_to(ROOT)}")
    write_report(pred, out_reports / slug(exam))
    return pred


def write_report(pred: DosePredictor, d: Path):
    d.mkdir(parents=True, exist_ok=True)
    frames = []
    for t, s in pred.cv_summary.items():
        s = s.copy(); s.insert(0, "target", t); frames.append(s)
    pd.concat(frames).to_csv(d / "cv_metrics.csv", index=False)

    a = pred.audit
    L = []
    L.append(f"# CT dose model report — {pred.exam}\n")
    L.append(f"Trained {pred.trained_at} on {pred.n_train} exams ({a['n_unique']} unique).\n")
    L.append("## Data audit\n")
    L.append(f"- kVp levels present: {', '.join(f'{v:g}' for v in a['kVp_levels'])}")
    L.append(f"- Exact duplicate rows, 1-based row numbers (kept together in CV): {a['duplicate_rows']}")
    for r in a["ctdi_per_mAs_by_kV"]:
        L.append(f"- CTDIvol/mAs at {r['kVp']:g} kV: median {r['median']:.4f} mGy/mAs (range {r['min']:.4f}–{r['max']:.4f}, n={r['count']})")
    if a["yield_outliers"]:
        L.append(f"- Rows with a different dose-per-mAs (≈{np.mean(a['yield_outlier_dev_pct']):.0f}%, likely different "
                 f"collimation/pitch/scan mode), 1-based row numbers: {a['yield_outliers']}")
    L.append(f"- DLP/CTDIvol (implied irradiated length): {a['L_eff_cm']} cm; "
             f"correlation with the 'scan length' field = {a['corr_Leff_vs_scan_length']}, "
             f"correlation of 'scan length' with BMI = {a['corr_BMI_vs_scan_length']}")
    if a["scan_length_is_z_coverage_suspect"]:
        L.append("  - ⚠ The 'scan length / SCAN SIZE' field does not behave like the irradiated z-length "
                 "(DLP/CTDIvol does not scale with it and it tracks BMI). It may be the patient size / FOV. Please verify.")
    phys = pred.physics_ctdi.params()
    L.append("\n## Physics law fitted to your scanner\n")
    for k_, v in phys.items():
        L.append(f"- {k_}: {v:.4g}")
    if getattr(pred, "prescan", None) is not None:
        L.append("\n## Pre-scan stage: predicting what the scanner will do, from BMI alone\n")
        for t in pred.prescan.cv_summary:
            s = pred.prescan.cv_summary[t]
            L.append(f"\n### {t} (selected: {pred.prescan.selected[t]})\n")
            L.append("| model | MAPE % | R² |")
            L.append("|---|---|---|")
            for _, r in s.iterrows():
                star = " **(selected)**" if r["selected"] else ""
                L.append(f"| {r['model']}{star} | {r['MAPE_%']:.2f} ± {r['MAPE_%_sd']:.2f} | {r['R2']:.3f} |")
    for t, s in pred.cv_summary.items():
        L.append(f"\n## {t}: repeated grouped 5-fold CV (mean ± sd over repeats)\n")
        L.append("| model | MAPE % | MAE | RMSE | R² | max APE % |")
        L.append("|---|---|---|---|---|---|")
        for _, r in s.iterrows():
            star = " **(selected)**" if r["selected"] else ""
            L.append(f"| {r['model']}{star} | {r['MAPE_%']:.2f} ± {r['MAPE_%_sd']:.2f} | {r['MAE']:.2f} | {r['RMSE']:.2f} | "
                     f"{r['R2']:.4f} | {r['MaxAPE_%']:.1f} |")
        c = pred.calibrators[t]
        L.append(f"\n{int((1-pred.alpha)*100)}% conformal prediction interval: prediction × [{np.exp(c.q_lo_):.3f}, {np.exp(c.q_hi_):.3f}]")
    (d / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"  report  {(d / 'report.md').relative_to(ROOT)}")


def train_all(paths: list[Path] | None = None, repeats: int = 10, folds: int = 5, alpha: float = 0.10):
    if not paths:
        dd = ROOT / "data"
        paths = sorted([*dd.glob("*.csv"), *dd.glob("*.xlsx")])
        if not paths:
            paths = sorted(dd.glob("*.pdf"))
    df = load_all(paths)
    out = {}
    for exam, g in df.groupby("exam"):
        if len(g) < 15:
            print(f"\n=== {exam}: only {len(g)} rows – need ≥15 to train, skipped")
            continue
        out[exam] = train_exam(g.reset_index(drop=True), exam, repeats, folds, alpha, ROOT / "models", ROOT / "reports")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="*", help="data files (default: all csv/xlsx in ./data, else pdf)")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.10, help="1 - interval coverage")
    a = ap.parse_args()
    train_all([Path(p) for p in a.data] if a.data else None, a.repeats, a.folds, a.alpha)


if __name__ == "__main__":
    main()
