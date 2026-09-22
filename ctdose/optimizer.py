"""Dose-reduction recommendation engine.

Given a patient (BMI fixed – it is a patient property) and a planned protocol
(kVp, mAs, scan length), search the protocol space for settings that lower
CTDIvol / DLP while keeping estimated image quality.

Image-quality model (iodine CNR, relative to the patient's own current plan)
---------------------------------------------------------------------------
For the SAME patient (same BMI / attenuation):
    noise σ           ∝ 1 / sqrt(CTDIvol)             (quantum noise)
    iodine contrast C ∝ kVp^(−γ),  γ ≈ 1.2–1.5        (closer to the K-edge at low kV)
  ⇒ CNR_rel = (kVp_cur / kVp_new)^γ · sqrt(CTDI_new / CTDI_cur)
To keep CNR_rel ≥ τ:
    CTDI_new ≥ CTDI_cur · τ² · (kVp_new / kVp_cur)^(2γ)
This is why lowering kVp is the most powerful lever in contrast-enhanced CT
(CCTA): at 100 kV the same iodine CNR needs ≈ 40 % less dose than at 120 kV.

Guard-rails
-----------
* Minimum kVp by BMI (photon-starvation risk in large patients).  Default
  follows PROTECTION VI (Stocker et al., EHJ 2018): 80 kV for BMI < 25,
  90–100 kV for 25–30, 120 kV above 30.
* Tube-output limits (mAs min/max) – if the needed mAs exceeds the limit the
  candidate is flagged as not reaching the CNR target.
* Scan length: DLP ∝ irradiated length. Every cm removed saves DLP/L.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .predictor import DosePredictor


@dataclass
class OptimizerConfig:
    kv_options: tuple = (70, 80, 90, 100, 120, 140)
    cnr_target: float = 1.00          # τ: fraction of current CNR to preserve
    iodine_gamma: float = 1.35        # contrast ∝ kV^-γ
    mas_min: float = 20.0
    mas_max: float = 1500.0
    length_cuts_cm: tuple = (0.0,)    # e.g. (0, 1, 2, 3)
    min_length_cm: float = 10.0
    apply_bmi_kv_rule: bool = True
    bmi_kv_rule: tuple = ((25.0, 80), (30.0, 100), (99.0, 120))   # (BMI upper bound, min kV)
    mas_step: float = 5.0


def min_kv_for_bmi(bmi: float, cfg: OptimizerConfig) -> float:
    for ub, kv in cfg.bmi_kv_rule:
        if bmi < ub:
            return kv
    return cfg.bmi_kv_rule[-1][1]


def cnr_rel(kv_cur, ctdi_cur, kv_new, ctdi_new, gamma):
    return (kv_cur / kv_new) ** gamma * np.sqrt(ctdi_new / ctdi_cur)


def recommend(pred: DosePredictor, BMI: float, kVp: float, mAs: float, scan_length_cm: float,
              cfg: OptimizerConfig | None = None) -> dict:
    cfg = cfg or OptimizerConfig()
    cur = pred.predict_one(BMI, kVp, mAs, scan_length_cm)
    c0, d0 = cur["CTDIvol"], cur["DLP"]
    kv_min = min_kv_for_bmi(BMI, cfg) if cfg.apply_bmi_kv_rule else min(cfg.kv_options)
    n_exp = pred.physics_ctdi.kv_exponent_

    rows = []

    def add(label, kv, mas_new, L_new, kind):
        mas_c = float(np.clip(np.round(mas_new / cfg.mas_step) * cfg.mas_step, cfg.mas_min, cfg.mas_max))
        p = pred.predict_one(BMI, kv, mas_c, scan_length_cm)
        ctdi = p["CTDIvol"]
        dlp = p["DLP"] * (L_new / scan_length_cm)       # DLP ∝ irradiated length
        cnr = cnr_rel(kVp, c0, kv, ctdi, cfg.iodine_gamma)
        flags = []
        if kv < kv_min:
            flags.append(f"kV below BMI minimum ({kv_min:g} kV)")
        if mas_c >= cfg.mas_max - 1e-6 and mas_new > cfg.mas_max:
            flags.append("tube limit reached")
        if cnr < cfg.cnr_target - 0.02:
            flags.append("CNR below target")
        if not p["in_domain"]:
            flags.append("extrapolation")
        rows.append(dict(option=label, kind=kind, kVp=kv, mAs=mas_c, scan_length_cm=L_new,
                         CTDIvol=ctdi, DLP=dlp, CTDI_change_pct=(ctdi / c0 - 1) * 100,
                         DLP_change_pct=(dlp / d0 - 1) * 100, CNR_rel=cnr,
                         feasible=not any(f for f in flags if f != "extrapolation"),
                         flags="; ".join(flags)))

    # 1) iso-CNR kV switching (+ optional length cuts)
    for kv in cfg.kv_options:
        target_ctdi = c0 * cfg.cnr_target ** 2 * (kv / kVp) ** (2 * cfg.iodine_gamma)
        mas_new = pred.mas_for_ctdi(target_ctdi, kv, BMI, scan_length_cm)
        for cut in cfg.length_cuts_cm:
            L_new = max(cfg.min_length_cm, scan_length_cm - cut)
            lab = f"{kv:g} kV, iso-CNR" + (f", −{cut:g} cm" if cut else "")
            add(lab, kv, mas_new, L_new, "kV + mAs (iso-CNR)" if kv != kVp else ("current" if not cut else "length"))

    # 2) peer benchmark at the current kV (local achievable dose for this BMI)
    peer = pred.peer_ctdi(BMI) if pred.peer_models else {}
    if peer:
        for q, lab in ((0.5, "peer median"), (0.25, "peer 25th pct (achievable)")):
            tgt = peer[q] * (kVp / 120.0) ** n_exp
            if tgt < c0 * 0.98:
                add(f"{kVp:g} kV, mAs → {lab}", kVp, pred.mas_for_ctdi(tgt, kVp, BMI, scan_length_cm),
                    scan_length_cm, "mAs (peer benchmark)")

    cand = pd.DataFrame(rows).drop_duplicates(subset=["kVp", "mAs", "scan_length_cm"]).reset_index(drop=True)
    feas = cand[cand.feasible & (cand.kind != "mAs (peer benchmark)")]
    best = feas.sort_values("DLP").iloc[0].to_dict() if len(feas) else None

    # ---------------- narrative
    msgs = []
    if best and best["DLP"] < d0 * 0.99:
        msgs.append(
            f"Switch to **{best['kVp']:g} kV** and **{best['mAs']:g} mAs**"
            + (f" with scan length **{best['scan_length_cm']:g} cm**" if best['scan_length_cm'] != scan_length_cm else "")
            + f": CTDIvol {c0:.1f} → {best['CTDIvol']:.1f} mGy ({best['CTDI_change_pct']:+.0f}%), "
              f"DLP {d0:.0f} → {best['DLP']:.0f} mGy·cm ({best['DLP_change_pct']:+.0f}%), "
              f"estimated iodine CNR kept at {best['CNR_rel']*100:.0f}% of the current plan.")
        if best["kVp"] not in pred.audit.get("kVp_levels", []):
            msgs.append(f"Note: {best['kVp']:g} kV does not appear in the training data – this estimate relies on the "
                        f"fitted physics kV-law (n = {n_exp:.2f}). Validate with the first scans / a phantom.")
    else:
        msgs.append(f"{kVp:g} kV is already the lowest kV recommended for BMI {BMI:.1f} – the remaining levers are mAs "
                    f"and scan length (see below).")
    if kVp < kv_min:
        msgs.append(f"⚠ Current {kVp:g} kV is below the suggested minimum for BMI {BMI:.1f} ({kv_min:g} kV): risk of photon starvation / high noise.")
    if peer:
        p_now = c0 / (kVp / 120.0) ** n_exp
        rel = (p_now / peer[0.5] - 1) * 100
        msgs.append(f"Local benchmark: at BMI {BMI:.1f} the median 120-kV-equivalent CTDIvol in your data is "
                    f"{peer[0.5]:.1f} mGy (25th pct {peer[0.25]:.1f}); this plan is at {p_now:.1f} mGy ({rel:+.0f}% vs median).")
        pm = cand[cand.kind == "mAs (peer benchmark)"]
        if len(pm):
            r0 = pm.sort_values("DLP", ascending=False).iloc[0]
            msgs.append(f"Tube output is above your own peers: lowering mAs to **{r0['mAs']:g}** (peer median) gives "
                        f"CTDIvol {r0['CTDIvol']:.1f} mGy ({r0['CTDI_change_pct']:+.0f}%), DLP {r0['DLP']:.0f} mGy·cm "
                        f"({r0['DLP_change_pct']:+.0f}%). Estimated CNR {r0['CNR_rel']*100:.0f}% of current – the level "
                        f"that peers with the same BMI were scanned at.")
    per_cm = d0 / scan_length_cm
    msgs.append(f"Scan length: each 1 cm less coverage saves ≈ {per_cm:.0f} mGy·cm DLP ({100/scan_length_cm:.1f}%) – "
                f"CTDIvol is unaffected by length.")
    return {"current": cur, "kv_min": kv_min, "candidates": cand, "best": best, "messages": msgs, "config": cfg}


def dose_curves(pred: DosePredictor, BMI: float, scan_length_cm: float, kvs=(80, 100, 120, 140),
                mas_range=(100, 1500), n=60) -> pd.DataFrame:
    """Model-predicted CTDIvol / DLP vs mAs for several kV values (for plotting)."""
    grid = np.linspace(*mas_range, n)
    out = []
    for kv in kvs:
        df = pd.DataFrame(dict(BMI=BMI, kVp=kv, mAs=grid, scan_length_cm=scan_length_cm))
        p = pred.predict(df)
        p["kVp"] = kv
        out.append(p)
    return pd.concat(out, ignore_index=True)
