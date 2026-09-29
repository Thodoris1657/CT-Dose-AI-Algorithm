"""CT Dose AI — predict CTDIvol / DLP and recommend dose-reduction settings.

Run:  streamlit run app.py      (or simply:  python app.py  — it relaunches itself with Streamlit)
"""
from __future__ import annotations

import sys

if __name__ == "__main__":
    # `python app.py` → relaunch properly through Streamlit's server (opens the browser)
    from streamlit import runtime as _rt
    if not _rt.exists():
        import subprocess
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", __file__, *sys.argv[1:]]))

import contextlib
import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ctdose.data import FEATURES, TARGETS, load_table
from ctdose.optimizer import OptimizerConfig, dose_curves, min_kv_for_bmi, recommend
from ctdose.predictor import DosePredictor, available_models
from ctdose.ui_learning import render_clinical, render_phantom, learned_gamma_options

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------- style
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
KV_RAMP = {70: "#c6dcf7", 80: "#9ec5f4", 90: "#7bb0ee", 100: "#5a9be6", 120: "#2a78d6", 140: "#1a4f94"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "rgba(0,0,0,0.08)"


def style(fig: go.Figure, h=380, **kw):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(color=INK2, size=13), hovermode="closest",
                      legend=dict(orientation="h", y=1.08, x=0), **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0.25)")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0.25)")
    return fig


st.set_page_config(page_title="CT Dose AI", page_icon="🩻", layout="wide")
st.markdown("""<style>
div[data-testid="stMetricValue"]{font-size:2.0rem}
.small{color:#52514e;font-size:0.85rem}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_pred(path: str, mtime: float) -> DosePredictor:  # mtime busts cache after retraining
    return DosePredictor.load(path)


def models_need_training() -> str | None:
    """Return a reason if the saved models are missing or were built with another scikit-learn version."""
    import warnings
    import sklearn
    ms = available_models()
    if not ms:
        return "no trained model found"
    for p in ms.values():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = DosePredictor.load(p)
            if getattr(m, "sklearn_version", "") != sklearn.__version__:
                return f"models were built with scikit-learn {getattr(m, 'sklearn_version', '?') or '?'}, you have {sklearn.__version__}"
        except Exception as e:  # noqa: BLE001
            return f"model file could not be loaded ({type(e).__name__})"
    return None


if "trained_checked" not in st.session_state:
    why = models_need_training()
    if why:
        from train import train_all
        with st.spinner(f"First run: training the models ({why}). This takes about a minute…"):
            with contextlib.redirect_stdout(io.StringIO()):
                train_all(repeats=3)
        st.cache_resource.clear()
    st.session_state.trained_checked = True

models = available_models()
st.sidebar.title("🩻 CT Dose AI")
if not models:
    st.warning("No trained model found. Put data in the data folder and run `python train.py` (or use the **Train** tab).")
exam = st.sidebar.selectbox("Exam / protocol", list(models) or ["—"])
pred = load_pred(str(models[exam]), models[exam].stat().st_mtime) if models else None

st.sidebar.markdown("### Patient & protocol")
d0 = pred.train_df.median(numeric_only=True) if pred else pd.Series(dict(BMI=27, kVp=120, mAs=750, scan_length_cm=30))
BMI = st.sidebar.number_input("BMI (kg/m²) — fixed", 12.0, 80.0, float(round(d0.BMI, 1)), 0.1)
kVp = st.sidebar.select_slider("kVp", options=[70, 80, 90, 100, 110, 120, 130, 140, 150], value=120)
mAs = st.sidebar.number_input("mAs", 10.0, 3000.0, float(round(d0.mAs)), 5.0)
L = st.sidebar.number_input("Scan length (cm)", 2.0, 150.0, float(round(d0.scan_length_cm, 1)), 0.1)
st.sidebar.markdown("### Reference level")
drl = st.sidebar.number_input("DLP reference (mGy·cm)", 50, 5000, 400, 10,
                              help="PROTECTION VI suggests 400 mGy·cm for CCTA. Set your national/local DRL.")

tab_pred, tab_pre, tab_opt, tab_bo, tab_clin, tab_batch, tab_model, tab_train = st.tabs(
    ["Predict", "Pre-scan (BMI only)", "Reduce dose", "Phantom optimisation (BO)", "Clinical learning (bandit)",
     "Batch", "Model & data", "Train / add exams"])

# =================================================================== PREDICT
with tab_pred:
    if pred:
        r = pred.predict_one(BMI, kVp, mAs, L)
        st.subheader(f"{exam}: predicted dose")
        c1, c2, c3 = st.columns(3)
        pct = int((1 - pred.alpha) * 100)
        c1.metric("CTDIvol (mGy)", f"{r['CTDIvol']:.1f}")
        c1.markdown(f"<span class=small>{pct}% interval {r['CTDIvol_lo']:.1f} – {r['CTDIvol_hi']:.1f}</span>", unsafe_allow_html=True)
        c2.metric("DLP (mGy·cm)", f"{r['DLP']:.0f}", delta=f"{(r['DLP']/drl-1)*100:+.0f}% vs reference",
                  delta_color="inverse")
        c2.markdown(f"<span class=small>{pct}% interval {r['DLP_lo']:.0f} – {r['DLP_hi']:.0f}</span>", unsafe_allow_html=True)
        c3.metric("Implied length DLP/CTDI (cm)", f"{r['DLP']/r['CTDIvol']:.1f}")
        c3.markdown(f"<span class=small>Suggested min kV for this BMI: {min_kv_for_bmi(BMI, OptimizerConfig()):g}</span>",
                    unsafe_allow_html=True)
        if not r["in_domain"]:
            st.warning("Outside the training data: " + r["domain_note"])

        st.markdown("#### How dose responds to mAs and kV (this patient)")
        kvs = sorted({80, 100, 120, 140, int(kVp)})
        curves = dose_curves(pred, BMI, L, kvs=kvs, mas_range=(50, max(1500, mAs * 1.2)))
        colA, colB = st.columns(2)
        for col, t, unit in ((colA, "CTDIvol", "mGy"), (colB, "DLP", "mGy·cm")):
            fig = go.Figure()
            for kv in kvs:
                c = curves[curves.kVp == kv]
                fig.add_trace(go.Scatter(x=c.mAs, y=c[t], mode="lines", name=f"{kv} kV",
                                         line=dict(width=2, color=KV_RAMP.get(kv, BLUE)),
                                         hovertemplate=f"{kv} kV<br>%{{x:.0f}} mAs → %{{y:.1f}} {unit}<extra></extra>"))
                fig.add_annotation(x=c.mAs.iloc[-1], y=c[t].iloc[-1], text=f"{kv} kV", showarrow=False,
                                   xanchor="left", font=dict(size=11, color=INK2))
            fig.add_trace(go.Scatter(x=[mAs], y=[r[t]], mode="markers", name="current plan",
                                     marker=dict(size=11, color=ORANGE, line=dict(color="white", width=2)),
                                     error_y=dict(type="data", symmetric=False, array=[r[t + "_hi"] - r[t]],
                                                  arrayminus=[r[t] - r[t + "_lo"]], color=ORANGE),
                                     hovertemplate=f"current: %{{y:.1f}} {unit}<extra></extra>"))
            if t == "DLP":
                fig.add_hline(y=drl, line=dict(color=INK2, dash="dot", width=1),
                              annotation_text=f"reference {drl}", annotation_position="top right")
            style(fig, title=f"{t} ({unit})", xaxis_title="mAs", showlegend=False)
            col.plotly_chart(fig, width="stretch")

# =================================================================== PRE-SCAN
with tab_pre:
    if pred:
        if getattr(pred, "prescan", None) is None:
            st.info("This model was trained before the pre-scan stage existed. Run `python train.py` to add it.")
        else:
            st.subheader("Dose estimate before the scan — from BMI alone")
            st.caption("The Predict tab needs the mAs, which the scanner only chooses once the patient is on the "
                       "table. Here a second, purely statistical model predicts what the tube-current modulation "
                       "will pick for this patient, and the physics law turns that into dose. Useful for "
                       "consenting, planning and triage — with honestly wider intervals.")
            a, b, c = st.columns(3)
            kv_pre = a.selectbox("Planned kVp", [70, 80, 90, 100, 110, 120, 130, 140],
                                 index=[70, 80, 90, 100, 110, 120, 130, 140].index(int(kVp)) if int(kVp) in
                                 [70, 80, 90, 100, 110, 120, 130, 140] else 5)
            know_len = b.toggle("I already know the scan length", False)
            L_known = c.number_input("Scan length (cm)", 2.0, 150.0, float(L), 0.1, disabled=not know_len)
            r = pred.predict_prescan(BMI, kVp=kv_pre, scan_length_cm=L_known if know_len else None)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Expected mAs", f"{r['mAs']:.0f}")
            m1.caption(f"90% range {r['mAs_lo']:.0f} – {r['mAs_hi']:.0f} (ATCM is not deterministic)")
            m2.metric("CTDIvol (mGy)", f"{r['CTDIvol']:.1f}")
            m2.caption(f"90% range {r['CTDIvol_lo']:.1f} – {r['CTDIvol_hi']:.1f}")
            m3.metric("DLP (mGy·cm)", f"{r['DLP']:.0f}", delta=f"{(r['DLP']/drl-1)*100:+.0f}% vs reference",
                      delta_color="inverse")
            m3.caption(f"90% range {r['DLP_lo']:.0f} – {r['DLP_hi']:.0f}")
            m4.metric("Scan length (cm)", f"{r['scan_length_cm']:.1f}")
            m4.caption("predicted from BMI" if r["length_predicted"] else "entered by you")
            if not r["in_domain"]:
                st.warning("Outside the training data: " + (r["domain_note"] or "BMI beyond the observed range"))
            st.caption("Once the scanner has chosen the mAs, the Predict tab gives a much tighter estimate "
                       f"(±{(1-np.exp(pred.calibrators['CTDIvol'].q_lo_))*100:.0f}% on CTDIvol). The width here is "
                       "the price of not yet knowing what the modulation will do.")

            cu = pred.prescan.curve()
            d_tr = pred.train_df
            colA, colB = st.columns(2)
            for col, t, unit, obs in ((colA, "mAs", "mAs", d_tr.mAs), (colB, "scan_length_cm", "cm", d_tr.scan_length_cm)):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=np.r_[cu.BMI, cu.BMI[::-1]], y=np.r_[cu[t + "_hi"], cu[t + "_lo"][::-1]],
                                         fill="toself", fillcolor=BLUE, opacity=0.13, line=dict(width=0),
                                         hoverinfo="skip", showlegend=False))
                fig.add_trace(go.Scatter(x=cu.BMI, y=cu[t], mode="lines", line=dict(color=BLUE, width=2),
                                         name="model", hovertemplate="BMI %{x:.1f} → %{y:.0f}<extra></extra>"))
                fig.add_trace(go.Scatter(x=d_tr.BMI, y=obs, mode="markers", name="past exams",
                                         marker=dict(size=8, color=INK2, opacity=0.55,
                                                     line=dict(color="white", width=1))))
                fig.add_trace(go.Scatter(x=[BMI], y=[r["mAs"] if t == "mAs" else r["scan_length_cm"]],
                                         mode="markers", name="this patient",
                                         marker=dict(size=13, color=ORANGE, symbol="star",
                                                     line=dict(color="white", width=1.5))))
                style(fig, 340, title=f"{'Tube current-time product' if t == 'mAs' else 'Scan length'} vs BMI "
                                      f"({pred.prescan.selected[t]}, 90% band)", xaxis_title="BMI",
                      yaxis_title=unit, showlegend=(t == "mAs"))
                col.plotly_chart(fig, width="stretch")

            with st.expander("How the pre-scan models were chosen"):
                st.caption("Same competition and cross-validation as the dose models. Unlike the dose law, there is "
                           "no formula here — so flexible models genuinely win.")
                for t in ("mAs", "scan_length_cm"):
                    st.markdown(f"**{t}** — selected: {pred.prescan.selected[t]}")
                    st.dataframe(pred.prescan.summary_table(t).style.format(
                        {"MAPE_%": "{:.2f}", "MAPE_%_sd": "{:.2f}", "MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.3f}"}),
                        width="stretch", hide_index=True)

# =================================================================== OPTIMIZE
with tab_opt:
    if pred:
        st.subheader("What should I change to lower CTDIvol & DLP?")
        st.caption("BMI is fixed (patient property). The engine searches kVp, mAs and scan length, "
                   "keeping estimated iodine CNR relative to the current plan.")
        o1, o2, o3, o4 = st.columns(4)
        tau = o1.slider("Keep CNR at ≥ (% of current)", 70, 110, 100, 5) / 100
        g_opts = learned_gamma_options(pred)
        g_src = o2.selectbox("Image-quality model (γ source)", list(g_opts),
                             help="Iodine contrast ∝ kV^-γ. Literature value, or γ calibrated on your phantom "
                                  "(Phantom optimisation tab) / learned from radiologist scores (Clinical learning tab).")
        gamma = o2.slider("Iodine contrast exponent γ", 0.8, 2.0, float(round(g_opts[g_src], 2)), 0.01)
        cuts = o3.multiselect("Try shortening scan by (cm)", [1, 2, 3, 4, 5], default=[2])
        rule = o4.toggle("Apply BMI-based minimum kV", True,
                         help="PROTECTION VI: 80 kV for BMI<25, 100 kV for 25–30, 120 kV above 30")
        p1, p2, p3 = st.columns(3)
        mas_max = p1.number_input("Max mAs (tube limit)", 100, 5000, 1500, 50)
        kv_opts = p2.multiselect("Available kV on your scanner", [70, 80, 90, 100, 110, 120, 130, 140, 150],
                                 default=[70, 80, 90, 100, 120, 140])
        min_len = p3.number_input("Minimum scan length (cm)", 1.0, 100.0, 10.0, 0.5)

        cfg = OptimizerConfig(kv_options=tuple(sorted(set(kv_opts) | {kVp})), cnr_target=tau, iodine_gamma=gamma,
                              mas_max=mas_max, length_cuts_cm=tuple([0] + cuts), min_length_cm=min_len,
                              apply_bmi_kv_rule=rule)
        res = recommend(pred, BMI, kVp, mAs, L, cfg)
        for m in res["messages"]:
            st.markdown("- " + m)
        from ctdose.ui_learning import clinical_hint
        hint = clinical_hint(pred, exam, BMI, L)
        if hint:
            st.success(hint)
        if pred.audit.get("scan_length_is_z_coverage_suspect"):
            st.info("Data check: in the training data DLP/CTDIvol does **not** scale with the 'scan length' field "
                    "(it tracks BMI instead). If that field is patient size/FOV rather than z-coverage, the "
                    "scan-length savings above use the physics rule DLP = CTDIvol × irradiated length.")

        cand = res["candidates"].copy()
        cur_dlp = res["current"]["DLP"]
        cand = cand[(cand.DLP <= cur_dlp * 1.01) | (cand.kind == "current")]   # hide options that raise dose
        cand = cand.sort_values("DLP").reset_index(drop=True)
        fig = go.Figure()
        colors = [AQUA if f else "rgba(82,81,78,0.35)" for f in cand.feasible]
        fig.add_trace(go.Bar(y=cand.option, x=cand.DLP, orientation="h", marker=dict(color=colors, cornerradius=4),
                             customdata=cand.assign(CNR_pct=cand.CNR_rel * 100)[["CTDIvol", "CNR_pct", "mAs", "flags"]].to_numpy(object),
                             hovertemplate="%{y}<br>DLP %{x:.0f} mGy·cm<br>CTDIvol %{customdata[0]:.1f} mGy"
                                           "<br>mAs %{customdata[2]:.0f}<br>CNR %{customdata[1]:.0f}%"
                                           "<br>%{customdata[3]}<extra></extra>",
                             text=[f"{v:.0f}" for v in cand.DLP], textposition="outside", cliponaxis=False))
        fig.add_vline(x=cur_dlp, line=dict(color=ORANGE, width=2), annotation_text="current",
                      annotation_position="top")
        fig.add_vline(x=drl, line=dict(color=INK2, dash="dot", width=1), annotation_text="reference",
                      annotation_position="top")
        style(fig, h=max(320, 28 * len(cand) + 80), title="Predicted DLP per option (green = meets all constraints)",
              xaxis_title="DLP (mGy·cm)", showlegend=False, bargap=0.25)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch")
        show = cand[["option", "kVp", "mAs", "scan_length_cm", "CTDIvol", "DLP", "CTDI_change_pct",
                     "DLP_change_pct", "CNR_rel", "feasible", "flags"]]
        st.dataframe(show.style.format({"CTDIvol": "{:.1f}", "DLP": "{:.0f}", "CTDI_change_pct": "{:+.0f}%",
                                        "DLP_change_pct": "{:+.0f}%", "CNR_rel": "{:.0%}", "mAs": "{:.0f}",
                                        "scan_length_cm": "{:.1f}"}), width="stretch", hide_index=True)

        st.markdown("#### Manual what-if")
        w1, w2, w3 = st.columns(3)
        kv2 = w1.select_slider("kVp ", [70, 80, 90, 100, 110, 120, 130, 140, 150], value=kVp)
        mas2 = w2.slider("mAs ", 20, int(max(mas_max, mAs)), int(mAs), 5)
        L2 = w3.slider("Scan length (cm) ", 2.0, float(max(L * 1.5, 20)), float(L), 0.1)
        a, b = pred.predict_one(BMI, kVp, mAs, L), pred.predict_one(BMI, kv2, mas2, L)
        dlp2 = b["DLP"] * L2 / L
        cnr = (kVp / kv2) ** gamma * np.sqrt(b["CTDIvol"] / a["CTDIvol"])
        m1, m2, m3 = st.columns(3)
        m1.metric("CTDIvol", f"{b['CTDIvol']:.1f} mGy", f"{(b['CTDIvol']/a['CTDIvol']-1)*100:+.0f}%", delta_color="inverse")
        m2.metric("DLP", f"{dlp2:.0f} mGy·cm", f"{(dlp2/a['DLP']-1)*100:+.0f}%", delta_color="inverse")
        m3.metric("Estimated iodine CNR", f"{cnr*100:.0f}% of current")
        if kv2 < min_kv_for_bmi(BMI, cfg):
            st.warning(f"{kv2} kV is below the suggested minimum for BMI {BMI:.1f}.")

# =================================================================== PHANTOM BO
with tab_bo:
    if pred:
        render_phantom(pred, exam)

# =================================================================== CLINICAL BANDIT
with tab_clin:
    if pred:
        render_clinical(pred, exam, BMI, L)

# =================================================================== BATCH
with tab_batch:
    if pred:
        st.subheader("Batch prediction")
        st.caption("Upload CSV / Excel / PDF with columns BMI, kVp (or KV), mAs, scan length. If CTDIvol / DLP "
                   "are included, the app also reports the error.")
        up = st.file_uploader("File", type=["csv", "xlsx", "xls", "pdf"], key="batch")
        if up:
            with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as f:
                f.write(up.getvalue())
            df = load_table(f.name, default_exam=exam).dropna(subset=FEATURES)
            out = pred.predict(df)
            for t in TARGETS:
                if df[t].notna().any():
                    out[t + "_actual"] = df[t].to_numpy()
                    out[t + "_err_%"] = (out[t] / out[t + "_actual"] - 1) * 100
            st.dataframe(out.round(2), width="stretch")
            for t in TARGETS:
                if t + "_err_%" in out:
                    st.write(f"**{t}** MAPE: {out[t + '_err_%'].abs().mean():.2f}%")
            st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode("utf-8-sig"),
                               "predictions.csv", "text/csv")

# =================================================================== MODEL
with tab_model:
    if pred:
        st.subheader(f"Model card — {exam}")
        st.write(f"Trained {pred.trained_at} on **{pred.n_train}** exams "
                 f"({pred.audit['n_unique']} unique). Selected models: "
                 + ", ".join(f"**{t}** → {n}" for t, n in pred.selected.items()))
        st.markdown("**Fitted physics law** — CTDIvol = k · mAs · (kV/120)ⁿ")
        st.json({k: round(v, 5) for k, v in pred.physics_ctdi.params().items()})
        for t in TARGETS:
            s = pred.cv_summary[t]
            st.markdown(f"#### {t}: repeated grouped 5-fold cross-validation")
            st.dataframe(s[["model", "MAPE_%", "MAPE_%_sd", "MAE", "RMSE", "R2", "MaxAPE_%", "selected"]]
                         .style.format({"MAPE_%": "{:.2f}", "MAPE_%_sd": "{:.2f}", "MAE": "{:.2f}", "RMSE": "{:.2f}",
                                        "R2": "{:.4f}", "MaxAPE_%": "{:.1f}"}), width="stretch", hide_index=True)
            y = pred.train_df[t].to_numpy()
            P = pred.oof[t][pred.selected[t]].mean(axis=0)
            fig = go.Figure()
            lo, hi = float(min(y.min(), P.min())) * 0.9, float(max(y.max(), P.max())) * 1.05
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=INK2, dash="dot", width=1),
                                     hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=y, y=P, mode="markers", marker=dict(size=9, color=BLUE, line=dict(color="white", width=1.5)),
                                     customdata=pred.train_df[FEATURES].to_numpy(),
                                     hovertemplate="actual %{x:.1f} · predicted %{y:.1f}<br>BMI %{customdata[0]}, "
                                                   "%{customdata[1]} kV, %{customdata[2]} mAs, %{customdata[3]} cm<extra></extra>",
                                     showlegend=False))
            style(fig, h=360, title=f"{t}: out-of-fold prediction vs actual", xaxis_title="actual", yaxis_title="predicted")
            st.plotly_chart(fig, width="stretch")
        st.markdown("#### Data audit")
        st.json(pred.audit)
        rep = ROOT / "reports" / exam / "report.md"
        if rep.exists():
            with st.expander("Full training report"):
                st.markdown(rep.read_text(encoding="utf-8"))

# =================================================================== TRAIN
with tab_train:
    st.subheader("Train or add a new exam type")
    st.caption("Upload data for another exam (CSV/Excel/PDF). Columns: exam (optional), BMI, kVp, mAs, scan length, "
               "CTDIvol, DLP. Each exam type gets its own model. Needs ≥15 exams; 50+ recommended.")
    up2 = st.file_uploader("Training data", type=["csv", "xlsx", "xls", "pdf"], key="train")
    name = st.text_input("Exam name (used if the file has no 'exam' column)", "NEW_EXAM")
    reps = st.slider("CV repeats (more = slower, more stable)", 2, 20, 5)
    if up2 and st.button("Train", type="primary"):
        from train import train_exam  # noqa: E402
        with tempfile.NamedTemporaryFile(suffix=Path(up2.name).suffix, delete=False) as f:
            f.write(up2.getvalue())
        df = load_table(f.name, default_exam=name).dropna(subset=FEATURES + TARGETS)
        save_to = ROOT / "data" / f"{Path(up2.name).stem}.csv"
        df.to_csv(save_to, index=False)
        buf = io.StringIO()
        with st.spinner("Training & cross-validating…"), contextlib.redirect_stdout(buf):
            for ex, g in df.groupby("exam"):
                if len(g) >= 15:
                    train_exam(g.reset_index(drop=True), ex, reps, 5, 0.10, ROOT / "models", ROOT / "reports")
                else:
                    print(f"{ex}: only {len(g)} rows – skipped (need ≥15)")
        st.code(buf.getvalue())
        st.success(f"Done. Data saved to data/{save_to.name}. Reload the page to select the new exam.")
