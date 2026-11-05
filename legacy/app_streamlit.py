"""CT Dose AI — predict the radiation dose of a CT scan and see how to lower it.

Run:  streamlit run app.py      (or simply:  python app.py  — it relaunches itself with Streamlit)

The interface has two modes:
  * Simple  (default) — four tabs, plain language, everything explained on screen.
  * Expert            — adds the phantom loop, the clinical-learning loop, batch mode,
                        the model card and training.
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
from ctdose.ui_learning import clinical_hint, learned_gamma_options, render_clinical, render_phantom

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------- style
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
KV_RAMP = {70: "#c6dcf7", 80: "#9ec5f4", 90: "#7bb0ee", 100: "#5a9be6", 120: "#2a78d6", 140: "#1a4f94"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "rgba(0,0,0,0.08)"
KV_CHOICES = [70, 80, 90, 100, 110, 120, 130, 140, 150]


def style(fig: go.Figure, h=380, **kw):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(color=INK2, size=13), hovermode="closest",
                      legend=dict(orientation="h", y=1.08, x=0), **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0.25)")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0.25)")
    return fig


st.set_page_config(page_title="CT Dose AI", page_icon="🩻", layout="wide")
st.markdown("""<style>
div[data-testid="stMetricValue"]{font-size:1.9rem}
div[data-testid="stMetricLabel"] p{font-size:0.92rem}
.small{color:#52514e;font-size:0.85rem}
.big-advice{font-size:1.15rem; line-height:1.5}
</style>""", unsafe_allow_html=True)


# ============================================================ model loading
@st.cache_resource
def load_pred(path: str, mtime: float) -> DosePredictor:  # mtime busts the cache after retraining
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
                return (f"the models were built with scikit-learn {getattr(m, 'sklearn_version', '?') or '?'}, "
                        f"you have {sklearn.__version__}")
        except Exception as e:  # noqa: BLE001
            return f"the model file could not be opened ({type(e).__name__})"
    return None


if "trained_checked" not in st.session_state:
    why = models_need_training()
    if why:
        from train import train_all
        with st.spinner(f"First run — training the models, because {why}. This takes about a minute. "
                        "It only happens once."):
            with contextlib.redirect_stdout(io.StringIO()):
                train_all(repeats=3)
        st.cache_resource.clear()
    st.session_state.trained_checked = True

models = available_models()

# ============================================================ sidebar
st.sidebar.title("🩻 CT Dose AI")
st.sidebar.caption("Radiation dose for CT, predicted and optimised.")

mode = st.sidebar.radio("Mode", ["Simple", "Expert"], index=0,
                        help="Simple shows the three everyday screens. Expert adds the phantom loop, the "
                             "learning loop, batch processing, the model card and training.")
expert = mode == "Expert"

exam = st.sidebar.selectbox("Examination", list(models) or ["—"],
                            help="Each examination type has its own model, trained on that exam's own data.")
pred = load_pred(str(models[exam]), models[exam].stat().st_mtime) if models else None

with st.sidebar.expander("Settings"):
    drl = st.number_input("Reference level for total dose (mGy·cm)", 50, 5000, 400, 10,
                          help="The dose level your department compares against. 400 mGy·cm is the published "
                               "value for coronary CT angiography. Replace it with your national or local one.")
    st.caption("Everything runs on this computer. No patient data is uploaded anywhere.")

if not models:
    st.error("No trained model found yet. Put a data file in the **data** folder and run `python train.py`, "
             "or use the **Train** tab in Expert mode.")
    st.stop()

# ============================================================ header + first-run guide
st.title("CT Dose AI")
st.markdown("Enter the patient and the planned scan. The app predicts the radiation dose, and shows you what to "
            "change to lower it without losing image quality.")

if not st.session_state.get("guide_dismissed"):
    with st.container(border=True):
        g1, g2 = st.columns([6, 1])
        with g1:
            st.markdown("""
**New here? It works in three steps.**

1. **Fill the box below** — the patient's BMI and the three scanner settings (tube voltage, tube current-time
   product, scan length). Not sure? Press *Load a typical patient*.
2. **Read tab 1** — the predicted dose, and whether it is above or below your reference level.
3. **Open tab 2** — the single change the app recommends, and how much dose it saves.

Tab 3 does the same *before* the scan, when only the BMI is known. The **Help** tab explains every term.
            """)
        with g2:
            st.write("")
            if st.button("Got it", width="stretch"):
                st.session_state.guide_dismissed = True
                st.rerun()

# ============================================================ the input box
d0 = pred.train_df.median(numeric_only=True)
defaults = dict(BMI=float(round(d0.BMI, 1)), kVp=120, mAs=float(round(d0.mAs)), L=float(round(d0.scan_length_cm, 1)))
for k, v in defaults.items():
    st.session_state.setdefault(f"in_{k}", v)


def load_example():
    """Reset the input boxes to the typical (median) patient of the training data."""
    for key, val in defaults.items():
        st.session_state[f"in_{key}"] = val

with st.container(border=True):
    st.markdown("#### The patient and the planned scan")
    c1, c2, c3, c4, c5 = st.columns([1.1, 1.1, 1.1, 1.1, 0.9])
    BMI = c1.number_input("BMI (kg/m²)", 12.0, 80.0, step=0.1, key="in_BMI",
                          help="Body mass index = weight ÷ height². The app cannot change it — it is a property "
                               "of the patient — but everything else is adapted to it.")
    kVp = c2.selectbox("Tube voltage — kV", KV_CHOICES, key="in_kVp",
                       help="How penetrating the X-ray beam is. Lower kV gives more contrast from the iodine and "
                            "much less dose, but more noise in large patients.")
    mAs = c3.number_input("Tube current-time — mAs", 10.0, 3000.0, step=5.0, key="in_mAs",
                          help="How many X-rays are used. Dose is exactly proportional to it. On most scanners "
                               "the automatic modulation chooses this value from the patient's size.")
    L = c4.number_input("Scan length (cm)", 2.0, 150.0, step=0.1, key="in_L",
                        help="The length of the body that is irradiated. Total dose (DLP) is proportional to it.")
    with c5:
        st.write("")
        # NOTE: the values must be written in a callback. A callback runs *before* the next redraw,
        # so the widgets are not yet instantiated and Streamlit allows the change.
        st.button("Load a typical patient", width="stretch", on_click=load_example,
                  help="Fills the boxes with the median values from the training data, so you can try the app.")

# ============================================================ tabs
simple_labels = ["1 · Dose of this scan", "2 · How to lower it", "3 · Before the scan", "Help"]
expert_labels = ["Phantom testing", "Learning from doctors", "Many scans at once", "Model details", "Add data"]
tabs = st.tabs(simple_labels + (expert_labels if expert else []))
tab_now, tab_low, tab_pre, tab_help = tabs[:4]


def verdict(dlp: float) -> tuple[str, str]:
    """Plain-language status versus the reference level."""
    r = dlp / drl
    if r <= 1.0:
        return "ok", f"This is **below** your reference level of {drl:g} mGy·cm."
    if r <= 1.5:
        return "warn", f"This is **{r:.1f}× your reference level** of {drl:g} mGy·cm. Worth reviewing."
    return "bad", (f"This is **{r:.1f}× your reference level** of {drl:g} mGy·cm. Tab 2 shows how to bring it down.")


# =================================================================== 1 · DOSE NOW
with tab_now:
    r = pred.predict_one(BMI, kVp, mAs, L)
    st.markdown("#### The predicted dose")
    st.caption("Two numbers, both taken from the scanner's own dose report in the training data. "
               "The range below each one holds the true value 9 times out of 10.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Dose per slice — CTDIvol", f"{r['CTDIvol']:.1f} mGy")
    c1.caption(f"Range {r['CTDIvol_lo']:.1f} – {r['CTDIvol_hi']:.1f} mGy · how strong the beam is at each slice")
    c2.metric("Total dose — DLP", f"{r['DLP']:.0f} mGy·cm", delta=f"{(r['DLP']/drl-1)*100:+.0f}% vs reference",
              delta_color="inverse")
    c2.caption(f"Range {r['DLP_lo']:.0f} – {r['DLP_hi']:.0f} · the whole examination")
    k_card = 0.026
    c3.metric("Roughly, in patient terms", f"≈ {r['DLP']*k_card:.1f} mSv")
    c3.caption("Effective dose, using the cardiac factor 0.026 mSv per mGy·cm. Natural background is ≈2.4 mSv/year.")

    status, msg = verdict(r["DLP"])
    (st.success if status == "ok" else st.warning if status == "warn" else st.error)(msg)

    if not r["in_domain"]:
        st.info("⚠ Note: " + r["domain_note"] + ". The estimate still follows the physics, but no past exam in "
                "your data looks like this, so treat it as an extrapolation.")

    with st.expander("Show how the dose changes with mAs and kV for this patient"):
        st.caption("Each line is one tube voltage. The orange dot is the scan you entered. Moving down a line "
                   "means less mAs; moving to a lighter line means lower kV.")
        kvs = sorted({80, 100, 120, 140, int(kVp)})
        curves = dose_curves(pred, BMI, L, kvs=kvs, mas_range=(50, max(1500, mAs * 1.2)))
        colA, colB = st.columns(2)
        for col, t, unit, nice in ((colA, "CTDIvol", "mGy", "Dose per slice"), (colB, "DLP", "mGy·cm", "Total dose")):
            fig = go.Figure()
            for kv in kvs:
                c = curves[curves.kVp == kv]
                fig.add_trace(go.Scatter(x=c.mAs, y=c[t], mode="lines", name=f"{kv} kV",
                                         line=dict(width=2, color=KV_RAMP.get(kv, BLUE)),
                                         hovertemplate=f"{kv} kV<br>%{{x:.0f}} mAs → %{{y:.1f}} {unit}<extra></extra>"))
                fig.add_annotation(x=c.mAs.iloc[-1], y=c[t].iloc[-1], text=f"{kv} kV", showarrow=False,
                                   xanchor="left", font=dict(size=11, color=INK2))
            fig.add_trace(go.Scatter(x=[mAs], y=[r[t]], mode="markers", name="your scan",
                                     marker=dict(size=11, color=ORANGE, line=dict(color="white", width=2)),
                                     error_y=dict(type="data", symmetric=False, array=[r[t + "_hi"] - r[t]],
                                                  arrayminus=[r[t] - r[t + "_lo"]], color=ORANGE),
                                     hovertemplate=f"your scan: %{{y:.1f}} {unit}<extra></extra>"))
            if t == "DLP":
                fig.add_hline(y=drl, line=dict(color=INK2, dash="dot", width=1),
                              annotation_text=f"reference {drl}", annotation_position="top right")
            style(fig, title=f"{nice} ({unit})", xaxis_title="mAs", showlegend=False)
            col.plotly_chart(fig, width="stretch")

    with st.expander("Where do these numbers come from?"):
        p = pred.physics_ctdi.params()
        st.markdown(f"""
The app fitted your scanner's own dose law on **{pred.n_train} past examinations**:

`dose per slice = {np.exp(pred.physics_ctdi.intercept_):.4f} × mAs × (kV / 120) ^ {pred.physics_ctdi.kv_exponent_:.2f}`

and `total dose = dose per slice × scan length`.

Tested against eight other models (including a neural network), this one was the most accurate, with a typical
error of **{pred.cv_summary['CTDIvol'].loc[pred.cv_summary['CTDIvol'].selected, 'MAPE_%'].iloc[0]:.1f}%** for the
dose per slice and
**{pred.cv_summary['DLP'].loc[pred.cv_summary['DLP'].selected, 'MAPE_%'].iloc[0]:.1f}%** for the total dose.
Switch to **Expert** mode in the sidebar to see the full comparison.
        """)

# =================================================================== 2 · LOWER IT
with tab_low:
    st.markdown("#### What to change")
    st.caption("The app keeps the estimated image quality the same as your current plan, and looks for the "
               "settings that deliver it with less dose. The BMI is never changed.")

    with st.expander("Scanner limits and safety rules (change these to match your scanner)"):
        a, b, c = st.columns(3)
        kv_opts = a.multiselect("kV values your scanner offers", KV_CHOICES, default=[80, 100, 120, 140],
                                help="Only these will be proposed.")
        mas_max = b.number_input("Largest mAs the tube can deliver", 100, 5000, 1500, 50)
        cuts = c.multiselect("Also try shortening the scan by (cm)", [1, 2, 3, 4, 5], default=[2],
                             help="Total dose is proportional to the scan length, so trimming the coverage is "
                                  "the simplest saving of all.")
        d, e, f = st.columns(3)
        tau = d.slider("Image quality to keep (% of now)", 70, 110, 100, 5,
                       help="100% means: do not lose any estimated contrast-to-noise. Lower it only after a "
                            "phantom test or radiologist agreement.") / 100
        rule = e.toggle("Apply the minimum-kV rule for BMI", True,
                        help="80 kV below BMI 25, 100 kV for 25–30, 120 kV above 30. It protects large patients "
                             "from excessive noise.")
        g_opts = learned_gamma_options(pred)
        g_src = f.selectbox("Image-quality model", list(g_opts),
                            help="How much contrast the iodine gains at lower kV. Start with the literature "
                                 "value; replace it with your own once you run a phantom test (Expert mode).")
        gamma = g_opts[g_src]
        min_len = 10.0

    cfg = OptimizerConfig(kv_options=tuple(sorted(set(kv_opts) | {kVp})), cnr_target=tau, iodine_gamma=gamma,
                          mas_max=mas_max, length_cuts_cm=tuple([0] + cuts), min_length_cm=min_len,
                          apply_bmi_kv_rule=rule)
    res = recommend(pred, BMI, kVp, mAs, L, cfg)
    best, cur = res["best"], res["current"]

    if best and best["DLP"] < cur["DLP"] * 0.99:
        with st.container(border=True):
            st.markdown("##### ✅ Recommended change")
            m1, m2, m3 = st.columns([1.4, 1, 1])
            m1.markdown(f"<div class='big-advice'>Set <b>{best['kVp']:g} kV</b> and <b>{best['mAs']:.0f} mAs</b>"
                        + (f", and shorten the scan to <b>{best['scan_length_cm']:.1f} cm</b>"
                           if abs(best["scan_length_cm"] - L) > 0.05 else "")
                        + "</div>", unsafe_allow_html=True)
            m1.caption(f"from {kVp:g} kV, {mAs:.0f} mAs, {L:.1f} cm")
            m2.metric("Dose per slice", f"{best['CTDIvol']:.1f} mGy", f"{best['CTDI_change_pct']:+.0f}%",
                      delta_color="inverse")
            m3.metric("Total dose", f"{best['DLP']:.0f} mGy·cm", f"{best['DLP_change_pct']:+.0f}%",
                      delta_color="inverse")
            st.caption(f"Estimated image quality: {best['CNR_rel']*100:.0f}% of the current plan "
                       f"(100% means unchanged).")
    else:
        st.info("With the current limits, the app finds no setting that lowers the dose while keeping the image "
                "quality. Try allowing a lower kV, or shortening the scan.")

    # ---------------------------------------------------------- dose vs image quality
    st.markdown("##### What the dose costs you in image quality")
    st.caption("Left: how much contrast-to-noise you keep at each dose. Right: how likely the image is still "
               "diagnostic. Both are for this patient; the orange dot is your current scan.")
    qcol1, qcol2 = st.columns(2)

    kv_show = sorted({int(kVp)} | {int(k) for k in kv_opts})[:5]
    cur_ctdi, cur_dlp = cur["CTDIvol"], cur["DLP"]
    dlp_per_ctdi = cur_dlp / cur_ctdi            # = irradiated length, keeps both charts on the same x-axis

    # --- left: estimated image quality (CNR) against dose
    fig_q = go.Figure()
    for kv in kv_show:
        ctdi_grid = np.geomspace(cur_ctdi * 0.15, cur_ctdi * 1.6, 70)
        cnr_rel = (kVp / kv) ** gamma * np.sqrt(ctdi_grid / cur_ctdi) * 100
        fig_q.add_trace(go.Scatter(x=ctdi_grid * dlp_per_ctdi, y=cnr_rel, mode="lines", name=f"{kv} kV",
                                   line=dict(width=2, color=KV_RAMP.get(kv, BLUE)),
                                   hovertemplate=f"{kv} kV<br>%{{x:.0f}} mGy·cm → image quality %{{y:.0f}}%"
                                                 "<extra></extra>"))
    fig_q.add_hline(y=tau * 100, line=dict(color=AQUA, dash="dash", width=1.5),
                    annotation_text=f"your target {tau*100:.0f}%", annotation_position="bottom right")
    fig_q.add_trace(go.Scatter(x=[cur_dlp], y=[100], mode="markers", name="your scan",
                               marker=dict(size=12, color=ORANGE, line=dict(color="white", width=2)),
                               hovertemplate="your scan<extra></extra>"))
    if best:
        fig_q.add_trace(go.Scatter(x=[best["DLP"]], y=[best["CNR_rel"] * 100], mode="markers", name="recommended",
                                   marker=dict(size=15, color=AQUA, symbol="star",
                                               line=dict(color="white", width=1.5)),
                                   hovertemplate="recommended<extra></extra>"))
    style(fig_q, 380, title="Image quality kept, versus total dose", xaxis_title="total dose (mGy·cm)",
          yaxis_title="% of the image quality you have now")
    fig_q.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.22, x=0))
    fig_q.update_yaxes(range=[40, 130])
    qcol1.plotly_chart(fig_q, width="stretch")
    qcol1.caption("Moving left along one line means less mAs: dose falls faster than quality, but it does fall. "
                  "Jumping to a lighter line means lower kV, which buys quality back because iodine shows up "
                  "better — that is why the recommendation sits left of your scan at the same height.")

    # --- right: probability the image is still diagnostic
    from ctdose.bandit import BanditConfig, DoseBandit, features as _bfeat
    bcfg = BanditConfig(gamma=gamma, kv_options=tuple(kv_show), apply_bmi_kv_rule=rule)
    bandit = DoseBandit(pred, bcfg, log=DoseBandit.load_log(exam), seed=0)
    n_scored = int(len(bandit.scored()))
    cr = bandit.ctdi_ref(BMI)
    W = bandit.model.sample(np.random.RandomState(0), 300)
    fig_p = go.Figure()
    for kv in kv_show:
        ctdi_grid = np.geomspace(cur_ctdi * 0.15, cur_ctdi * 1.6, 70)
        F = _bfeat(np.full_like(ctdi_grid, BMI), ctdi_grid, np.full_like(ctdi_grid, kv),
                   np.full_like(ctdi_grid, cr))
        draws = bandit.model.prob(F, W)
        mu = bandit.model.prob(F)[:, 0]
        lo, hi = np.quantile(draws, 0.1, axis=1), np.quantile(draws, 0.9, axis=1)
        col = KV_RAMP.get(kv, BLUE)
        fig_p.add_trace(go.Scatter(x=np.r_[ctdi_grid, ctdi_grid[::-1]] * dlp_per_ctdi, y=np.r_[hi, lo[::-1]],
                                   fill="toself", fillcolor=col, opacity=0.12, line=dict(width=0),
                                   hoverinfo="skip", showlegend=False))
        fig_p.add_trace(go.Scatter(x=ctdi_grid * dlp_per_ctdi, y=mu * 100, mode="lines", name=f"{kv} kV",
                                   line=dict(width=2, color=col),
                                   hovertemplate=f"{kv} kV<br>%{{x:.0f}} mGy·cm → %{{y:.0f}}% chance"
                                                 "<extra></extra>"))
    fig_p.add_hline(y=97, line=dict(color=AQUA, dash="dash", width=1.5), annotation_text="97% target",
                    annotation_position="bottom left")
    fig_p.add_trace(go.Scatter(x=[cur_dlp], y=[float(bandit.model.prob(
        _bfeat([BMI], [cur_ctdi], [kVp], [cr]))[0, 0]) * 100], mode="markers", name="your scan",
        marker=dict(size=14, color=ORANGE, line=dict(color="white", width=2)),
        hovertemplate="your scan<extra></extra>"))
    if best:
        p_best = float(bandit.model.prob(_bfeat([BMI], [best["CTDIvol"]], [best["kVp"]], [cr]))[0, 0]) * 100
        fig_p.add_trace(go.Scatter(x=[best["DLP"]], y=[p_best], mode="markers", name="recommended",
                                   marker=dict(size=17, color=AQUA, symbol="star",
                                               line=dict(color="white", width=1.5)),
                                   hovertemplate="recommended<extra></extra>"))
    style(fig_p, 380, title="Chance the image is still diagnostic", xaxis_title="total dose (mGy·cm)",
          yaxis_title="% chance of a diagnostic image")
    fig_p.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.22, x=0))
    fig_p.update_yaxes(range=[0, 105])
    qcol2.plotly_chart(fig_p, width="stretch")
    qcol2.caption(
        (f"Learned from **{n_scored}** examinations scored by your radiologists, plus the physics prior. "
         if n_scored else
         "No scored examinations yet, so this is the physics prior only: more dose helps, and lower kV helps at "
         "the same dose. ")
        + "The shaded band is the uncertainty. It narrows every time a doctor scores an image in the "
          "*Learning from doctors* tab (Expert mode).")

    with st.expander("Why this change, and what else was considered"):
        for m in res["messages"]:
            st.markdown("- " + m)
        hint = clinical_hint(pred, exam, BMI, L)
        if hint:
            st.success(hint)
        if pred.audit.get("scan_length_is_z_coverage_suspect"):
            st.info("One caution about your data: the 'scan length' column does not behave like the irradiated "
                    "length (it follows the BMI instead). The length savings above assume it really is the "
                    "irradiated length.")

        cand = res["candidates"].copy()
        cand = cand[(cand.DLP <= cur["DLP"] * 1.01) | (cand.kind == "current")].sort_values("DLP").reset_index(drop=True)
        fig = go.Figure()
        colors = [AQUA if f_ else "rgba(82,81,78,0.35)" for f_ in cand.feasible]
        fig.add_trace(go.Bar(y=cand.option, x=cand.DLP, orientation="h", marker=dict(color=colors, cornerradius=4),
                             customdata=cand.assign(CNR_pct=cand.CNR_rel * 100)[
                                 ["CTDIvol", "CNR_pct", "mAs", "flags"]].to_numpy(object),
                             hovertemplate="%{y}<br>total dose %{x:.0f} mGy·cm<br>per slice %{customdata[0]:.1f} mGy"
                                           "<br>%{customdata[2]:.0f} mAs<br>image quality %{customdata[1]:.0f}%"
                                           "<br>%{customdata[3]}<extra></extra>",
                             text=[f"{v:.0f}" for v in cand.DLP], textposition="outside", cliponaxis=False))
        fig.add_vline(x=cur["DLP"], line=dict(color=ORANGE, width=2), annotation_text="your scan",
                      annotation_position="top")
        fig.add_vline(x=drl, line=dict(color=INK2, dash="dot", width=1), annotation_text="reference",
                      annotation_position="top")
        style(fig, h=max(300, 28 * len(cand) + 80),
              title="Total dose of each option — green = allowed by every rule you set", xaxis_title="DLP (mGy·cm)",
              showlegend=False, bargap=0.25)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch")
        show = cand.rename(columns={"option": "Option", "kVp": "kV", "mAs": "mAs", "scan_length_cm": "length (cm)",
                                    "CTDIvol": "per slice (mGy)", "DLP": "total (mGy·cm)",
                                    "DLP_change_pct": "change", "CNR_rel": "image quality",
                                    "feasible": "allowed", "flags": "notes"})
        st.dataframe(show[["Option", "kV", "mAs", "length (cm)", "per slice (mGy)", "total (mGy·cm)", "change",
                           "image quality", "allowed", "notes"]].style.format(
            {"per slice (mGy)": "{:.1f}", "total (mGy·cm)": "{:.0f}", "change": "{:+.0f}%",
             "image quality": "{:.0%}", "mAs": "{:.0f}", "length (cm)": "{:.1f}"}),
            width="stretch", hide_index=True)

    with st.expander("Try your own combination"):
        w1, w2, w3 = st.columns(3)
        kv2 = w1.select_slider("kV", KV_CHOICES, value=kVp)
        mas2 = w2.slider("mAs", 20, int(max(mas_max, mAs)), int(mAs), 5)
        L2 = w3.slider("Scan length (cm)", 2.0, float(max(L * 1.5, 20)), float(L), 0.1)
        a_, b_ = pred.predict_one(BMI, kVp, mAs, L), pred.predict_one(BMI, kv2, mas2, L)
        dlp2 = b_["DLP"] * L2 / L
        cnr = (kVp / kv2) ** gamma * np.sqrt(b_["CTDIvol"] / a_["CTDIvol"])
        m1, m2, m3 = st.columns(3)
        m1.metric("Dose per slice", f"{b_['CTDIvol']:.1f} mGy", f"{(b_['CTDIvol']/a_['CTDIvol']-1)*100:+.0f}%",
                  delta_color="inverse")
        m2.metric("Total dose", f"{dlp2:.0f} mGy·cm", f"{(dlp2/a_['DLP']-1)*100:+.0f}%", delta_color="inverse")
        m3.metric("Estimated image quality", f"{cnr*100:.0f}% of now")
        if kv2 < min_kv_for_bmi(BMI, cfg):
            st.warning(f"{kv2} kV is below the suggested minimum for BMI {BMI:.1f} — expect a noisy image.")

# =================================================================== 3 · BEFORE THE SCAN
with tab_pre:
    st.markdown("#### Before the patient is on the table")
    if getattr(pred, "prescan", None) is None:
        st.info("This model was trained before this feature existed. Run `python train.py` once to add it.")
    else:
        st.caption("Tab 1 needs the mAs, which the scanner only chooses when the patient is already positioned. "
                   "Here the app predicts that choice from the BMI, so you can estimate the dose in advance — "
                   "for consent, planning or an audit. The ranges are wider, because there is more unknown.")
        a, b, c = st.columns(3)
        kv_pre = a.selectbox("Tube voltage you plan to use", KV_CHOICES[:-1],
                             index=KV_CHOICES[:-1].index(int(kVp)) if int(kVp) in KV_CHOICES[:-1] else 5)
        know_len = b.toggle("I already know the scan length", False)
        L_known = c.number_input("Scan length (cm) ", 2.0, 150.0, float(L), 0.1, disabled=not know_len)
        rp = pred.predict_prescan(BMI, kVp=kv_pre, scan_length_cm=L_known if know_len else None)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("mAs the scanner will likely pick", f"{rp['mAs']:.0f}")
        m1.caption(f"range {rp['mAs_lo']:.0f} – {rp['mAs_hi']:.0f}")
        m2.metric("Dose per slice", f"{rp['CTDIvol']:.1f} mGy")
        m2.caption(f"range {rp['CTDIvol_lo']:.1f} – {rp['CTDIvol_hi']:.1f}")
        m3.metric("Total dose", f"{rp['DLP']:.0f} mGy·cm", delta=f"{(rp['DLP']/drl-1)*100:+.0f}% vs reference",
                  delta_color="inverse")
        m3.caption(f"range {rp['DLP_lo']:.0f} – {rp['DLP_hi']:.0f}")
        m4.metric("Scan length", f"{rp['scan_length_cm']:.1f} cm")
        m4.caption("predicted from BMI" if rp["length_predicted"] else "you entered it")
        if not rp["in_domain"]:
            st.info("⚠ This BMI is outside the range of the training data, so the estimate is an extrapolation.")

        with st.expander("Show how this was learned"):
            st.caption("Blue line: what the model expects for each BMI, with its 90% band. Grey dots: the past "
                       "examinations it learned from. Orange star: this patient.")
            cu = pred.prescan.curve()
            d_tr = pred.train_df
            colA, colB = st.columns(2)
            for col, t, unit, obs, nice in ((colA, "mAs", "mAs", d_tr.mAs, "mAs chosen by the scanner"),
                                            (colB, "scan_length_cm", "cm", d_tr.scan_length_cm, "Scan length")):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=np.r_[cu.BMI, cu.BMI[::-1]], y=np.r_[cu[t + "_hi"], cu[t + "_lo"][::-1]],
                                         fill="toself", fillcolor=BLUE, opacity=0.13, line=dict(width=0),
                                         hoverinfo="skip", showlegend=False))
                fig.add_trace(go.Scatter(x=cu.BMI, y=cu[t], mode="lines", line=dict(color=BLUE, width=2),
                                         name="model", hovertemplate="BMI %{x:.1f} → %{y:.0f}<extra></extra>"))
                fig.add_trace(go.Scatter(x=d_tr.BMI, y=obs, mode="markers", name="past exams",
                                         marker=dict(size=8, color=INK2, opacity=0.55,
                                                     line=dict(color="white", width=1))))
                fig.add_trace(go.Scatter(x=[BMI], y=[rp["mAs"] if t == "mAs" else rp["scan_length_cm"]],
                                         mode="markers", name="this patient",
                                         marker=dict(size=13, color=ORANGE, symbol="star",
                                                     line=dict(color="white", width=1.5))))
                style(fig, 340, title=f"{nice} vs BMI", xaxis_title="BMI", yaxis_title=unit,
                      showlegend=(t == "mAs"))
                col.plotly_chart(fig, width="stretch")
            if expert:
                for t in ("mAs", "scan_length_cm"):
                    st.markdown(f"**{t}** — selected model: {pred.prescan.selected[t]}")
                    st.dataframe(pred.prescan.summary_table(t).style.format(
                        {"MAPE_%": "{:.2f}", "MAPE_%_sd": "{:.2f}", "MAE": "{:.2f}", "RMSE": "{:.2f}",
                         "R2": "{:.3f}"}), width="stretch", hide_index=True)

# =================================================================== HELP
with tab_help:
    st.markdown("#### Help")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
##### How to use it

1. Fill the box at the top: BMI, kV, mAs, scan length.
2. **Tab 1** shows the predicted dose and compares it with your reference level.
3. **Tab 2** gives one recommended change, with the saving it produces. Open *Scanner limits* first so the
   proposals match your machine.
4. **Tab 3** does the same before the scan, from the BMI alone.

Nothing is saved or uploaded. Close the browser tab and nothing is kept, except what you enter in the
learning loop in Expert mode.

##### Frequently asked

**Can I trust a single number?** No — use the range. It is built so the true value falls inside it 9 times
out of 10.

**It says "extrapolation". What now?** The setting is outside anything in your data (for example 100 kV when
all past exams used 120). The physics still applies, but confirm it on a phantom before clinical use.

**Can I add my own examination?** Yes — Expert mode, *Add data*. You need at least 15 exams, ideally 50+.

**Does it replace the physicist?** No. It is a decision-support tool for a research setting. Any protocol
change needs the medical physicist and the responsible radiologist.
        """)
    with c2:
        st.markdown("""
##### The words on screen

| Term | What it means |
|---|---|
| **BMI** | Weight ÷ height². A rough measure of how much of the beam the patient absorbs. |
| **kV** | How penetrating the X-ray beam is. Lower kV = less dose but more noise; iodine also shows up better. |
| **mAs** | How many X-rays are used. Dose is exactly proportional to it. |
| **Scan length** | How much of the body is irradiated. Total dose is proportional to it. |
| **CTDIvol** (dose per slice) | How strong the radiation is at each slice, in mGy. Does not depend on scan length. |
| **DLP** (total dose) | CTDIvol × scan length, in mGy·cm. What reference levels are written in. |
| **mSv** | Effective dose — a risk-weighted number. For cardiac CT, roughly DLP × 0.026. |
| **Reference level** | A benchmark, not a limit. If your median is above it, the protocol should be reviewed. |
| **Image quality (CNR)** | How clearly the contrast-filled vessel stands out from the noise. The thing that must not get worse. |
| **Phantom** | A plastic test object scanned instead of a patient, used to measure image quality. |
        """)
    st.caption("A fuller glossary is in docs/CT_Dose_AI_Cheatsheet.pdf, and the method is in "
               "docs/CT_Dose_AI_Report.pdf.")

# =================================================================== EXPERT TABS
if expert:
    tab_bo, tab_clin, tab_batch, tab_model, tab_train = tabs[4:]

    with tab_bo:
        render_phantom(pred, exam)

    with tab_clin:
        render_clinical(pred, exam, BMI, L)

    with tab_batch:
        st.subheader("Predict many scans at once")
        st.caption("Upload a CSV, Excel or PDF file with the columns BMI, kV, mAs and scan length. If it also "
                   "contains the measured CTDIvol and DLP, the app reports its own error against them.")
        up = st.file_uploader("File", type=["csv", "xlsx", "xls", "pdf"], key="batch")
        if up:
            with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as f_:
                f_.write(up.getvalue())
            df = load_table(f_.name, default_exam=exam).dropna(subset=FEATURES)
            out = pred.predict(df)
            for t in TARGETS:
                if df[t].notna().any():
                    out[t + "_actual"] = df[t].to_numpy()
                    out[t + "_err_%"] = (out[t] / out[t + "_actual"] - 1) * 100
            st.dataframe(out.round(2), width="stretch")
            for t in TARGETS:
                if t + "_err_%" in out:
                    st.write(f"**{t}** average error: {out[t + '_err_%'].abs().mean():.2f}%")
            st.download_button("Download the predictions (CSV)", out.to_csv(index=False).encode("utf-8-sig"),
                               "predictions.csv", "text/csv")

    with tab_model:
        st.subheader(f"Model details — {exam}")
        st.write(f"Trained {pred.trained_at} on **{pred.n_train}** examinations "
                 f"({pred.audit['n_unique']} unique). Selected: "
                 + ", ".join(f"**{t}** → {n}" for t, n in pred.selected.items()))
        st.markdown("**The fitted dose law** — CTDIvol = k · mAs · (kV/120)ⁿ")
        st.json({k: round(v, 5) for k, v in pred.physics_ctdi.params().items()})
        for t in TARGETS:
            s = pred.cv_summary[t]
            st.markdown(f"#### {t}: every model, cross-validated")
            st.dataframe(s[["model", "MAPE_%", "MAPE_%_sd", "MAE", "RMSE", "R2", "MaxAPE_%", "selected"]]
                         .style.format({"MAPE_%": "{:.2f}", "MAPE_%_sd": "{:.2f}", "MAE": "{:.2f}",
                                        "RMSE": "{:.2f}", "R2": "{:.4f}", "MaxAPE_%": "{:.1f}"}),
                         width="stretch", hide_index=True)
            y = pred.train_df[t].to_numpy()
            P = pred.oof[t][pred.selected[t]].mean(axis=0)
            fig = go.Figure()
            lo, hi = float(min(y.min(), P.min())) * 0.9, float(max(y.max(), P.max())) * 1.05
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                     line=dict(color=INK2, dash="dot", width=1), hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=y, y=P, mode="markers",
                                     marker=dict(size=9, color=BLUE, line=dict(color="white", width=1.5)),
                                     customdata=pred.train_df[FEATURES].to_numpy(),
                                     hovertemplate="measured %{x:.1f} · predicted %{y:.1f}<br>BMI %{customdata[0]}, "
                                                   "%{customdata[1]} kV, %{customdata[2]} mAs, "
                                                   "%{customdata[3]} cm<extra></extra>", showlegend=False))
            style(fig, h=360, title=f"{t}: predicted vs measured (each point unseen by the model)",
                  xaxis_title="measured", yaxis_title="predicted")
            st.plotly_chart(fig, width="stretch")
        with st.expander("Data audit"):
            st.json(pred.audit)
        rep = ROOT / "reports" / exam / "report.md"
        if rep.exists():
            with st.expander("Full training report"):
                st.markdown(rep.read_text(encoding="utf-8"))

    with tab_train:
        st.subheader("Add another examination type")
        st.caption("Upload data for a different examination (CSV, Excel or PDF). Columns: exam (optional), BMI, "
                   "kV, mAs, scan length, CTDIvol, DLP. Each examination gets its own model. At least 15 rows "
                   "are needed; 50 or more is better.")
        up2 = st.file_uploader("Training data", type=["csv", "xlsx", "xls", "pdf"], key="train")
        name = st.text_input("Name of the examination (used if the file has no 'exam' column)", "NEW_EXAM")
        reps = st.slider("How thorough the validation is (more = slower)", 2, 20, 5)
        if up2 and st.button("Train the model", type="primary"):
            from train import train_exam  # noqa: E402
            with tempfile.NamedTemporaryFile(suffix=Path(up2.name).suffix, delete=False) as f_:
                f_.write(up2.getvalue())
            df = load_table(f_.name, default_exam=name).dropna(subset=FEATURES + TARGETS)
            save_to = ROOT / "data" / f"{Path(up2.name).stem}.csv"
            df.to_csv(save_to, index=False)
            buf = io.StringIO()
            with st.spinner("Training and cross-validating…"), contextlib.redirect_stdout(buf):
                for ex, g in df.groupby("exam"):
                    if len(g) >= 15:
                        train_exam(g.reset_index(drop=True), ex, reps, 5, 0.10, ROOT / "models", ROOT / "reports")
                    else:
                        print(f"{ex}: only {len(g)} rows — skipped (15 needed)")
            st.code(buf.getvalue())
            st.success(f"Done. The data was saved as data/{save_to.name}. Reload the page to select the new "
                       "examination in the sidebar.")
