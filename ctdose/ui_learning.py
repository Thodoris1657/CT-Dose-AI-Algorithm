"""Streamlit UI for the two learning loops: phantom Bayesian optimisation and the clinical bandit."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .bandit import BanditConfig, DoseBandit, run_bandit_simulation
from .phantom_bo import Campaign, dose_fn_from_predictor, run_simulation, suggest

BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
KV_RAMP = {70: "#c6dcf7", 80: "#9ec5f4", 90: "#7bb0ee", 100: "#5a9be6", 110: "#4189de", 120: "#2a78d6",
           130: "#22649f", 140: "#1a4f94"}
INK2, GRID = "#52514e", "rgba(0,0,0,0.08)"
KV_ALL = [70, 80, 90, 100, 110, 120, 130, 140]


def _style(fig, h=380, **kw):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(color=INK2, size=13),
                      legend=dict(orientation="h", yanchor="top", y=-0.18, x=0), **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# ============================================================================ shared hooks
def learned_gamma_options(pred) -> dict[str, float]:
    """γ sources for the Reduce-dose optimiser: literature + every phantom campaign with enough scans."""
    opts = {"Literature (γ = 1.35)": 1.35}
    f = dose_fn_from_predictor(pred)
    for name, p in Campaign.list().items():
        try:
            c = Campaign.load(p)
            if len(c.scans) >= 4 and c.target():
                r = suggest(c, f)
                if r.surrogate is not None:
                    opts[f"Phantom '{c.name}' (γ = {r.surrogate.gamma:.2f} ± {r.surrogate.gamma_sd:.2f})"] = r.surrogate.gamma
        except Exception:  # noqa: BLE001
            continue
    return opts


def _bandit(pred, exam, cfg: BanditConfig | None = None) -> DoseBandit:
    return DoseBandit(pred, cfg or st.session_state.get(f"bandit_cfg_{exam}", BanditConfig()),
                      log=DoseBandit.load_log(exam), seed=np.random.randint(1_000_000))


def clinical_hint(pred, exam, bmi, L, min_scored: int = 20) -> str | None:
    log = DoseBandit.load_log(exam)
    n = int(pd.to_numeric(log.score, errors="coerce").notna().sum())
    if n < min_scored:
        return None
    b = _bandit(pred, exam)
    p = b.propose(bmi, explore=False)
    d = pred.predict_one(bmi, p["kVp"], p["mAs"], L)
    return (f"**Learned from {n} radiologist-scored exams:** for BMI {bmi:.1f} the lowest-dose setting with "
            f"P(acceptable) ≥ {b.cfg.target_p:.2f} is **{p['kVp']:g} kV / {p['mAs']:.0f} mAs**: CTDIvol "
            f"{d['CTDIvol']:.1f} mGy, DLP {d['DLP']:.0f} mGy·cm (P = {p['p_mean']:.2f}, 80% CI "
            f"{p['p_lo']:.2f}–{p['p_hi']:.2f}).")


# ============================================================================ PHANTOM BO
def render_phantom(pred, exam):
    st.subheader("Phantom optimisation: Bayesian optimisation of kV / mAs")
    st.caption("Goal: the **lowest CTDIvol** whose measured **CNR** still meets your target. The app proposes the "
               "next phantom scan; you scan, measure CNR (ROI in the iodine insert vs background) and enter it. "
               "Typically converges in 8–20 scans. The calibrated γ then feeds the *Reduce dose* tab.")
    f = dose_fn_from_predictor(pred)
    camps = Campaign.list()
    NEW = "➕ New campaign"
    c1, c2 = st.columns([3, 1])
    choice = c1.selectbox("Campaign", [NEW, *camps.keys()], index=1 if camps else 0)
    if c2.button("Run simulated demo", help="Runs a full campaign on a synthetic phantom and saves it as DEMO_simulated"):
        sim = run_simulation(f, seed=np.random.randint(1000), ref=(120, float(round(pred.train_df.mAs.median() / 5) * 5)))
        c = sim["campaign"]
        c.name, c.phantom = "DEMO_simulated", "synthetic phantom (simulator)"
        c.save()
        st.session_state["bo_flash"] = (f"Demo finished in {sim['n_scans']} scans. Recommended "
                                        f"{sim['recommended']['kVp']:g} kV / {sim['recommended']['mAs']:.0f} mAs "
                                        f"({sim['recommended']['CTDIvol']:.1f} mGy) vs true optimum "
                                        f"{sim['true_optimum'][2]:.1f} mGy; reference {sim['ref_dose']:.1f} mGy.")
        st.rerun()
    if msg := st.session_state.pop("bo_flash", None):
        st.success(msg)

    if choice == NEW:
        with st.form("new_campaign"):
            a, b, cc = st.columns(3)
            name = a.text_input("Campaign name", f"{exam}_phantom_1")
            phantom = b.text_input("Phantom", "cardiac phantom, medium size (≈ BMI 25–30)")
            kvs = cc.multiselect("kV values allowed", KV_ALL, default=[80, 100, 120])
            a, b, cc, d = st.columns(4)
            ref_kv = a.selectbox("Reference kV (current protocol)", KV_ALL, index=KV_ALL.index(120))
            ref_mas = b.number_input("Reference mAs", 10.0, 5000.0, float(round(pred.train_df.mAs.median() / 5) * 5), 5.0)
            mas_min = cc.number_input("Min mAs", 5.0, 2000.0, 50.0, 5.0)
            mas_max = d.number_input("Max mAs (tube limit)", 50.0, 6000.0, 1500.0, 50.0)
            a, b = st.columns(2)
            mode = a.radio("CNR target", ["% of reference scan CNR", "absolute CNR value"], horizontal=True)
            tau = b.number_input("Target (% of reference, or absolute CNR)", 0.1, 500.0, 100.0, 1.0)
            if st.form_submit_button("Create campaign", type="primary"):
                c = Campaign(name=name, exam=exam, phantom=phantom, kv_options=sorted(kvs), mas_min=mas_min,
                             mas_max=mas_max, ref_kVp=float(ref_kv), ref_mAs=float(ref_mas),
                             cnr_target=None if mode.startswith("%") else float(tau),
                             tau=tau / 100 if mode.startswith("%") else 1.0)
                c.save()
                st.rerun()
        return

    c = Campaign.load(camps[choice])
    res = suggest(c, f)
    st.markdown(f"**{c.phantom}** · kV options {c.kv_options} · mAs {c.mas_min:g}–{c.mas_max:g} · reference "
                f"{c.ref_kVp:g} kV / {c.ref_mAs:g} mAs · scans so far: **{len(c.scans)}** · CNR target: "
                f"**{res.target:.2f}**" if res.target else
                f"**{c.phantom}** · reference {c.ref_kVp:g} kV / {c.ref_mAs:g} mAs · scans so far: **0**")

    k1, k2, k3 = st.columns(3)
    if res.next_scan:
        ns = res.next_scan
        k1.metric("Next scan to perform", f"{ns['kVp']:g} kV · {ns['mAs']:.0f} mAs", f"CTDIvol ≈ {ns['CTDIvol']:.1f} mGy",
                  delta_color="off")
        k1.caption(ns["why"] + (f" Predicted CNR {ns['CNR_pred']:.2f}, P(meets target) {ns['PoF']:.2f}." if "CNR_pred" in ns else ""))
    else:
        k1.metric("Campaign status", "Converged ✓")
        k1.caption(res.reason)
    if res.recommended:
        rc = res.recommended
        ref_dose = float(f(c.ref_kVp, c.ref_mAs))
        k2.metric("Recommended protocol", f"{rc['kVp']:g} kV · {rc['mAs']:.0f} mAs",
                  f"{(rc['CTDIvol']/ref_dose-1)*100:+.0f}% dose vs reference", delta_color="inverse")
        k2.caption(f"CTDIvol {rc['CTDIvol']:.1f} mGy, predicted CNR {rc['CNR_mean']:.2f} "
                   f"(90% {rc['CNR_lo']:.2f}–{rc['CNR_hi']:.2f}), P(meets target) {rc['PoF']:.2f}")
    if res.surrogate is not None:
        k3.metric("Calibrated iodine exponent γ", f"{res.surrogate.gamma:.2f} ± {res.surrogate.gamma_sd:.2f}")
        k3.caption(f"noise–dose exponent δ = {res.surrogate.delta:.2f} (theory 0.5). Used by *Reduce dose* when selected.")

    with st.form("add_scan", clear_on_submit=True):
        st.markdown("**Enter a phantom measurement**")
        ns = res.next_scan or {}
        a, b, cc, d, e = st.columns(5)
        kv = a.selectbox("kV", KV_ALL, index=KV_ALL.index(int(ns.get("kVp", c.ref_kVp))) if int(ns.get("kVp", c.ref_kVp)) in KV_ALL else 0)
        mas = b.number_input("mAs", 1.0, 6000.0, float(ns.get("mAs", c.ref_mAs)), 5.0)
        cnr = cc.number_input("Measured CNR", 0.0, 1000.0, 0.0, 0.1)
        ctdi = d.number_input("CTDIvol on console (optional)", 0.0, 500.0, 0.0, 0.1)
        note = e.text_input("Note")
        if st.form_submit_button("Add measurement", type="primary") and cnr > 0:
            c.add_scan(kv, mas, cnr, ctdi if ctdi > 0 else None, note)
            c.save()
            st.rerun()

    if len(c.scans):
        with st.expander(f"Scans ({len(c.scans)}) — edit or delete rows, then save"):
            ed = st.data_editor(c.scans_df(), num_rows="dynamic", width="stretch", key=f"ed_{c.name}")
            if st.button("Save table"):
                c.scans = [dict(kVp=float(r.kVp), mAs=float(r.mAs), CNR=float(r.CNR),
                                CTDIvol=None if pd.isna(r.CTDIvol) else float(r.CTDIvol), note=str(r.note or ""))
                           for r in ed.dropna(subset=["kVp", "mAs", "CNR"]).itertuples()]
                c.save()
                st.rerun()

    if res.surrogate is not None and len(res.grid):
        G, obs = res.grid, c.scans_df()
        obs["CTDI"] = [r.CTDIvol if pd.notna(r.CTDIvol) else float(f(r.kVp, r.mAs)) for r in obs.itertuples()]
        fig = go.Figure()
        for kv in c.kv_options:
            g = G[G.kVp == kv]
            col = KV_RAMP.get(kv, BLUE)
            fig.add_trace(go.Scatter(x=np.r_[g.CTDIvol, g.CTDIvol[::-1]], y=np.r_[g.CNR_hi, g.CNR_lo[::-1]],
                                     fill="toself", fillcolor=col, opacity=0.15, line=dict(width=0),
                                     hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=g.CTDIvol, y=g.CNR_mean, mode="lines", line=dict(color=col, width=2),
                                     name=f"{kv} kV (model)",
                                     hovertemplate=f"{kv} kV · %{{x:.1f}} mGy → CNR %{{y:.2f}}<extra></extra>"))
            o = obs[obs.kVp == kv]
            if len(o):
                fig.add_trace(go.Scatter(x=o.CTDI, y=o.CNR, mode="markers", name=f"{kv} kV (measured)",
                                         marker=dict(size=10, color=col, line=dict(color="white", width=2)),
                                         hovertemplate=f"measured {kv} kV · %{{x:.1f}} mGy · CNR %{{y:.2f}}<extra></extra>"))
        fig.add_hline(y=res.target, line=dict(color=INK2, dash="dot", width=1), annotation_text="CNR target",
                      annotation_position="top left")
        if res.recommended:
            fig.add_trace(go.Scatter(x=[res.recommended["CTDIvol"]], y=[res.recommended["CNR_mean"]], mode="markers",
                                     name="recommended", marker=dict(size=14, color=ORANGE, symbol="star",
                                                                     line=dict(color="white", width=1.5))))
        _style(fig, 480, title="Model of CNR vs dose per kV (band = 90% credible interval)",
               xaxis_title="CTDIvol (mGy)", yaxis_title="CNR")
        fig.update_xaxes(type="log")
        st.plotly_chart(fig, width="stretch")


# ============================================================================ CLINICAL BANDIT
def render_clinical(pred, exam, bmi_default, L_default):
    st.subheader("Clinical learning: safe contextual bandit with radiologist scores")
    st.caption("For each patient the app proposes a protocol. A radiologist **approves or overrides** it. After "
               "the scan the reader scores diagnostic quality **1–4** (≥ 3 = acceptable). The model learns the "
               "lowest dose that stays acceptable for each BMI and lowers the dose step by step, within the safety "
               "limits. Every scored exam also becomes new training data for the dose predictor.")

    with st.expander("Safety & learning settings"):
        a, b, c, d = st.columns(4)
        target_p = a.slider("Target P(acceptable)", 0.80, 0.995, 0.97, 0.005)
        safety_p = b.slider("Safety floor (pessimistic P)", 0.50, 0.99, 0.85, 0.01)
        max_step = c.slider("Max dose step below proven level", 0.05, 0.40, 0.15, 0.05)
        acc = d.selectbox("Acceptable if score ≥", [2, 3, 4], index=1)
        a, b, c = st.columns(3)
        kvs = a.multiselect("kV options", KV_ALL, default=[80, 100, 120], key="clin_kv")
        g_opts = learned_gamma_options(pred)
        gsrc = b.selectbox("γ source", list(g_opts), key="clin_g")
        rule = c.toggle("BMI minimum-kV rule", True, key="clin_rule")
    cfg = BanditConfig(kv_options=tuple(sorted(kvs)) or (120,), gamma=g_opts[gsrc], target_p=target_p,
                       safety_p=safety_p, max_step=max_step, accept_score=acc, apply_bmi_kv_rule=rule)
    st.session_state[f"bandit_cfg_{exam}"] = cfg
    bandit = _bandit(pred, exam, cfg)

    # --------------------------------------------------------------- 1. propose
    st.markdown("#### 1 · New patient → proposal")
    a, b, c, d = st.columns([1, 1, 1, 1])
    pid = a.text_input("Patient ID (pseudonymised)", "")
    bmi = b.number_input("BMI", 12.0, 80.0, float(bmi_default), 0.1, key="clin_bmi")
    L = c.number_input("Scan length (cm)", 2.0, 150.0, float(L_default), 0.1, key="clin_L")
    if d.button("Propose protocol", type="primary"):
        st.session_state["clin_prop"] = dict(bandit.propose(bmi), BMI=bmi, L=L, pid=pid)
    prop = st.session_state.get("clin_prop")
    if prop and abs(prop["BMI"] - bmi) < 1e-9:
        dose = pred.predict_one(bmi, prop["kVp"], prop["mAs"], L)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Proposed", f"{prop['kVp']:g} kV · {prop['mAs']:.0f} mAs")
        m1.caption(f"mode: {prop['mode']}")
        m2.metric("CTDIvol / DLP", f"{dose['CTDIvol']:.1f} mGy · {dose['DLP']:.0f}")
        m3.metric("P(acceptable)", f"{prop['p_mean']:.2f}")
        m3.caption(f"80% credible interval {prop['p_lo']:.2f}–{prop['p_hi']:.2f}")
        m4.metric("CTDIvol vs current practice", f"{-prop['saving_vs_ref_pct']:+.0f}%")
        m4.caption(f"local median for this BMI at 120 kV: {prop['ctdi_ref']:.1f} mGy")
        with st.expander("All candidate protocols and safety checks"):
            A = bandit.arms(bmi)
            st.dataframe(A[["kVp", "level", "mAs", "CTDIvol", "CNR_rel", "p_mean", "p_lo", "p_hi", "allowed", "blocked_by"]]
                         .style.format({"CTDIvol": "{:.1f}", "CNR_rel": "{:.0%}", "p_mean": "{:.3f}", "p_lo": "{:.3f}",
                                        "p_hi": "{:.3f}", "level": "{:.2f}", "mAs": "{:.0f}"}),
                         width="stretch", hide_index=True)
        with st.form("approve"):
            st.markdown("**Radiologist approval**")
            a, b, c = st.columns(3)
            decision = a.radio("Decision", ["Approve as proposed", "Override"], horizontal=False)
            kv_o = b.selectbox("Override kV", KV_ALL, index=KV_ALL.index(int(prop["kVp"])) if int(prop["kVp"]) in KV_ALL else 0)
            mas_o = c.number_input("Override mAs", 1.0, 6000.0, float(prop["mAs"]), 5.0)
            if st.form_submit_button("Register scan (score later)", type="primary"):
                kv_u, mas_u = (prop["kVp"], prop["mAs"]) if decision.startswith("Approve") else (kv_o, mas_o)
                dd = pred.predict_one(bmi, kv_u, mas_u, L)
                bandit.record(exam=exam, patient_id=prop["pid"], BMI=bmi, scan_length_cm=L,
                              proposed_kVp=prop["kVp"], proposed_mAs=prop["mAs"], proposal_mode=prop["mode"],
                              kVp=kv_u, mAs=mas_u, approved_as_proposed=decision.startswith("Approve"),
                              CTDIvol=dd["CTDIvol"], DLP=dd["DLP"], dose_source="predicted")
                bandit.save_log(exam)
                st.session_state.pop("clin_prop", None)
                st.rerun()

    # --------------------------------------------------------------- 2. score
    log = bandit.log
    pending = log[pd.to_numeric(log.score, errors="coerce").isna()]
    st.markdown(f"#### 2 · Score completed scans ({len(pending)} pending)")
    if len(pending):
        labels = {i: f"#{i} · {r.timestamp} · {r.patient_id if isinstance(r.patient_id, str) else ''} · BMI {r.BMI} · "
                     f"{r.kVp:g} kV {r.mAs:.0f} mAs" for i, r in pending.iterrows()}
        with st.form("score"):
            i = st.selectbox("Scan", list(labels), format_func=labels.get)
            a, b, c, d, e = st.columns(5)
            score = a.selectbox("Diagnostic quality", [4, 3, 2, 1],
                                format_func=lambda s: {4: "4 · excellent", 3: "3 · diagnostic", 2: "2 · limited",
                                                       1: "1 · non-diagnostic"}[s])
            ctdi = b.number_input("CTDIvol on console (optional)", 0.0, 500.0, 0.0, 0.1)
            dlp = c.number_input("DLP on console (optional)", 0.0, 20000.0, 0.0, 1.0)
            reader = d.text_input("Reader")
            comment = e.text_input("Comment")
            if st.form_submit_button("Save score", type="primary"):
                log.loc[i, "score"] = score
                log.loc[i, "acceptable"] = score >= cfg.accept_score
                log.loc[i, "reader"], log.loc[i, "comment"] = reader, comment
                if ctdi > 0 and dlp > 0:
                    log.loc[i, "CTDIvol"], log.loc[i, "DLP"], log.loc[i, "dose_source"] = ctdi, dlp, "scanner"
                bandit.log = log
                bandit.save_log(exam)
                st.rerun()

    # --------------------------------------------------------------- 3. dashboard
    st.markdown("#### 3 · What the model has learned")
    s = bandit.summary()
    a, b, c = st.columns(3)
    a.metric("Scored exams", s.get("n_scored", 0))
    if s.get("n_scored"):
        b.metric("Acceptable rate", f"{s['acceptable_rate']*100:.1f}%")
        c.metric("Mean dose vs local median", f"{-s['mean_saving_vs_local_median_pct']:+.0f}%")
    else:
        b.caption("No scored exams yet. The curves below show the physics-informed prior.")

    bmi_show = st.slider("Show learned curves for BMI", 16.0, 50.0, float(bmi), 0.5)
    cr = bandit.ctdi_ref(bmi_show)
    fig = go.Figure()
    from .bandit import features
    W = bandit.model.sample(np.random.RandomState(0), 300)
    for kv in cfg.kv_options:
        ct = np.geomspace(cr * 0.15, cr * 1.5, 80)
        F = features(np.full_like(ct, bmi_show), ct, np.full_like(ct, kv), np.full_like(ct, cr))
        draws = bandit.model.prob(F, W)
        lo, hi, mu = np.quantile(draws, 0.1, axis=1), np.quantile(draws, 0.9, axis=1), bandit.model.prob(F)[:, 0]
        col = KV_RAMP.get(kv, BLUE)
        fig.add_trace(go.Scatter(x=np.r_[ct, ct[::-1]], y=np.r_[hi, lo[::-1]], fill="toself", fillcolor=col,
                                 opacity=0.15, line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=ct, y=mu, mode="lines", line=dict(color=col, width=2), name=f"{kv} kV",
                                 hovertemplate=f"{kv} kV · %{{x:.1f}} mGy → P %{{y:.3f}}<extra></extra>"))
    fig.add_hline(y=cfg.target_p, line=dict(color=INK2, dash="dot", width=1), annotation_text="target",
                  annotation_position="bottom right")
    _style(fig, 380, title=f"P(acceptable) vs CTDIvol at BMI {bmi_show:.1f} (band = 80% credible interval)",
           xaxis_title="CTDIvol (mGy)", yaxis_title="P(acceptable)")
    fig.update_xaxes(type="log")
    st.plotly_chart(fig, width="stretch")

    sc = bandit.scored()
    if len(sc):
        sc = sc.reset_index(drop=True)
        ok = sc.score.astype(float) >= cfg.accept_score
        ref = np.array([bandit.ctdi_ref(x) for x in sc.BMI.astype(float)])
        rel = sc.CTDIvol.astype(float) / ref * 100
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=sc.index[ok] + 1, y=rel[ok], mode="markers", name="acceptable",
                                  marker=dict(size=8, color=BLUE)))
        fig2.add_trace(go.Scatter(x=sc.index[~ok] + 1, y=rel[~ok], mode="markers", name="not acceptable",
                                  marker=dict(size=10, color=RED, symbol="x")))
        fig2.add_hline(y=100, line=dict(color=INK2, dash="dot", width=1), annotation_text="local median dose")
        _style(fig2, 320, title="Dose of each scored exam relative to the local median for its BMI",
               xaxis_title="exam #", yaxis_title="% of local median CTDIvol")
        st.plotly_chart(fig2, width="stretch")
        st.download_button("Download log (CSV)", bandit.log.to_csv(index=False).encode("utf-8-sig"),
                           f"{exam}_bandit_log.csv", "text/csv")

    with st.expander("Simulation: what to expect (synthetic radiologist, 300 patients)"):
        st.caption("Runs the full loop against a simulated reader whose acceptance depends on the true iodine CNR. "
                   "Nothing is written to your log.")
        if st.button("Run simulation"):
            with st.spinner("Simulating 300 patients…"):
                r = run_bandit_simulation(pred, 300, seed=np.random.randint(1000), cfg=cfg)
            r["saving"] = (1 - r.CTDIvol / r.ctdi_ref) * 100
            r["bad"] = (r.score < cfg.accept_score) * 100
            roll = r[["saving", "bad"]].rolling(30, min_periods=10).mean()
            a, b = st.columns(2)
            f1 = _style(go.Figure(go.Scatter(x=r.i + 1, y=roll.saving, line=dict(color=BLUE, width=2))), 300,
                        title="Dose saving vs local median (%, rolling 30)", xaxis_title="patient #")
            f2 = _style(go.Figure(go.Scatter(x=r.i + 1, y=roll.bad, line=dict(color=RED, width=2))), 300,
                        title="Non-acceptable scans (%, rolling 30)", xaxis_title="patient #")
            a.plotly_chart(f1, width="stretch")
            b.plotly_chart(f2, width="stretch")
            st.write(f"Last 150 patients: mean saving **{r.saving.iloc[150:].mean():.0f}%**, non-acceptable "
                     f"**{r.bad.iloc[150:].mean():.1f}%** (target ≤ {(1-cfg.target_p)*100:.1f}% + safety margin).")
