"""Plotly figures shared by the user interface.

Pure functions: they take numbers, they return a figure. No Shiny, no Streamlit, no globals —
so they can be unit-tested, reused in a report, or swapped for a different front end.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

# one palette for the whole application
BLUE, DARK, AQUA, ORANGE, RED = "#2a78d6", "#1a4f94", "#1baf7a", "#eb6834", "#e34948"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8985"
GRID = "rgba(0,0,0,0.07)"
KV_COLOR = {70: "#cfe0f7", 80: "#9ec5f4", 90: "#7bb0ee", 100: "#5a9be6",
            110: "#4189de", 120: "#2a78d6", 130: "#225f9f", 140: "#1a4f94", 150: "#143c71"}


def _base(fig: go.Figure, h: int = 320, title: str = "", **kw) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=h, margin=dict(l=8, r=8, t=38 if title else 8, b=8),
        font=dict(color=INK2, size=13, family="system-ui, -apple-system, Segoe UI, sans-serif"),
        title=dict(text=title, font=dict(size=14, color=INK), x=0, xanchor="left") if title else None,
        hoverlabel=dict(font_size=13), showlegend=False, **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0.2)")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0.2)")
    return fig


# --------------------------------------------------------------------------- dose thermometer
def dose_bar(dlp: float, lo: float, hi: float, reference: float) -> go.Figure:
    """One horizontal bar: this scan against the reference level. The first thing a clinician reads."""
    colour = AQUA if dlp <= reference else (ORANGE if dlp <= 1.5 * reference else RED)
    top = max(dlp * 1.25, reference * 1.6, hi * 1.1)
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, x1=reference, y0=-0.42, y1=0.42,
                  fillcolor="rgba(27,175,122,0.10)", line_width=0, layer="below")
    fig.add_trace(go.Bar(x=[dlp], y=[""], orientation="h", width=0.55,
                         marker=dict(color=colour, cornerradius=6),
                         error_x=dict(type="data", symmetric=False, array=[hi - dlp], arrayminus=[dlp - lo],
                                      color="rgba(0,0,0,0.35)", thickness=1.5, width=6),
                         hovertemplate="this scan: %{x:.0f} mGy·cm<extra></extra>"))
    fig.add_vline(x=reference, line=dict(color=INK2, width=2, dash="dash"))
    fig.add_annotation(x=reference, y=0.55, text=f"reference {reference:g}", showarrow=False,
                       font=dict(size=12, color=INK2), yanchor="bottom")
    fig.add_annotation(x=dlp, y=-0.6, text=f"<b>{dlp:.0f}</b> mGy·cm", showarrow=False,
                       font=dict(size=13, color=colour), yanchor="top")
    _base(fig, 130)
    fig.update_xaxes(range=[0, top], title=None)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=8, r=8, t=26, b=30), bargap=0.3)
    return fig


# --------------------------------------------------------------------------- dose vs settings
def dose_vs_mas(curves, kvs, mas_now: float, value_now: float, lo: float, hi: float,
                target: str = "DLP", reference: float | None = None) -> go.Figure:
    unit = "mGy·cm" if target == "DLP" else "mGy"
    fig = go.Figure()
    for kv in kvs:
        c = curves[curves.kVp == kv]
        fig.add_trace(go.Scatter(x=c.mAs, y=c[target], mode="lines", name=f"{kv:g} kV",
                                 line=dict(width=2.2, color=KV_COLOR.get(int(kv), BLUE)),
                                 hovertemplate=f"{kv:g} kV · %{{x:.0f}} mAs → %{{y:.0f}} {unit}<extra></extra>"))
        fig.add_annotation(x=c.mAs.iloc[-1], y=c[target].iloc[-1], text=f"{kv:g} kV", showarrow=False,
                           xanchor="left", xshift=4, font=dict(size=11, color=INK2))
    fig.add_trace(go.Scatter(x=[mas_now], y=[value_now], mode="markers", name="this scan",
                             marker=dict(size=13, color=ORANGE, line=dict(color="white", width=2)),
                             error_y=dict(type="data", symmetric=False, array=[hi - value_now],
                                          arrayminus=[value_now - lo], color=ORANGE, thickness=1.5),
                             hovertemplate=f"this scan: %{{y:.0f}} {unit}<extra></extra>"))
    if reference and target == "DLP":
        fig.add_hline(y=reference, line=dict(color=INK2, dash="dot", width=1.2),
                      annotation_text="reference", annotation_position="top left",
                      annotation_font=dict(size=11, color=INK2))
    _base(fig, 330, f"{'Total dose' if target == 'DLP' else 'Dose per slice'} ({unit}) as mAs changes")
    fig.update_xaxes(title="mAs")
    return fig


# --------------------------------------------------------------------------- the trade-off
def quality_vs_dose(kv_now: float, ctdi_now: float, dlp_per_ctdi: float, kvs, gamma: float,
                    target_pct: float, best: dict | None) -> go.Figure:
    """Image quality kept (% of the current plan) against total dose. The core explanatory picture."""
    fig = go.Figure()
    grid = np.geomspace(ctdi_now * 0.12, ctdi_now * 1.7, 80)
    for kv in kvs:
        cnr = (kv_now / kv) ** gamma * np.sqrt(grid / ctdi_now) * 100
        fig.add_trace(go.Scatter(x=grid * dlp_per_ctdi, y=cnr, mode="lines", name=f"{kv:g} kV",
                                 line=dict(width=2.2, color=KV_COLOR.get(int(kv), BLUE)),
                                 hovertemplate=f"{kv:g} kV · %{{x:.0f}} mGy·cm → %{{y:.0f}}% quality<extra></extra>"))
        j = int(len(grid) * 0.97)
        fig.add_annotation(x=grid[j] * dlp_per_ctdi, y=cnr[j], text=f"{kv:g} kV", showarrow=False,
                           xanchor="left", xshift=3, font=dict(size=11, color=INK2))
    fig.add_hline(y=target_pct, line=dict(color=AQUA, dash="dash", width=1.6),
                  annotation_text=f"keep at least {target_pct:.0f}%", annotation_position="top left",
                  annotation_font=dict(size=11, color=AQUA))
    fig.add_trace(go.Scatter(x=[ctdi_now * dlp_per_ctdi], y=[100], mode="markers", name="this scan",
                             marker=dict(size=14, color=ORANGE, line=dict(color="white", width=2)),
                             hovertemplate="this scan<extra></extra>"))
    if best:
        fig.add_trace(go.Scatter(x=[best["DLP"]], y=[best["CNR_rel"] * 100], mode="markers",
                                 marker=dict(size=19, color=AQUA, symbol="star", line=dict(color="white", width=1.5)),
                                 hovertemplate="suggested setting<extra></extra>"))
        fig.add_annotation(x=best["DLP"], y=best["CNR_rel"] * 100 + 9, text="suggested", showarrow=False,
                           font=dict(size=12, color=AQUA))
    _base(fig, 340, "Image quality you keep, against the dose it costs")
    fig.update_xaxes(title="total dose (mGy·cm)")
    fig.update_yaxes(title="% of the quality you have now", range=[45, 135])
    return fig


# --------------------------------------------------------------------------- acceptability
def acceptability(model, feats, bmi: float, ctdi_ref: float, ctdi_now: float, dlp_per_ctdi: float,
                  kvs, target_p: float, best: dict | None, n_scored: int) -> go.Figure:
    """P(image is diagnostic) against dose — the prior at first, your readers' data later."""
    rng = np.random.RandomState(0)
    W = model.sample(rng, 300)
    fig = go.Figure()
    grid = np.geomspace(ctdi_now * 0.12, ctdi_now * 1.7, 80)
    for kv in kvs:
        F = feats(np.full_like(grid, bmi), grid, np.full_like(grid, kv), np.full_like(grid, ctdi_ref))
        draws = model.prob(F, W)
        mu = model.prob(F)[:, 0] * 100
        lo, hi = np.quantile(draws, 0.1, axis=1) * 100, np.quantile(draws, 0.9, axis=1) * 100
        col = KV_COLOR.get(int(kv), BLUE)
        fig.add_trace(go.Scatter(x=np.r_[grid, grid[::-1]] * dlp_per_ctdi, y=np.r_[hi, lo[::-1]],
                                 fill="toself", fillcolor=col, opacity=0.13, line=dict(width=0),
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=grid * dlp_per_ctdi, y=mu, mode="lines", name=f"{kv:g} kV",
                                 line=dict(width=2.2, color=col),
                                 hovertemplate=f"{kv:g} kV · %{{x:.0f}} mGy·cm → %{{y:.0f}}%<extra></extra>"))
    fig.add_hline(y=target_p * 100, line=dict(color=AQUA, dash="dash", width=1.6),
                  annotation_text=f"target {target_p*100:.0f}%", annotation_position="bottom left",
                  annotation_font=dict(size=11, color=AQUA))
    p_now = float(model.prob(feats([bmi], [ctdi_now], [kvs[-1] if kvs else 120], [ctdi_ref]))[0, 0]) * 100
    fig.add_trace(go.Scatter(x=[ctdi_now * dlp_per_ctdi], y=[p_now], mode="markers",
                             marker=dict(size=14, color=ORANGE, line=dict(color="white", width=2)),
                             hovertemplate="this scan<extra></extra>"))
    if best:
        p_b = float(model.prob(feats([bmi], [best["CTDIvol"]], [best["kVp"]], [ctdi_ref]))[0, 0]) * 100
        fig.add_trace(go.Scatter(x=[best["DLP"]], y=[p_b], mode="markers",
                                 marker=dict(size=19, color=AQUA, symbol="star",
                                             line=dict(color="white", width=1.5)),
                                 hovertemplate="suggested setting<extra></extra>"))
    title = ("Chance the image is diagnostic — learned from %d scored exams" % n_scored) if n_scored \
        else "Chance the image is diagnostic — physics prior, no scored exams yet"
    _base(fig, 340, title)
    fig.update_xaxes(title="total dose (mGy·cm)")
    fig.update_yaxes(title="% chance", range=[0, 104])
    return fig


# --------------------------------------------------------------------------- options
def options_bar(cand, dlp_now: float, reference: float) -> go.Figure:
    colours = [AQUA if f else "rgba(82,81,78,0.30)" for f in cand.feasible]
    fig = go.Figure(go.Bar(
        y=cand.option, x=cand.DLP, orientation="h", marker=dict(color=colours, cornerradius=5),
        customdata=cand.assign(q=cand.CNR_rel * 100)[["CTDIvol", "q", "mAs", "flags"]].to_numpy(object),
        hovertemplate="%{y}<br>total %{x:.0f} mGy·cm · per slice %{customdata[0]:.1f} mGy"
                      "<br>%{customdata[2]:.0f} mAs · quality %{customdata[1]:.0f}%"
                      "<br>%{customdata[3]}<extra></extra>",
        text=[f"{v:.0f}" for v in cand.DLP], textposition="outside", cliponaxis=False))
    fig.add_vline(x=dlp_now, line=dict(color=ORANGE, width=2),
                  annotation_text="this scan", annotation_position="top",
                  annotation_font=dict(size=11, color=ORANGE))
    fig.add_vline(x=reference, line=dict(color=INK2, dash="dot", width=1.2),
                  annotation_text="reference", annotation_position="bottom",
                  annotation_font=dict(size=11, color=INK2))
    _base(fig, max(260, 30 * len(cand) + 70), "Every option the engine considered")
    fig.update_xaxes(title="total dose (mGy·cm)")
    fig.update_yaxes(autorange="reversed")
    return fig


# --------------------------------------------------------------------------- pre-scan
def prescan_curve(curve, train_df, bmi: float, value: float, column: str, title: str,
                  ylab: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[curve.BMI, curve.BMI[::-1]],
                             y=np.r_[curve[column + "_hi"], curve[column + "_lo"][::-1]],
                             fill="toself", fillcolor=BLUE, opacity=0.12, line=dict(width=0), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=curve.BMI, y=curve[column], mode="lines", line=dict(width=2.4, color=BLUE),
                             hovertemplate="BMI %{x:.1f} → %{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=train_df.BMI, y=train_df[column], mode="markers",
                             marker=dict(size=7, color=INK2, opacity=0.5, line=dict(color="white", width=1)),
                             hovertemplate="past exam: BMI %{x:.1f} → %{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[bmi], y=[value], mode="markers",
                             marker=dict(size=17, color=ORANGE, symbol="star", line=dict(color="white", width=1.5)),
                             hovertemplate="this patient<extra></extra>"))
    _base(fig, 300, title)
    fig.update_xaxes(title="BMI")
    fig.update_yaxes(title=ylab)
    return fig


# --------------------------------------------------------------------------- model card
def predicted_vs_measured(y, p, label: str) -> go.Figure:
    lo, hi = float(min(y.min(), p.min())) * 0.9, float(max(y.max(), p.max())) * 1.06
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(color=INK3, dash="dot", width=1), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=y, y=p, mode="markers",
                             marker=dict(size=8, color=BLUE, line=dict(color="white", width=1.2)),
                             hovertemplate="measured %{x:.1f} · predicted %{y:.1f}<extra></extra>"))
    _base(fig, 310, f"{label}: predicted against measured (every point unseen by the model)")
    fig.update_xaxes(title="measured")
    fig.update_yaxes(title="predicted")
    return fig


def phantom_curve(grid, obs, kvs, target: float, recommended: dict | None) -> go.Figure:
    fig = go.Figure()
    for kv in kvs:
        g = grid[grid.kVp == kv]
        col = KV_COLOR.get(int(kv), BLUE)
        fig.add_trace(go.Scatter(x=np.r_[g.CTDIvol, g.CTDIvol[::-1]], y=np.r_[g.CNR_hi, g.CNR_lo[::-1]],
                                 fill="toself", fillcolor=col, opacity=0.14, line=dict(width=0), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=g.CTDIvol, y=g.CNR_mean, mode="lines", line=dict(width=2.2, color=col),
                                 hovertemplate=f"{kv:g} kV · %{{x:.1f}} mGy → CNR %{{y:.2f}}<extra></extra>"))
        o = obs[obs.kVp == kv]
        if len(o):
            fig.add_trace(go.Scatter(x=o.CTDI, y=o.CNR, mode="markers",
                                     marker=dict(size=11, color=col, line=dict(color="white", width=2)),
                                     hovertemplate=f"measured {kv:g} kV · %{{x:.1f}} mGy · CNR %{{y:.2f}}<extra></extra>"))
    fig.add_hline(y=target, line=dict(color=INK2, dash="dot", width=1.3), annotation_text="CNR target",
                  annotation_position="top left", annotation_font=dict(size=11, color=INK2))
    if recommended:
        fig.add_trace(go.Scatter(x=[recommended["CTDIvol"]], y=[recommended["CNR_mean"]], mode="markers",
                                 marker=dict(size=19, color=ORANGE, symbol="star",
                                             line=dict(color="white", width=1.5)),
                                 hovertemplate="recommended<extra></extra>"))
    _base(fig, 340, "Measured image quality against dose")
    fig.update_xaxes(title="CTDIvol (mGy)", type="log")
    fig.update_yaxes(title="CNR")
    return fig
