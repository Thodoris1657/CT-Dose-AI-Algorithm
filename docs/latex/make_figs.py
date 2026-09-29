import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/claude/proj")
from ctdose.predictor import DosePredictor
from ctdose.data import load_table
from ctdose.optimizer import OptimizerConfig, recommend
from ctdose.phantom_bo import Campaign, PhantomSimulator, dose_fn_from_predictor, suggest
from ctdose.bandit import run_bandit_simulation

BLUE, ORANGE, AQUA, RED, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#0b0b0b", "#52514e"
KV = {80: "#9ec5f4", 100: "#5a9be6", 120: "#2a78d6", 140: "#1a4f94"}
GRID = "#e6e6e3"
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.edgecolor": "#c9c9c4",
                     "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .8, "axes.axisbelow": True,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
OUT = "/home/claude/docs/figs/"
def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT + name, dpi=200, bbox_inches="tight"); plt.close(fig); print(name)

p = DosePredictor.load("/home/claude/proj/models/CCTA.joblib")
d = p.train_df
f_dose = dose_fn_from_predictor(p)

# F1 CTDI vs mAs
fig, ax = plt.subplots(figsize=(6.2, 3.6))
for kv in (120, 140):
    s = d[d.kVp == kv]
    ax.scatter(s.mAs, s.CTDIvol, s=34, color=KV[kv], edgecolor="white", lw=1.2, zorder=3, label=f"{kv} kV (measured)")
    x = np.linspace(300, 1400, 10)
    ax.plot(x, f_dose(kv, x), color=KV[kv], lw=1.6, ls="--", zorder=2)
ax.set_xlabel("mAs"); ax.set_ylabel("CTDIvol (mGy)")
ax.set_title("CTDIvol is a straight line in mAs — the fitted scanner law (dashed)", loc="left", fontsize=11, color=INK)
ax.legend(frameon=False, loc="upper left")
save(fig, "f1_physics.png")

# F2 model benchmark
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
for ax, t in zip(axes, ["CTDIvol", "DLP"]):
    s = p.cv_summary[t].sort_values("MAPE_%", ascending=False)
    cols = [AQUA if sel else "#b9b9b4" for sel in s.selected]
    ax.barh(s.model, s["MAPE_%"], xerr=s["MAPE_%_sd"], color=cols, height=.65, error_kw=dict(ecolor=INK2, lw=1))
    for y, (v, sd) in enumerate(zip(s["MAPE_%"], s["MAPE_%_sd"])):
        ax.text(v + sd + max(s["MAPE_%"]) * .035, y, f"{v:.1f}%", va="center", fontsize=9, color=INK2)
    ax.set_xlim(0, max(s["MAPE_%"]) * 1.32)
    ax.set_xlabel("cross-validated MAPE (%)")
    ax.set_title(t + "  (green = selected)", loc="left", fontsize=11, color=INK)
save(fig, "f2_benchmark.png")

# F3 OOF pred vs actual
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
for ax, t, u in zip(axes, ["CTDIvol", "DLP"], ["mGy", "mGy·cm"]):
    y = d[t].to_numpy(); P = p.oof[t][p.selected[t]].mean(axis=0)
    lo, hi = min(y.min(), P.min()) * .92, max(y.max(), P.max()) * 1.06
    ax.plot([lo, hi], [lo, hi], color=INK2, ls=":", lw=1)
    ax.scatter(y, P, s=34, color=BLUE, edgecolor="white", lw=1.1, zorder=3)
    ax.set_xlabel(f"measured {t} ({u})"); ax.set_ylabel("predicted (out-of-fold)")
    mape = np.mean(np.abs(P - y) / y) * 100
    ax.set_title(f"{t}: {mape:.1f}% mean error", loc="left", fontsize=11, color=INK)
save(fig, "f3_oof.png")

# F4 DLP vs reference levels
fig, ax = plt.subplots(figsize=(6.6, 3.6))
ax.hist(d.DLP, bins=14, color=BLUE, edgecolor="white")
ax.set_ylim(0, 11)
for x, lab, c, yy in [(400, "reference for CCTA: 400", ORANGE, 10.4),
                      (870, "French DRL, retrospective gating: 870", INK2, 9.4),
                      (float(d.DLP.median()), f"median of these 50 exams: {d.DLP.median():.0f}", AQUA, 8.4)]:
    ax.axvline(x, color=c, lw=1.8, ls="--" if c != AQUA else "-")
    ax.annotate(lab, xy=(x, yy), xytext=(x + 60, yy), fontsize=8.5, color=c, va="center",
                arrowprops=None)
ax.set_xlabel("DLP (mGy·cm)"); ax.set_ylabel("number of exams")
ax.set_title("Where the 50 exams sit versus published reference levels", loc="left", fontsize=11, color=INK)
save(fig, "f4_drl.png")

# F5 reduction options for a median patient
med = d.median(numeric_only=True)
res = recommend(p, float(med.BMI), 120, float(med.mAs), float(med.scan_length_cm),
                OptimizerConfig(kv_options=(80, 100, 120), length_cuts_cm=(0, 2)))
c = res["candidates"].copy()
c = c[c.DLP <= res["current"]["DLP"] * 1.01].sort_values("DLP")
fig, ax = plt.subplots(figsize=(6.6, 3.6))
cols = [AQUA if f else "#b9b9b4" for f in c.feasible]
ax.barh(c.option, c.DLP, color=cols, height=.65)
for y, (v, cn) in enumerate(zip(c.DLP, c.CNR_rel)):
    ax.text(v + 25, y, f"{v:.0f}", va="center", fontsize=9, color=INK2)
ax.axvline(res["current"]["DLP"], color=ORANGE, lw=2)
ax.text(res["current"]["DLP"], -0.75, "current plan", color=ORANGE, ha="right", fontsize=9)
ax.axvline(400, color=INK2, ls=":", lw=1.2)
ax.set_xlabel("predicted DLP (mGy·cm)")
ax.set_title(f"Options for a typical patient: BMI {med.BMI:.0f}, {med.mAs:.0f} mAs, 120 kV", loc="left",
             fontsize=11, color=INK, pad=16)
ax.invert_yaxis()
save(fig, "f5_options.png")

# F6 BO convergence: best feasible dose vs scan number, 12 seeds
traces, trues = [], []
for seed in range(12):
    sim = PhantomSimulator(seed=seed)
    cp = Campaign(name="s", kv_options=[80, 100, 120], ref_kVp=120, ref_mAs=float(round(med.mAs / 5) * 5))
    best = []
    for i in range(16):
        r = suggest(cp, f_dose)
        if r.next_scan is None: 
            best.append(best[-1]); continue
        ns = r.next_scan
        cp.add_scan(ns["kVp"], ns["mAs"], sim.scan(ns["kVp"], f_dose(ns["kVp"], ns["mAs"])))
        df = cp.scans_df(); tgt = cp.target()
        ok = df[df.CNR >= tgt] if tgt else df.iloc[0:0]
        best.append(float(min(f_dose(r_.kVp, r_.mAs) for r_ in ok.itertuples())) if len(ok) else np.nan)
    traces.append(best); trues.append(sim.true_optimum(cp, f_dose, cp.target())[2])
T = np.array(traces, float); ref = float(f_dose(120, round(med.mAs / 5) * 5))
fig, ax = plt.subplots(figsize=(6.4, 3.5))
x = np.arange(1, T.shape[1] + 1)
ax.fill_between(x, np.nanpercentile(T, 10, 0), np.nanpercentile(T, 90, 0), color=BLUE, alpha=.18, lw=0)
ax.plot(x, np.nanmedian(T, 0), color=BLUE, lw=2, label="best setting found (median of 12 runs)")
ax.axhline(np.mean(trues), color=AQUA, lw=1.8, ls="--", label="true optimum")
ax.axhline(ref, color=ORANGE, lw=1.8, label="starting protocol (120 kV)")
ax.set_xlabel("phantom scans performed"); ax.set_ylabel("CTDIvol (mGy)")
ax.set_title("Phantom optimisation: dose of the best setting that still meets the CNR target",
             loc="left", fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=9)
save(fig, "f6_bo.png")

# F7 bandit
runs = [run_bandit_simulation(p, 300, seed=s) for s in range(5)]
sav = np.array([[(1 - r.CTDIvol / r.ctdi_ref).rolling(40, min_periods=10).mean()] for r in runs]).squeeze() * 100
bad = np.array([[((r.score < 3) * 1.0).rolling(40, min_periods=10).mean()] for r in runs]).squeeze() * 100
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
x = np.arange(1, 301)
for ax, M, c, t, yl in ((axes[0], sav, BLUE, "Dose reduction vs current practice", "% lower CTDIvol"),
                        (axes[1], bad, RED, "Scans scored non-diagnostic", "% of scans")):
    ax.fill_between(x, np.nanpercentile(M, 10, 0), np.nanpercentile(M, 90, 0), color=c, alpha=.18, lw=0)
    ax.plot(x, np.nanmedian(M, 0), color=c, lw=2)
    ax.set_xlabel("patients scanned and scored"); ax.set_ylabel(yl)
    ax.set_title(t + "  (rolling 40, 5 runs)", loc="left", fontsize=10.5, color=INK)
axes[1].axhline(3, color=INK2, ls=":", lw=1.2); axes[1].text(5, 3.2, "3% target allowance", fontsize=8.5, color=INK2)
save(fig, "f7_bandit.png")
print("DONE", res["best"]["kVp"], res["best"]["mAs"], round(res["current"]["DLP"]), round(res["best"]["DLP"]))
