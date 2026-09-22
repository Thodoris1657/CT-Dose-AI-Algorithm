"""Bayesian optimisation of kV / mAs on a phantom.

Goal:  minimise dose (CTDIvol)  subject to  CNR(kV, mAs) >= CNR_target.

* Dose is known and cheap to evaluate: CTDIvol comes from the fitted scanner law
  (or from the console reading you type in).
* CNR is the unknown, noisy, expensive function. Each phantom scan measures it once.

Surrogate model for log CNR, physics-informed
---------------------------------------------
    log CNR = b0 - γ·log(kV/120) + δ·log(CTDIvol) + GP residual
Physics says γ ≈ 1.2–1.5 (iodine contrast) and δ ≈ 0.5 (quantum noise ∝ 1/√dose).
These are Bayesian priors, so the model is useful after 2–3 scans. The data
then corrects them, and a Gaussian process captures anything the formula
misses (e.g. photon starvation at low kV or detector/reconstruction effects).

Acquisition: constrained expected improvement
---------------------------------------------
    a(x) = max(0, dose_best_feasible − dose(x)) · P(CNR(x) ≥ target)
The next scan maximises a(x). Before a feasible point exists, it picks the
lowest-dose point that is likely (P ≥ 0.5) feasible. The campaign has converged
when no untested setting offers a meaningful expected dose reduction.

Output:
  * the next setting to scan,
  * the recommended protocol (lowest dose with P(feasible) ≥ 0.95),
  * the calibrated γ, which the "Reduce dose" optimiser can use instead of the
    literature default.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

KV_REF = 120.0
PHANTOM_DIR = Path(__file__).resolve().parent.parent / "data" / "phantom"


@dataclass
class Campaign:
    name: str
    exam: str = "CCTA"
    phantom: str = "cardiac phantom, medium size"
    kv_options: list = field(default_factory=lambda: [80, 100, 120])
    mas_min: float = 50.0
    mas_max: float = 1500.0
    cnr_target: float | None = None        # absolute CNR target; if None → tau × CNR of the reference scan
    tau: float = 1.0                       # fraction of reference CNR (used when cnr_target is None)
    ref_kVp: float = 120.0                 # current clinical protocol = first scan
    ref_mAs: float = 750.0
    scans: list = field(default_factory=list)   # dicts: kVp, mAs, CNR, CTDIvol, note

    # ------------------------------------------------------------- persistence
    def path(self, directory: Path | None = None) -> Path:
        d = Path(directory or PHANTOM_DIR)
        d.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.name)
        return d / f"{safe}.json"

    def save(self, directory: Path | None = None) -> Path:
        p = self.path(directory)
        p.write_text(json.dumps(asdict(self), indent=1, ensure_ascii=False), encoding="utf-8")
        return p

    @staticmethod
    def load(path) -> "Campaign":
        return Campaign(**json.loads(Path(path).read_text(encoding="utf-8")))

    @staticmethod
    def list(directory: Path | None = None) -> dict[str, Path]:
        d = Path(directory or PHANTOM_DIR)
        return {p.stem: p for p in sorted(d.glob("*.json"))} if d.exists() else {}

    def scans_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.scans, columns=["kVp", "mAs", "CNR", "CTDIvol", "note"])

    def add_scan(self, kVp, mAs, CNR, CTDIvol=None, note=""):
        self.scans.append(dict(kVp=float(kVp), mAs=float(mAs), CNR=float(CNR),
                               CTDIvol=None if CTDIvol in (None, "") or pd.isna(CTDIvol) else float(CTDIvol),
                               note=str(note)))

    def target(self) -> float | None:
        if self.cnr_target:
            return float(self.cnr_target)
        ref = [s for s in self.scans if abs(s["kVp"] - self.ref_kVp) < 1e-6 and abs(s["mAs"] - self.ref_mAs) < 1e-6]
        if ref:
            return self.tau * float(np.mean([s["CNR"] for s in ref]))
        return None


# ----------------------------------------------------------------------------- surrogate
class CNRSurrogate:
    """Bayesian linear physics model + GP residual on log CNR."""

    PRIOR_MEAN = np.array([0.0, 1.35, 0.5])   # [b0, γ, δ]
    PRIOR_SD = np.array([10.0, 0.5, 0.15])

    def __init__(self, noise_sd: float = 0.04):
        self.noise_sd = noise_sd

    @staticmethod
    def design(kv, ctdi):
        kv, ctdi = np.asarray(kv, float), np.asarray(ctdi, float)
        return np.column_stack([np.ones_like(kv), -np.log(kv / KV_REF), np.log(ctdi)])

    def fit(self, kv, ctdi, cnr):
        X, y = self.design(kv, ctdi), np.log(np.asarray(cnr, float))
        s2 = self.noise_sd ** 2
        P0 = np.diag(1 / self.PRIOR_SD ** 2)
        A = P0 + X.T @ X / s2
        self.cov_ = np.linalg.inv(A)
        self.mean_ = self.cov_ @ (P0 @ self.PRIOR_MEAN + X.T @ y / s2)
        r = y - X @ self.mean_
        self.gp_ = None
        if len(y) >= 6:
            k = ConstantKernel(0.01, (1e-5, 1)) * RBF([0.3, 0.5], (0.05, 10)) + WhiteKernel(s2, (1e-5, 0.1))
            self.gp_ = GaussianProcessRegressor(k, normalize_y=False, random_state=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.gp_.fit(self._gpx(kv, ctdi), r)
        self.n_ = len(y)
        return self

    @staticmethod
    def _gpx(kv, ctdi):
        return np.column_stack([np.log(np.asarray(kv, float) / KV_REF), np.log(np.asarray(ctdi, float))])

    def predict(self, kv, ctdi):
        """Return mean and sd of log CNR."""
        X = self.design(kv, ctdi)
        mu = X @ self.mean_
        var = np.einsum("ij,jk,ik->i", X, self.cov_, X)          # latent (noise-free) uncertainty
        if self.gp_ is not None:
            m2, s2 = self.gp_.predict(self._gpx(kv, ctdi), return_std=True)
            white = self.gp_.kernel_.k2.noise_level                  # remove measurement noise
            mu, var = mu + m2, var + np.clip(s2 ** 2 - white, 0, None)
        return mu, np.sqrt(var)

    @property
    def gamma(self) -> float:
        return float(self.mean_[1])

    @property
    def gamma_sd(self) -> float:
        return float(np.sqrt(self.cov_[1, 1]))

    @property
    def delta(self) -> float:
        return float(self.mean_[2])


# ----------------------------------------------------------------------------- optimiser
@dataclass
class BOResult:
    next_scan: dict | None
    recommended: dict | None
    best_observed: dict | None
    converged: bool
    reason: str
    grid: pd.DataFrame
    surrogate: CNRSurrogate | None
    target: float | None


def dose_fn_from_predictor(pred):
    """CTDIvol(kV, mAs) from the trained scanner law."""
    k = float(np.exp(pred.physics_ctdi.intercept_))
    n = pred.physics_ctdi.kv_exponent_
    return lambda kv, mas: k * np.asarray(mas, float) * (np.asarray(kv, float) / KV_REF) ** n


def suggest(c: Campaign, dose_fn, n_grid: int = 80, pof_recommend: float = 0.95,
            min_gain_frac: float = 0.02, min_scans: int = 8) -> BOResult:
    df = c.scans_df()
    empty = pd.DataFrame()
    # --- step 0: reference scan first (defines the CNR target when relative)
    if c.target() is None:
        nxt = dict(kVp=c.ref_kVp, mAs=c.ref_mAs, CTDIvol=float(dose_fn(c.ref_kVp, c.ref_mAs)),
                   why="Reference scan with the current clinical protocol; it defines the CNR target.")
        return BOResult(nxt, None, None, False, "waiting for reference scan", empty, None, None)

    tgt = c.target()
    df["CTDI_used"] = [r.CTDIvol if pd.notna(r.CTDIvol) else float(dose_fn(r.kVp, r.mAs)) for r in df.itertuples()]
    sur = CNRSurrogate().fit(df.kVp, df.CTDI_used, df.CNR)

    # candidate grid
    mas = np.geomspace(c.mas_min, c.mas_max, n_grid)
    G = pd.DataFrame([(kv, m) for kv in c.kv_options for m in mas], columns=["kVp", "mAs"])
    G["CTDIvol"] = dose_fn(G.kVp, G.mAs)
    mu, sd = sur.predict(G.kVp, G.CTDIvol)
    G["CNR_mean"], G["CNR_lo"], G["CNR_hi"] = np.exp(mu), np.exp(mu - 1.64 * sd), np.exp(mu + 1.64 * sd)
    G["PoF"] = norm.cdf((mu - np.log(tgt)) / sd)

    feas_obs = df[df.CNR >= tgt]
    best_obs = feas_obs.sort_values("CTDI_used").iloc[0].to_dict() if len(feas_obs) else None

    # recommendation: lowest dose with high posterior probability of meeting target
    ok = G[G.PoF >= pof_recommend]
    rec = ok.sort_values("CTDIvol").iloc[0].to_dict() if len(ok) else None

    # acquisition
    if best_obs is not None:
        best_dose = best_obs["CTDI_used"]
        G["acq"] = np.clip(best_dose - G.CTDIvol, 0, None) * G.PoF
    else:
        best_dose = np.inf
        G["acq"] = np.where(G.PoF >= 0.5, 1.0 / G.CTDIvol, G.PoF * 1e-6)

    # avoid re-proposing (almost) identical settings
    for r in df.itertuples():
        near = (G.kVp == r.kVp) & (np.abs(np.log(G.mAs / r.mAs)) < 0.03)
        G.loc[near, "acq"] *= 0.1
    i = int(G.acq.idxmax())
    cand = G.loc[i].to_dict()
    gain = (best_dose - cand["CTDIvol"]) / best_dose if np.isfinite(best_dose) else 1.0
    converged = bool(np.isfinite(best_dose) and len(df) >= min_scans
                     and (G.acq.max() < 0.1 * min_gain_frac * best_dose or gain < min_gain_frac))
    reason = ("converged: no untested setting is expected to lower dose by ≥ "
              f"{min_gain_frac*100:.0f}% while meeting the target") if converged else "exploring"
    nxt = None if converged else dict(
        kVp=float(cand["kVp"]), mAs=float(round(cand["mAs"] / 5) * 5), CTDIvol=float(cand["CTDIvol"]),
        CNR_pred=float(cand["CNR_mean"]), PoF=float(cand["PoF"]),
        why=("expected dose saving × probability of meeting the CNR target is highest here"
             if best_obs is not None else "lowest-dose setting that is more likely than not to meet the target"))
    return BOResult(nxt, rec, best_obs, converged, reason, G, sur, tgt)


# ----------------------------------------------------------------------------- simulator
class PhantomSimulator:
    """Synthetic phantom with a 'true' CNR response (for demos and tests).

    True log CNR = c0 − γ·u + 0.5·log CTDI − starvation(u)  + noise,
    where the starvation term penalises very low kV (non-linear, not in the prior).
    """

    def __init__(self, gamma=1.5, c0=np.log(2.2), starvation=1.2, noise=0.03, seed=0):
        self.gamma, self.c0, self.starv, self.noise = gamma, c0, starvation, noise
        self.rng = np.random.RandomState(seed)

    def true_log_cnr(self, kv, ctdi):
        u = np.log(np.asarray(kv, float) / KV_REF)
        return self.c0 - self.gamma * u + 0.5 * np.log(ctdi) - self.starv * np.clip(-u - 0.25, 0, None) ** 1.5

    def scan(self, kv, ctdi):
        return float(np.exp(self.true_log_cnr(kv, ctdi) + self.rng.normal(0, self.noise)))

    def true_optimum(self, c: Campaign, dose_fn, target):
        mas = np.geomspace(c.mas_min, c.mas_max, 4000)
        best = None
        for kv in c.kv_options:
            d = dose_fn(kv, mas)
            ok = np.exp(self.true_log_cnr(kv, d)) >= target
            if ok.any():
                j = np.argmax(ok)
                if best is None or d[j] < best[2]:
                    best = (kv, mas[j], float(d[j]))
        return best


def run_simulation(dose_fn, max_scans=30, seed=0, kv_options=(80, 100, 120), ref=(120, 750)) -> dict:
    sim = PhantomSimulator(seed=seed)
    c = Campaign(name=f"sim{seed}", kv_options=list(kv_options), ref_kVp=ref[0], ref_mAs=ref[1])
    history = []
    for i in range(max_scans):
        r = suggest(c, dose_fn)
        if r.converged or r.next_scan is None:
            break
        s = r.next_scan
        c.add_scan(s["kVp"], s["mAs"], sim.scan(s["kVp"], dose_fn(s["kVp"], s["mAs"])))
        history.append(s)
    r = suggest(c, dose_fn)
    true = sim.true_optimum(c, dose_fn, c.target())
    rec = r.recommended
    true_cnr_rec = float(np.exp(sim.true_log_cnr(rec["kVp"], rec["CTDIvol"]))) if rec else None
    return dict(n_scans=len(c.scans), converged=r.converged, recommended=rec, true_optimum=true,
                ref_dose=float(dose_fn(*ref)), target=c.target(), true_cnr_at_rec=true_cnr_rec,
                gamma_hat=r.surrogate.gamma, campaign=c)
