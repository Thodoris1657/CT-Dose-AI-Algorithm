"""Safe contextual bandit that learns the lowest acceptable dose from radiologist scores.

Loop (one patient at a time)
----------------------------
1. Context: the patient's BMI.
2. Arms: protocol settings (kVp × image-quality level). Each arm is turned into a
   concrete mAs by the scanner law, relative to the local reference dose for that BMI.
3. The bandit PROPOSES an arm. A radiologist APPROVES it or overrides it.
4. After the scan the radiologist SCORES diagnostic quality (1–4). A score ≥ 3
   counts as acceptable.
5. The model updates, and every scored exam is also added as a training row for
   the dose-prediction model (it contains BMI, kV, mAs, length, CTDIvol, DLP).

Model: Bayesian logistic regression (Laplace approximation)
------------------------------------------------------------
    logit P(acceptable) = w0 + w1·BMI_z + w2·log(CTDIvol/CTDI_ref) + w3·log(kV/120)
All patients share one model, so it learns from every scan, not one arm at a
time. The priors encode physics: more dose → better (w2 > 0), lower kV → better
iodine CNR at equal dose (w3 < 0), larger patient → worse (w1 < 0).

Safety (SafeOpt-style rules)
----------------------------
* The reference protocol (local median dose at 120 kV, or the BMI-appropriate
  kV at iso-CNR) is always allowed.
* An arm is allowed only if the pessimistic (10th-percentile) posterior
  P(acceptable) ≥ `safety_p`.
* An arm may not go more than `max_step` below the lowest dose already proven
  acceptable for similar BMI (±`bmi_band`), so the dose comes down step by step.
* kV must respect the BMI minimum-kV rule.
* A human approves every proposal.

Decision: Thompson sampling
---------------------------
Draw one plausible model from the posterior. Among the allowed arms, choose the
lowest-dose arm whose sampled P(acceptable) ≥ `target_p`. Randomness in the
draw provides exploration. Uncertainty shrinks with data, so proposals settle.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

KV_REF = 120.0
BMI_REF = 27.0
CLIN_DIR = Path(__file__).resolve().parent.parent / "data" / "clinical"

LOG_COLUMNS = ["timestamp", "exam", "patient_id", "BMI", "scan_length_cm", "proposed_kVp", "proposed_mAs",
               "proposal_mode", "kVp", "mAs", "approved_as_proposed", "CTDIvol", "DLP", "dose_source",
               "score", "acceptable", "reader", "comment"]


@dataclass
class BanditConfig:
    kv_options: tuple = (80, 100, 120)
    levels: tuple = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)   # dose fraction of iso-CNR reference at each kV
    gamma: float = 1.35               # iodine contrast exponent (use the phantom-calibrated value when available)
    target_p: float = 0.97            # desired P(acceptable)
    safety_p: float = 0.85            # pessimistic P(acceptable) must exceed this
    max_step: float = 0.15            # max fractional dose step below proven-acceptable dose
    bmi_band: float = 3.0
    accept_score: int = 3             # score ≥ this counts as acceptable (1–4 scale)
    apply_bmi_kv_rule: bool = True
    bmi_kv_rule: tuple = ((25.0, 80), (30.0, 100), (99.0, 120))


def min_kv(bmi, cfg: BanditConfig):
    for ub, kv in cfg.bmi_kv_rule:
        if bmi < ub:
            return kv
    return cfg.bmi_kv_rule[-1][1]


class BayesLogit:
    """Bayesian logistic regression, Laplace approximation, physics-informed prior."""

    # features: [1, BMI_z, log(CTDI/CTDI_ref), log(kV/120)]
    PRIOR_MEAN = np.array([4.0, -0.5, 6.0, -16.0])   # w3 ≈ -γ·(w2/0.5): iso-CNR arms equally likely acceptable
    PRIOR_SD = np.array([1.5, 1.0, 3.0, 6.0])

    def fit(self, F: np.ndarray, y: np.ndarray, iters: int = 50):
        m0, P0 = self.PRIOR_MEAN, np.diag(1 / self.PRIOR_SD ** 2)
        w = m0.copy()
        for _ in range(iters):
            p = expit(F @ w)
            g = F.T @ (y - p) - P0 @ (w - m0)
            H = -(F.T * (p * (1 - p))) @ F - P0
            step = np.linalg.solve(H, g)
            w = w - step
            if np.abs(step).max() < 1e-8:
                break
        p = expit(F @ w)
        H = -(F.T * (p * (1 - p))) @ F - P0
        self.mean_, self.cov_ = w, np.linalg.inv(-H)
        return self

    def sample(self, rng, n=1):
        return rng.multivariate_normal(self.mean_, self.cov_, size=n)

    def prob(self, F, W=None):
        W = self.mean_[None, :] if W is None else np.atleast_2d(W)
        return expit(F @ W.T)            # (n_points, n_draws)


def features(bmi, ctdi, kv, ctdi_ref):
    bmi, ctdi, kv, ctdi_ref = (np.asarray(a, float) for a in (bmi, ctdi, kv, ctdi_ref))
    return np.column_stack([np.ones_like(ctdi), (bmi - BMI_REF) / 7.0, np.log(ctdi / ctdi_ref), np.log(kv / KV_REF)])


@dataclass
class DoseBandit:
    pred: object                       # DosePredictor (scanner law + local peer reference)
    cfg: BanditConfig = field(default_factory=BanditConfig)
    log: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=LOG_COLUMNS))
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.RandomState(self.seed)
        k = float(np.exp(self.pred.physics_ctdi.intercept_))
        n = self.pred.physics_ctdi.kv_exponent_
        self._ctdi = lambda kv, mas: k * np.asarray(mas, float) * (np.asarray(kv, float) / KV_REF) ** n
        self._mas = lambda kv, ctdi: np.asarray(ctdi, float) / (k * (np.asarray(kv, float) / KV_REF) ** n)
        self.refit()

    # ------------------------------------------------------------------ reference
    def ctdi_ref(self, bmi) -> float:
        """Local reference CTDIvol at 120 kV for this BMI (peer median)."""
        return float(self.pred.peer_ctdi(bmi)[0.5])

    # ------------------------------------------------------------------ model
    def scored(self) -> pd.DataFrame:
        d = self.log.copy()
        d = d[pd.to_numeric(d.score, errors="coerce").notna()]
        return d

    def refit(self):
        d = self.scored()
        self.model = BayesLogit()
        if len(d):
            bmi = d.BMI.astype(float).to_numpy()
            ctdi = d.CTDIvol.astype(float).to_numpy()
            F = features(bmi, ctdi, d.kVp.astype(float), [self.ctdi_ref(b) for b in bmi])
            y = (d.score.astype(float).to_numpy() >= self.cfg.accept_score).astype(float)
        else:
            F, y = np.zeros((0, 4)), np.zeros(0)
        self.model.fit(F, y)
        return self

    # ------------------------------------------------------------------ arms
    def arms(self, bmi: float) -> pd.DataFrame:
        cr = self.ctdi_ref(bmi)
        rows = []
        for kv in self.cfg.kv_options:
            iso = cr * (kv / KV_REF) ** (2 * self.cfg.gamma)          # iso-CNR dose at this kV
            for lv in self.cfg.levels:
                ctdi = iso * lv
                rows.append(dict(kVp=kv, level=lv, CTDIvol=ctdi, mAs=float(self._mas(kv, ctdi)),
                                 CNR_rel=np.sqrt(lv)))
        A = pd.DataFrame(rows)
        A["mAs"] = (A.mAs / 5).round() * 5
        A["CTDIvol"] = self._ctdi(A.kVp, A.mAs)
        F = features(np.full(len(A), bmi), A.CTDIvol, A.kVp, np.full(len(A), cr))
        draws = self.model.prob(F, self.model.sample(self.rng, 400))
        A["p_mean"] = self.model.prob(F)[:, 0]
        A["p_lo"] = np.quantile(draws, 0.10, axis=1)
        A["p_hi"] = np.quantile(draws, 0.90, axis=1)
        return self._safety(A, bmi, cr)

    def _safety(self, A: pd.DataFrame, bmi: float, cr: float) -> pd.DataFrame:
        kvmin = min_kv(bmi, self.cfg) if self.cfg.apply_bmi_kv_rule else 0
        # reference arm: BMI-appropriate kV (≥ kvmin), level 1.0 (iso-CNR with local median)
        ref_kv = min([k for k in self.cfg.kv_options if k >= kvmin] or [max(self.cfg.kv_options)])
        A["is_reference"] = (A.kVp == ref_kv) & (A.level == 1.0)
        # proven-acceptable floor for similar BMI (in units of CTDI relative to iso-CNR at 120 kV)
        d = self.scored()
        floor = None
        if len(d):
            near = d[(d.BMI.astype(float) - bmi).abs() <= self.cfg.bmi_band]
            near = near[near.score.astype(float) >= self.cfg.accept_score]
            if len(near):
                eq = near.CTDIvol.astype(float) / (near.kVp.astype(float) / KV_REF) ** (2 * self.cfg.gamma)
                eq = eq / np.array([self.ctdi_ref(b) for b in near.BMI.astype(float)])
                floor = float(eq.min())
        A["cnr_equiv_level"] = A.CTDIvol / (cr * (A.kVp / KV_REF) ** (2 * self.cfg.gamma))
        step_ok = A.cnr_equiv_level >= ((floor if floor is not None else 1.0) * (1 - self.cfg.max_step) - 1e-9)
        reasons = []
        for r, s in zip(A.itertuples(), step_ok):
            why = []
            if r.kVp < kvmin:
                why.append(f"kV < {kvmin:g} for this BMI")
            if r.p_lo < self.cfg.safety_p and not r.is_reference:
                why.append(f"pessimistic P(acceptable) {r.p_lo:.2f} < {self.cfg.safety_p}")
            if not s and not r.is_reference:
                why.append("dose step too large vs proven level")
            reasons.append("; ".join(why))
        A["blocked_by"] = reasons
        A["allowed"] = (A.blocked_by == "") | A.is_reference
        return A

    # ------------------------------------------------------------------ decide
    def propose(self, bmi: float, explore: bool = True) -> dict:
        A = self.arms(bmi)
        allowed = A[A.allowed]
        if explore:
            w = self.model.sample(self.rng, 1)
            F = features(np.full(len(allowed), bmi), allowed.CTDIvol, allowed.kVp,
                         np.full(len(allowed), self.ctdi_ref(bmi)))
            ps = self.model.prob(F, w)[:, 0]
        else:
            ps = allowed.p_mean.to_numpy()
        good = allowed[ps >= self.cfg.target_p]
        if len(good):
            pick = good.sort_values("CTDIvol").iloc[0]
            mode = "explore (Thompson)" if explore else "exploit"
        else:
            pick = allowed[allowed.is_reference].iloc[0] if allowed.is_reference.any() else allowed.sort_values("CTDIvol").iloc[-1]
            mode = "reference (no arm meets target yet)"
        out = pick.to_dict()
        out["mode"] = mode
        out["ctdi_ref"] = self.ctdi_ref(bmi)
        out["saving_vs_ref_pct"] = (1 - out["CTDIvol"] / out["ctdi_ref"]) * 100   # vs local 120-kV median
        return out

    # ------------------------------------------------------------------ record
    def record(self, **row) -> None:
        row.setdefault("timestamp", dt.datetime.now().isoformat(timespec="seconds"))
        if row.get("score") not in (None, "") and not pd.isna(row.get("score")):
            row["acceptable"] = float(row["score"]) >= self.cfg.accept_score
        new = pd.DataFrame([{c: row.get(c) for c in LOG_COLUMNS}])
        self.log = new if self.log.empty else pd.concat([self.log, new], ignore_index=True)
        self.refit()

    def summary(self) -> dict:
        d = self.scored()
        if not len(d):
            return dict(n_scored=0)
        bmi = d.BMI.astype(float)
        ref = np.array([self.ctdi_ref(b) for b in bmi])
        return dict(n_scored=int(len(d)),
                    acceptable_rate=float((d.score.astype(float) >= self.cfg.accept_score).mean()),
                    mean_ctdi=float(d.CTDIvol.astype(float).mean()),
                    mean_saving_vs_local_median_pct=float((1 - d.CTDIvol.astype(float) / ref).mean() * 100),
                    w=self.model.mean_.round(3).tolist())

    # ------------------------------------------------------------------ persistence
    @staticmethod
    def log_path(exam: str, directory: Path | None = None) -> Path:
        d = Path(directory or CLIN_DIR)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{exam}_bandit_log.csv"

    def save_log(self, exam: str, directory: Path | None = None) -> Path:
        p = self.log_path(exam, directory)
        self.log.to_csv(p, index=False, encoding="utf-8-sig")
        return p

    @staticmethod
    def load_log(exam: str, directory: Path | None = None) -> pd.DataFrame:
        p = DoseBandit.log_path(exam, directory)
        if p.exists():
            d = pd.read_csv(p, encoding="utf-8-sig")
            for c in LOG_COLUMNS:
                if c not in d:
                    d[c] = np.nan
            return d[LOG_COLUMNS]
        return pd.DataFrame(columns=LOG_COLUMNS)


# ============================================================================ simulator
class ReaderSimulator:
    """Synthetic 'radiologist': acceptability driven by true iodine CNR (for demos/tests)."""

    def __init__(self, gamma=1.5, cnr50=0.62, slope=12.0, size_coef=0.10, seed=0):
        self.gamma, self.cnr50, self.slope, self.size = gamma, cnr50, slope, size_coef
        self.rng = np.random.RandomState(seed)

    def p_accept(self, bmi, kv, ctdi, ctdi_ref):
        # CNR relative to local-median 120 kV scan of an average patient
        log_cnr = -self.gamma * np.log(kv / KV_REF) + 0.5 * np.log(ctdi / ctdi_ref) \
                  - self.size * (bmi - BMI_REF) / 7.0
        starv = 3.0 * np.clip(np.log(KV_REF / kv) - 0.1, 0, None) * (bmi > 30)   # low kV fails in large patients
        return expit(self.slope * (log_cnr - np.log(self.cnr50)) - 4 * starv)

    def score(self, bmi, kv, ctdi, ctdi_ref):
        p = self.p_accept(bmi, kv, ctdi, ctdi_ref)
        ok = self.rng.rand() < p
        return int(self.rng.choice([3, 4])) if ok else int(self.rng.choice([1, 2]))


def run_bandit_simulation(pred, n_patients=300, seed=0, cfg: BanditConfig | None = None) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    bmis = pred.train_df.BMI.to_numpy()
    bandit = DoseBandit(pred, cfg or BanditConfig(), seed=seed)
    reader = ReaderSimulator(seed=seed)
    rows = []
    for i in range(n_patients):
        bmi = float(np.clip(rng.choice(bmis) + rng.normal(0, 1.5), 17, 60))
        prop = bandit.propose(bmi)
        cr = bandit.ctdi_ref(bmi)
        s = reader.score(bmi, prop["kVp"], prop["CTDIvol"], cr)
        bandit.log = pd.concat([bandit.log, pd.DataFrame([dict(
            timestamp=i, exam="SIM", BMI=bmi, kVp=prop["kVp"], mAs=prop["mAs"], CTDIvol=prop["CTDIvol"],
            score=s, acceptable=s >= 3, proposal_mode=prop["mode"])])], ignore_index=True)
        if i % 5 == 4 or i < 20:
            bandit.refit()
        rows.append(dict(i=i, BMI=bmi, kVp=prop["kVp"], CTDIvol=prop["CTDIvol"], ctdi_ref=cr,
                         p_true=reader.p_accept(bmi, prop["kVp"], prop["CTDIvol"], cr), score=s, mode=prop["mode"]))
    return pd.DataFrame(rows)
