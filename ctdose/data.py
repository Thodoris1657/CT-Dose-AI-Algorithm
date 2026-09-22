"""Data loading, cleaning and auditing for CT dose datasets.

Canonical schema (one row = one exam):
    exam            str    exam / protocol name (e.g. "CCTA")
    BMI             float  kg/m^2 (patient property, NOT tunable)
    kVp             float  tube voltage
    mAs             float  tube current-time product as reported by the scanner
    scan_length_cm  float  scan length / "scan size" as reported
    CTDIvol         float  mGy       (target)
    DLP             float  mGy*cm    (target)
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["BMI", "kVp", "mAs", "scan_length_cm"]
TARGETS = ["CTDIvol", "DLP"]
SCHEMA = ["exam"] + FEATURES + TARGETS

# Header aliases (Greek / English / scanner exports) -> canonical names
_ALIASES = {
    "exam": ["exam", "εξεταση", "εξέταση", "protocol", "study"],
    "BMI": ["bmi", "βμι"],
    "kVp": ["kvp", "kv"],
    "mAs": ["mas", "effective mas", "eff mas"],
    "scan_length_cm": ["scan_length_cm", "scan size cm", "scan size", "scan length", "scan length cm", "length"],
    "CTDIvol": ["ctdivol", "ctdi", "ctdi vol"],
    "DLP": ["dlp", "total dlp"],
}

# Short, file-system-safe names for known Greek exam labels
EXAM_SHORT = {"ΥΤ ΣΤΕΦΑΝΙΟΓΡΑΦΙΑ": "CCTA"}


def _canon(col: str) -> str | None:
    c = re.sub(r"[\s_]+", " ", str(col).strip().lower())
    for k, alist in _ALIASES.items():
        if c in alist:
            return k
    return None


def _num(x):
    if isinstance(x, str):
        x = x.strip().replace(",", ".")
    return pd.to_numeric(x, errors="coerce")


def load_table(path: str | Path, default_exam: str = "CCTA") -> pd.DataFrame:
    """Load CSV / Excel / PDF into the canonical schema."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".pdf":
        df = parse_pdf(path, default_exam)
    elif suf in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep=None, engine="python")
    return standardize(df, default_exam)


def standardize(df: pd.DataFrame, default_exam: str = "CCTA") -> pd.DataFrame:
    ren = {c: _canon(c) for c in df.columns if _canon(c)}
    df = df.rename(columns=ren)
    if "exam" not in df:
        df["exam"] = default_exam
    df["exam"] = df["exam"].astype(str).str.strip().replace(EXAM_SHORT)
    for c in FEATURES + TARGETS:
        if c not in df:
            df[c] = np.nan
        df[c] = df[c].map(_num).astype(float)
    return df[SCHEMA]


def parse_pdf(path: str | Path, default_exam: str = "CCTA") -> pd.DataFrame:
    """Parse the spreadsheet-printed-to-PDF layout (column blocks spread over pages).

    The PDF contains three blocks, each spanning several pages:
      [ΕΞΕΤΑΣΗ, ΒΜΙ, KV]  ->  [MAS, SCAN SIZE CM, DLP]  ->  [CTDI]
    Rows are aligned by order.
    """
    import pdfplumber

    with pdfplumber.open(str(path)) as pdf:
        text = "\n".join((p.extract_text() or "") for p in pdf.pages)
    blocks: dict[str, list[list[str]]] = {}
    headers: dict[str, list[str]] = {}
    cur = None
    for line in (l.strip() for l in text.splitlines()):
        if not line:
            continue
        toks = line.split()
        # a header line has no numeric token
        if not any(re.fullmatch(r"-?\d+([.,]\d+)?", t) for t in toks):
            cur = line
            blocks[cur] = []
            headers[cur] = _split_header(line)
            continue
        if cur is None:
            continue
        nums = [t for t in toks if re.fullmatch(r"-?\d+([.,]\d+)?", t)]
        label = " ".join(t for t in toks if t not in nums)
        blocks[cur].append(([label] if label else []) + nums)

    cols: dict[str, list] = {}
    for key, rows in blocks.items():
        hdr = headers[key]
        for i, h in enumerate(hdr):
            cols[h] = [r[i] if i < len(r) else None for r in rows]
    df = pd.DataFrame(cols)
    return standardize(df, default_exam)


def _split_header(line: str) -> list[str]:
    # Known multi-word headers first
    known = ["SCAN SIZE CM", "SCAN LENGTH CM", "SCAN LENGTH", "SCAN SIZE"]
    out, rest = [], line
    for k in known:
        if k in rest.upper():
            i = rest.upper().index(k)
            left, right = rest[:i], rest[i + len(k):]
            return left.split() + [k] + right.split()
    return rest.split()


def row_groups(df: pd.DataFrame) -> np.ndarray:
    """Group id per row: exact duplicate rows share a group (prevents CV leakage)."""
    key = df[FEATURES + TARGETS].round(4).astype(str).agg("|".join, axis=1)
    return key.map(lambda s: int(hashlib.md5(s.encode()).hexdigest()[:8], 16)).to_numpy()


def audit(df: pd.DataFrame) -> dict:
    """Data-quality & physics-consistency audit. Returns a dict of findings."""
    d = df.dropna(subset=FEATURES + TARGETS).copy()
    rep: dict = {"n_rows": int(len(df)), "n_complete": int(len(d))}
    dup = d.duplicated(FEATURES + TARGETS, keep=False)
    rep["duplicate_rows"] = (d.index[dup] + 1).tolist()   # 1-based row numbers
    rep["n_unique"] = int((~d.duplicated(FEATURES + TARGETS)).sum())
    rep["kVp_levels"] = sorted(d.kVp.unique().tolist())

    # Physics: CTDIvol / mAs should be a (kV-dependent) scanner constant
    d["ctdi_per_mas"] = d.CTDIvol / d.mAs
    per_kv = d.groupby("kVp").ctdi_per_mas.agg(["median", "min", "max", "count"])
    rep["ctdi_per_mAs_by_kV"] = per_kv.round(5).reset_index().to_dict("records")
    # Minority "protocol cluster": rows deviating >4% from kV-median yield
    med = d.groupby("kVp").ctdi_per_mas.transform("median")
    dev = d.ctdi_per_mas / med - 1
    rep["yield_outliers"] = (d.index[dev.abs() > 0.04] + 1).tolist()   # 1-based
    rep["yield_outlier_dev_pct"] = (dev[dev.abs() > 0.04] * 100).round(1).tolist()

    # Effective irradiated length implied by DLP/CTDIvol
    d["L_eff"] = d.DLP / d.CTDIvol
    rep["L_eff_cm"] = {k: round(float(v), 2) for k, v in d.L_eff.describe()[["mean", "std", "min", "max"]].items()}
    rep["corr_Leff_vs_scan_length"] = round(float(d.L_eff.corr(d.scan_length_cm)), 3)
    rep["corr_BMI_vs_scan_length"] = round(float(d.BMI.corr(d.scan_length_cm)), 3)
    # If scan_length were the true z-coverage, L_eff ≈ scan_length (+ overranging)
    rep["scan_length_is_z_coverage_suspect"] = bool(
        rep["corr_Leff_vs_scan_length"] < 0.5 or abs(d.L_eff.median() - d.scan_length_cm.median()) > 3
    )
    return rep
