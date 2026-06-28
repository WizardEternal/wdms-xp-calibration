"""Calibration audit of Li et al.'s published Gaia XP WD-MS catalog.

Treat Li's shipped `prob_binary` column as an uncalibrated score and certify the
purity / false-discovery rate within the published 30k sample against ground-truth
labels, via conformal selection (Jin & Candes 2023, arXiv:2210.01408), Mondrian over
CMD position, with weighted/covariate-shift conformal (Tibshirani et al. 2019,
arXiv:1904.06019) for the confirmed-subset-vs-full-pool shift. Light compute, no GPU.

The shipped file is the post-0.8 FaintQC sample (prob_binary>0.8 + QC + M_G0>7), so
this is not a replacement of Li's 0.8 cut and not a completeness statement: the
pre-cut pool (452,433 GPC sources) is not shipped. The deliverable is a purity/FDR
certificate within the 30k.

Catalog: data/wdms_gpc_30k_1229_mg7_msms.csv (30,131 rows). Key columns:
  prob_binary        Li's P_WDMS score (recalibrate THIS; arbitrary cut is >0.8)
  chi2_diff_renorm   renormalized Delta-chi2 (single vs binary fit)
  bp_rp0, abs_g0     dereddened CMD position (Mondrian binning axis)
  flag_MSMS          Li's MS-MS contamination flag (2615 systems)
  source_id          for ground-truth cross-match
  v-g,v-r,v-i,...    SkyMapper-like colors (the GPC features)

Ground truth: GALEX NUV-excess, SDSS RM+2013/16, LAMOST Ren+2018, Shahaf+2024,
Nayak+2023, cross-matched on source_id / position to a label subset against which
prob_binary is conformal-calibrated.
"""

from __future__ import annotations
import csv
import json
import math
from pathlib import Path

CATALOG = Path(__file__).resolve().parents[2] / "data" / "wdms_gpc_30k_1229_mg7_msms.csv"


# --- loading + characterization (stdlib only, runs with no env) ---------------

def load_catalog(path=CATALOG):
    """Read the catalog CSV into a dict of column-lists. Numeric where possible."""
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    cols = {k: [] for k in rows[0]}
    for r in rows:
        for k, v in r.items():
            cols[k].append(v)
    return cols


def _as_float(xs):
    out = []
    for x in xs:
        try:
            out.append(float(x))
        except ValueError:
            out.append(math.nan)
    return out


def characterize(cols):
    """First-look summary of the published catalog (no labels needed)."""
    p = _as_float(cols["prob_binary"])
    msms = [v.strip().lower() == "true" for v in cols["flag_MSMS"]]
    bp_rp0 = _as_float(cols["bp_rp0"])
    abs_g0 = _as_float(cols["abs_g0"])
    n = len(p)

    def frac(pred):
        return sum(1 for x in pred if x) / n

    thr = {t: sum(1 for x in p if x > t) for t in (0.5, 0.8, 0.9, 0.95)}
    return {
        "n_rows": n,
        "prob_binary": {
            "min": min(p), "max": max(p),
            "mean": sum(p) / n,
            "n_above": thr,
            "frac_above_0.8": thr[0.8] / n,
        },
        "flag_MSMS_true": sum(msms),
        "flag_MSMS_frac": frac(msms),
        "cmd_coverage": {
            "bp_rp0": [min(bp_rp0), max(bp_rp0)],
            "abs_g0": [min(abs_g0), max(abs_g0)],
        },
        "note": (
            "Shipped file is the POST-0.8 FaintQC sample (all prob_binary>0.8). "
            "flag_MSMS = the authors' model-based MS-MS classification (suspicion, not a "
            "confirmed contamination rate), retained in the catalog. The conformal-FDR "
            "audit certifies purity/FDR WITHIN this 30k vs ground-truth labels; it is "
            "NOT a replacement of Li's cut and NOT a completeness statement."
        ),
    }


# --- conformal selection with FDR (Jin & Candes 2023) -------------------------
# Implemented numerically; needs a labeled calibration set (clean negatives = single
# stars / non-WD-MS). One-sided: large score => more likely a true WD-MS binary.

def conformal_pvalues(cal_neg_scores, test_scores):
    """One-sided conformal p-values for each test point vs a calibration set of
    KNOWN NEGATIVES (Jin & Candes Eq. for selection). Larger score = stronger.
    p_j = (1 + #{i: V_i >= V_test_j}) / (n_cal + 1)."""
    import numpy as np
    cal = np.asarray(cal_neg_scores, float)
    test = np.asarray(test_scores, float)
    n = cal.size
    # count calibration negatives scoring >= each test point
    ge = (cal[None, :] >= test[:, None]).sum(axis=1)
    return (1.0 + ge) / (n + 1.0)


def bh_select(pvals, fdr=0.1):
    """Benjamini-Hochberg on conformal p-values -> selected indices at target FDR."""
    import numpy as np
    p = np.asarray(pvals, float)
    m = p.size
    order = np.argsort(p)
    thresh = fdr * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    if not passed.any():
        return np.array([], dtype=int)
    kmax = np.where(passed)[0].max()
    return order[: kmax + 1]


def mondrian_bins(bp_rp0, abs_g0, n_color=6, n_mag=6):
    """Coarse CMD grid label per source for class-conditional (Mondrian) conformal,
    so coverage is controlled within CMD region (review R3: confirmed subsets are
    CMD-localized vs the full pool)."""
    import numpy as np
    c = np.asarray(bp_rp0, float)
    g = np.asarray(abs_g0, float)
    cb = np.quantile(c, np.linspace(0, 1, n_color + 1))
    gb = np.quantile(g, np.linspace(0, 1, n_mag + 1))
    ci = np.clip(np.digitize(c, cb[1:-1]), 0, n_color - 1)
    gi = np.clip(np.digitize(g, gb[1:-1]), 0, n_mag - 1)
    return ci * n_mag + gi
