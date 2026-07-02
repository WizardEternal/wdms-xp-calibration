"""Catalog magnitude and WD-flux-fraction descriptors, committed to JSON.

Reads the committed Li 2025 GPC catalog CSV
(data/wdms_gpc_30k_1229_mg7_msms.csv, 30,131 rows, Zenodo 14569070) and emits
outputs/audit/catalog_fractions.json with:
  * frac_G_gt_17p5   : fraction with phot_g_mean_mag > 17.5
  * frac_BP_gt_17p5  : fraction with phot_bp_mean_mag > 17.5
  * frac_BP_gt_18    : fraction with phot_bp_mean_mag > 18   (the faint-BP regime;
                       companion to bp_extension.py's BP>18 / BP>19.5 audit)
  * median_wd_g_flux_fraction : median over the catalog of the WD's G-band flux
                       fraction f_WD/(f_WD+f_MS), from the per-component absolute
                       G magnitudes abs_g_wd (WD) and abs_g_md (M-dwarf/MS):
                         ratio = 10^(-0.4 * (abs_g_wd - abs_g_md)) = f_WD/f_MS,
                         frac  = ratio / (1 + ratio).
  * frac_wd_frac_below_0p024 : fraction of the catalog with WD flux fraction < 0.024
  * frac_wd_frac_in_0p05_0p30: fraction with WD flux fraction in [0.05, 0.30]
                       (the PCA-model mixing-fraction prior box, appendix.tex App. A.4).

Only stdlib + numpy; deterministic; no git actions.
Usage:  python scripts/catalog_fractions.py
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "wdms_gpc_30k_1229_mg7_msms.csv"
OUT = ROOT / "outputs" / "audit"


def fcol(rows, name):
    return np.array([float(r[name]) if r.get(name) not in ("", "nan", "NaN", None) else np.nan
                     for r in rows])


def main():
    rows = list(csv.DictReader(open(CSV, newline="")))
    n = len(rows)

    G = fcol(rows, "phot_g_mean_mag")
    BP = fcol(rows, "phot_bp_mean_mag")
    abs_g_wd = fcol(rows, "abs_g_wd")
    abs_g_md = fcol(rows, "abs_g_md")

    def frac(mask_bool, valid):
        v = valid
        return float(np.count_nonzero(mask_bool & v)) / float(np.count_nonzero(v))

    gval = np.isfinite(G)
    bval = np.isfinite(BP)

    frac_G_gt_17p5 = frac(G > 17.5, gval)
    frac_BP_gt_17p5 = frac(BP > 17.5, bval)
    frac_BP_gt_18 = frac(BP > 18.0, bval)

    # WD G-band flux fraction from the per-component absolute magnitudes.
    # magnitudes -> flux: f ~ 10^(-0.4 M); ratio_WD_over_MS = 10^(-0.4*(M_wd - M_md)).
    ratio = 10.0 ** (-0.4 * (abs_g_wd - abs_g_md))       # f_WD / f_MS
    wd_frac = ratio / (1.0 + ratio)                      # f_WD / (f_WD + f_MS)
    wval = np.isfinite(wd_frac)

    median_wd_g_flux_fraction = float(np.nanmedian(wd_frac[wval]))
    frac_below_0024 = frac(wd_frac < 0.024, wval)
    frac_in_005_030 = frac((wd_frac >= 0.05) & (wd_frac <= 0.30), wval)

    res = {
        "description": "Magnitude and WD-flux-fraction descriptors of the Li 2025 GPC "
                       "catalog (Zenodo 14569070), committed backing for note numbers "
                       "in 05-note/note.tex.",
        "catalog_csv": str(CSV.relative_to(ROOT)),
        "n_rows": n,
        "n_G_finite": int(np.count_nonzero(gval)),
        "n_BP_finite": int(np.count_nonzero(bval)),
        "n_wd_frac_finite": int(np.count_nonzero(wval)),
        "frac_G_gt_17p5": frac_G_gt_17p5,
        "frac_BP_gt_17p5": frac_BP_gt_17p5,
        "frac_BP_gt_18": frac_BP_gt_18,
        "wd_flux_fraction_definition": "f_WD/(f_WD+f_MS) with f ~ 10^(-0.4 M); "
                                       "ratio = 10^(-0.4*(abs_g_wd - abs_g_md)).",
        "median_wd_g_flux_fraction": median_wd_g_flux_fraction,
        "frac_wd_frac_below_0p024": frac_below_0024,
        "frac_wd_frac_in_0p05_0p30": frac_in_005_030,
        "counts": {
            "n_G_gt_17p5": int(np.count_nonzero((G > 17.5) & gval)),
            "n_BP_gt_17p5": int(np.count_nonzero((BP > 17.5) & bval)),
            "n_BP_gt_18": int(np.count_nonzero((BP > 18.0) & bval)),
            "n_wd_frac_below_0p024": int(np.count_nonzero((wd_frac < 0.024) & wval)),
            "n_wd_frac_in_0p05_0p30": int(np.count_nonzero((wd_frac >= 0.05) & (wd_frac <= 0.30) & wval)),
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "catalog_fractions.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
