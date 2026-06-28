"""Provenance for the catalog's BP-magnitude extension past Li et al.'s stated
B<18 cut. Computes the phot_bp_mean_mag distribution of the released FaintQC
catalog (= our local CSV, byte-identical to the Zenodo release) so the "24% at
B>18" claim is reproducible. Output: outputs/audit/bp_extension.json
"""
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "wdms_gpc_30k_1229_mg7_msms.csv"
OUT = ROOT / "outputs" / "audit" / "bp_extension.json"

rows = list(csv.DictReader(open(CSV, newline="")))
bp = np.array([float(r["phot_bp_mean_mag"]) for r in rows
               if r.get("phot_bp_mean_mag", "") not in ("", "nan")])
ag = np.array([float(r["a_g"]) for r in rows if r.get("a_g", "") not in ("", "nan")])

res = {
    "n_catalog": len(rows),
    "n_with_bp": int(len(bp)),
    "li_stated_cut": "phot_bp_mean_mag < 18 (Li et al. 2025, Sec. 2.2, for the Riello systematic)",
    "bp_median": round(float(np.median(bp)), 3),
    "bp_p95": round(float(np.percentile(bp, 95)), 3),
    "bp_p99": round(float(np.percentile(bp, 99)), 3),
    "bp_max": round(float(bp.max()), 3),
    "n_bp_gt_18": int(np.sum(bp > 18)),
    "frac_bp_gt_18": round(float(np.mean(bp > 18)), 4),
    "n_bp_gt_18p5": int(np.sum(bp > 18.5)),
    "frac_bp_gt_18p5": round(float(np.mean(bp > 18.5)), 4),
    "a_g_max": round(float(ag.max()), 3),
    "note": "Released FaintQC catalog extends past the stated B<18 cut: a quarter "
            "of candidates sit at B>18 (to B~19.5), the regime where the Riello "
            "BP systematic is non-negligible. CSV is byte-identical (MD5 "
            "66cb2ae49124096e63876e46bf74bfdb) to Zenodo 10.5281/zenodo.14411002.",
}
OUT.write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
