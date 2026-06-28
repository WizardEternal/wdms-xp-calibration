"""Reliability-audit follow-ups: (a) selection-effect control for the flag_MSMS finding, and
(b) GALEX UV-excess comparison across flag groups (does UV corroborate the flags?).

(a) The confirmed WD-MS have flag_MSMS=40.9% vs 25.3% full. Is that over-flagging, or
    selection? Control: for each confirmed source, draw magnitude(G)+color(bp_rp0)-matched
    sources from the NON-confirmed catalog and measure THEIR flag_MSMS rate. If the matched
    control also ~40%, it's selection; if ~25%, flag_MSMS genuinely disagrees with spectroscopy.

(b) UV truth: a WD-MS has a hot WD => excess UV, esp. FUV (single MS stars are ~undetected
    in FUV). Compare GALEX NUV-G and FUV-detection across: confirmed WD-MS, flag_MSMS true/false,
    flag_wdmsfit_in true/false. UV should track real-WD presence, not the noisy MS-MS flag.
"""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "wdms_gpc_30k_1229_mg7_msms.csv"
OUT = ROOT / "outputs" / "audit"


def main():
    import astropy.units as u
    from astropy.table import Table
    from astroquery.xmatch import XMatch
    from sklearn.neighbors import NearestNeighbors

    rows = list(csv.DictReader(open(CSV, newline="")))
    n = len(rows)
    def fc(name):
        return np.array([float(r[name]) if r[name] not in ("", "nan") else np.nan for r in rows])
    def bc(name):
        return np.array([str(r[name]).strip().lower() == "true" for r in rows])
    G, bp_rp0, abs_g0 = fc("phot_g_mean_mag"), fc("bp_rp0"), fc("abs_g0")
    msms, wdin = bc("flag_MSMS"), bc("flag_wdmsfit_in")
    ra, dec = fc("ra"), fc("dec")

    confirmed = np.zeros(n, bool)
    for r in csv.DictReader(open(OUT / "confirmed_wdms.csv", newline="")):
        confirmed[int(r["row"])] = True
    nconf = int(confirmed.sum())

    res = {"n": n, "n_confirmed": nconf}

    # (a) selection-effect control: G+color-matched non-confirmed neighbors
    X = np.column_stack([G, bp_rp0])
    good = np.isfinite(X).all(1)
    Xs = (X - np.nanmean(X[good], 0)) / np.nanstd(X[good], 0)
    src = good & ~confirmed
    tgt = good & confirmed
    nn = NearestNeighbors(n_neighbors=10).fit(Xs[src])
    _, idx = nn.kneighbors(Xs[tgt])
    control_rows = np.unique(np.array(np.where(src)[0])[idx.ravel()])
    res["flag_MSMS_selection_control"] = {
        "confirmed_flag_MSMS_frac": round(float(msms[tgt].mean()), 4),
        "matched_control_flag_MSMS_frac": round(float(msms[control_rows].mean()), 4),
        "full_catalog_flag_MSMS_frac": round(float(msms.mean()), 4),
        "n_control": int(control_rows.size),
        "reading": "control ~0.41 => SELECTION effect; control ~0.25 => flag_MSMS truly disagrees w/ spectroscopy",
    }

    # (b) GALEX UV cross-match (NUV + FUV)
    try:
        up = Table({"myid": np.arange(n), "ra": ra, "dec": dec})
        xm = XMatch.query(cat1=up, cat2="vizier:II/335/galex_ais",
                          max_distance=3 * u.arcsec, colRA1="ra", colDec1="dec")
        nuvc = next((c for c in xm.colnames if c.upper() == "NUVMAG"), None)
        fuvc = next((c for c in xm.colnames if c.upper() == "FUVMAG"), None)
        ang = np.asarray(xm["angDist"]); mid = np.asarray(xm["myid"]).astype(int)
        nuv = np.full(n, np.nan); fuv = np.full(n, np.nan)
        order = np.argsort(ang)  # nearest first
        seen = set()
        for k in order:
            m = mid[k]
            if m in seen:
                continue
            seen.add(m)
            v = xm[nuvc][k]; nuv[m] = float(v) if np.ma.is_masked(v) is False and v is not None else np.nan
            if fuvc is not None:
                w = xm[fuvc][k]; fuv[m] = float(w) if np.ma.is_masked(w) is False and w is not None else np.nan
        has_nuv = np.isfinite(nuv)
        has_fuv = np.isfinite(fuv)
        nuv_g = nuv - G

        def grp(mask, name):
            mm = mask & has_nuv
            return {
                "label": name, "n_galex": int(mm.sum()),
                "median_NUV_minus_G": round(float(np.nanmedian(nuv_g[mm])), 3) if mm.any() else None,
                "FUV_detection_frac": round(float((has_fuv & mask).sum() / max((mask & has_nuv).sum(), 1)), 4),
            }
        res["galex_uv_by_group"] = {
            "coverage_frac_NUV": round(float(has_nuv.mean()), 4),
            "confirmed_WDMS": grp(confirmed, "confirmed WD-MS"),
            "flag_MSMS_true": grp(msms, "flag_MSMS=1 (suspected MS-MS)"),
            "flag_MSMS_false": grp(~msms, "flag_MSMS=0"),
            "wdfit_in_true": grp(wdin, "WD on cooling seq"),
            "wdfit_in_false": grp(~wdin, "WD OFF cooling seq (suspect)"),
            "reading": "Real WD presence => bluer NUV-G and higher FUV-detection. If flag_MSMS=1 looks "
                       "UV-similar to flag_MSMS=0, the flag is not cleanly separating WD-less systems.",
        }
    except Exception as e:
        res["galex_uv_by_group"] = {"FAIL": f"{type(e).__name__}: {str(e)[:150]}"}

    (OUT / "groundtruth_followup.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
