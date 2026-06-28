"""Real-data validation of Li's 30k using external ground truth.

(1) Spectroscopic confirmation (robust, no UV needed): cross-match the 30k to SDSS
    (Rebassa-Mansergas+2016) and LAMOST (Ren+2018) confirmed WD-MS. Among the confirmed
    subset, compare flag_MSMS / flag_wdmsfit_in / prob_binary / chi2_diff_renorm vs the
    full catalog. Key question: does Li's own contamination flag (flag_MSMS) ANTI-correlate
    with independent spectroscopic confirmation? (validates or undercuts the flag).
(2) GALEX NUV cross-match (CDS X-Match) as a supporting UV check, if reachable.

Writes outputs/audit/groundtruth.json and a labeled subset outputs/audit/confirmed_wdms.csv.
Robust: each external step is independent and logged; never hard-crashes.
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "wdms_gpc_30k_1229_mg7_msms.csv"
OUT = ROOT / "outputs" / "audit"


def load_li():
    rows = list(csv.DictReader(open(CSV, newline="")))
    def fcol(name):
        out = []
        for r in rows:
            try: out.append(float(r[name]))
            except (ValueError, KeyError): out.append(np.nan)
        return np.array(out)
    def bcol(name):
        return np.array([str(r[name]).strip().lower() == "true" for r in rows])
    return rows, fcol, bcol


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier
    Vizier.ROW_LIMIT = -1

    rows, fcol, bcol = load_li()
    ra, dec = fcol("ra"), fcol("dec")
    prob = fcol("prob_binary")
    cdr = fcol("chi2_diff_renorm")
    msms = bcol("flag_MSMS")
    wdin = bcol("flag_wdmsfit_in")
    n = len(rows)
    li = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

    # (1) spectroscopic confirmation cross-match
    confirmed = np.zeros(n, bool)
    per_cat = {}
    for name, vid in [("RM2016", "J/MNRAS/458/3808"), ("Ren2018", "J/MNRAS/477/4641")]:
        try:
            tab = Vizier.get_catalogs(vid)[0]
            rc = next(c for c in tab.colnames if c.upper() in ("RAJ2000", "RA_ICRS", "_RAJ2000"))
            dc = next(c for c in tab.colnames if c.upper() in ("DEJ2000", "DE_ICRS", "_DEJ2000"))
            cc = SkyCoord(ra=tab[rc], dec=tab[dc], unit=(u.deg, u.deg))
            idx, sep, _ = cc.match_to_catalog_sky(li)
            hit = sep.arcsec < 1.5
            matched_li = np.unique(idx[hit])
            confirmed[matched_li] = True
            per_cat[name] = int(hit.sum())
        except Exception as e:
            per_cat[name] = f"FAIL: {type(e).__name__}"

    nconf = int(confirmed.sum())

    def summ(mask, label):
        return {
            "label": label, "n": int(mask.sum()),
            "flag_MSMS_frac": round(float(msms[mask].mean()), 4) if mask.any() else None,
            "flag_wdmsfit_in_frac": round(float(wdin[mask].mean()), 4) if mask.any() else None,
            "prob_binary_median": round(float(np.nanmedian(prob[mask])), 4) if mask.any() else None,
            "chi2_diff_renorm_median": round(float(np.nanmedian(cdr[mask])), 4) if mask.any() else None,
            "chi2_diff_renorm_frac_lt0": round(float((cdr[mask] < 0).mean()), 4) if mask.any() else None,
        }

    res = {
        "n_catalog": n,
        "spectroscopic_confirmations": {**per_cat, "unique_in_30k": nconf},
        "compare_confirmed_vs_full": {
            "FULL_catalog": summ(np.ones(n, bool), "full 30k"),
            "CONFIRMED_WDMS": summ(confirmed, "spectroscopically-confirmed WD-MS"),
            "NOT_confirmed": summ(~confirmed, "rest"),
        },
        "key_question": (
            "If flag_MSMS truly marks contaminants, the spectroscopically-CONFIRMED WD-MS "
            "should have a LOWER flag_MSMS fraction than the full catalog's 25.3%. Compare above."
        ),
    }

    # (2) GALEX via CDS X-Match (supporting; optional)
    try:
        from astroquery.xmatch import XMatch
        from astropy.table import Table
        up = Table({"myid": np.arange(n), "ra": ra, "dec": dec})
        xm = XMatch.query(cat1=up, cat2="vizier:II/335/galex_ais",
                          max_distance=3 * u.arcsec, colRA1="ra", colDec1="dec")
        ncov = len(np.unique(xm["myid"])) if len(xm) else 0
        nuvcol = next((c for c in xm.colnames if "NUV" in c.upper() and "mag" in c.lower()), None)
        res["galex"] = {"matched_rows": len(xm), "unique_sources_with_galex": int(ncov),
                        "coverage_frac": round(ncov / n, 4), "nuv_col": nuvcol,
                        "cols_sample": list(xm.colnames)[:15]}
    except Exception as e:
        res["galex"] = {"FAIL": f"{type(e).__name__}: {str(e)[:150]}"}

    (OUT / "groundtruth.json").write_text(json.dumps(res, indent=2))
    # save labeled subset for the conformal step
    with open(OUT / "confirmed_wdms.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["row", "source_id", "prob_binary", "confirmed"])
        for i in range(n):
            if confirmed[i]:
                w.writerow([i, rows[i]["source_id"], prob[i], 1])
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
