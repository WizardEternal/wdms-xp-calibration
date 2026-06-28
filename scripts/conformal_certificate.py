"""Conformal / purity certificate on REAL labels (GALEX FUV as WD-presence truth).

FUV detection is a near-sufficient WD-presence indicator: a hot WD emits strongly in FUV,
a cool single MS essentially does not. So FUV-detected => a hot component is really there
(a purity FLOOR; FUV non-detection is ambiguous due to GALEX-AIS depth, so it is NOT a
clean negative - caveat). We:
  (1) certify the WD-present FRACTION (FUV-detection rate) with Wilson 95% binomial CIs,
      overall and split by prob_binary tercile and by flag_wdmsfit_in (on/off WD sequence);
  (2) demonstrate conformal selection (Jin & Candes) controls FDR on real data, using
      prob_binary as the score and FUV-detection as the label (validity check of the method,
      even though prob_binary turns out to be a weak WD-presence score).

Caveats: GALEX coverage 12.2% (UV-bright bias); FUV non-detection != no WD; observed mags.
"""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "wdms_gpc_30k_1229_mg7_msms.csv"
OUT = ROOT / "outputs" / "audit"
import sys
sys.path.insert(0, str(ROOT / "src"))
from sbiwdms import audit  # conformal_pvalues, bh_select  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - half) / d, 4), round((c + half) / d, 4))


def galex_merge(ra, dec, n):
    import astropy.units as u
    from astropy.table import Table
    from astroquery.xmatch import XMatch
    up = Table({"myid": np.arange(n), "ra": ra, "dec": dec})
    xm = XMatch.query(cat1=up, cat2="vizier:II/335/galex_ais",
                      max_distance=3 * u.arcsec, colRA1="ra", colDec1="dec")
    fuvc = next((c for c in xm.colnames if c.upper() == "FUVMAG"), None)
    ang = np.asarray(xm["angDist"]); mid = np.asarray(xm["myid"]).astype(int)
    fuv = np.full(n, np.nan); covered = np.zeros(n, bool)
    for k in np.argsort(ang):
        m = mid[k]
        if covered[m]:
            continue
        covered[m] = True
        v = xm[fuvc][k]
        try:
            fuv[m] = float(v)
        except (TypeError, ValueError):
            fuv[m] = np.nan
    return covered, np.isfinite(fuv)


def main():
    rows = list(csv.DictReader(open(CSV, newline="")))
    n = len(rows)
    prob = np.array([float(r["prob_binary"]) for r in rows])
    wdin = np.array([str(r["flag_wdmsfit_in"]).strip().lower() == "true" for r in rows])
    ra = np.array([float(r["ra"]) for r in rows]); dec = np.array([float(r["dec"]) for r in rows])

    covered, fuv_det = galex_merge(ra, dec, n)
    cov = covered  # has any GALEX match
    res = {"n": n, "galex_covered": int(cov.sum()), "fuv_detected": int(fuv_det.sum())}

    # (1) WD-present fraction (FUV-det) with Wilson CIs
    def cert(mask, label):
        m = mask & cov
        k = int((fuv_det & m).sum()); nn = int(m.sum())
        lo, hi = wilson(k, nn)
        return {"label": label, "n_galex": nn, "fuv_det": k,
                "fuv_det_frac": round(k / nn, 4) if nn else None, "wilson95": [lo, hi]}

    terc = np.quantile(prob, [1/3, 2/3])
    res["purity_floor_fuv"] = {
        "ALL_galex_covered": cert(np.ones(n, bool), "all GALEX-covered"),
        "prob_binary_low_third": cert(prob <= terc[0], f"prob<= {terc[0]:.3f}"),
        "prob_binary_mid_third": cert((prob > terc[0]) & (prob <= terc[1]), "mid third"),
        "prob_binary_high_third": cert(prob > terc[1], f"prob> {terc[1]:.3f}"),
        "wd_on_sequence": cert(wdin, "flag_wdmsfit_in=1 (on seq)"),
        "wd_off_sequence": cert(~wdin, "flag_wdmsfit_in=0 (off seq)"),
        "reading": "FUV-det frac is a certified WD-present FLOOR. If it is ~flat across prob_binary "
                   "terciles, Li's score does NOT rank true WD presence.",
    }

    # (2) conformal selection FDR validity on real data (score=prob_binary, label=fuv_det)
    rng = np.random.default_rng(0)
    idx = np.where(cov)[0]
    rng.shuffle(idx)
    half = len(idx) // 2
    cal, test = idx[:half], idx[half:]
    cal_neg = cal[~fuv_det[cal]]                 # calibration NEGATIVES (FUV non-detected)
    pvals = audit.conformal_pvalues(prob[cal_neg], prob[test])
    for q in (0.1, 0.2, 0.3):
        sel = audit.bh_select(pvals, fdr=q)
        if len(sel):
            fdp = float((~fuv_det[test[sel]]).mean())   # fraction selected that are FUV-non-det
            res.setdefault("conformal_selection", {})[f"q={q}"] = {
                "n_selected": int(len(sel)), "empirical_FDP_vs_fuv": round(fdp, 4),
                "note": "FDP uses FUV-non-det as 'false'; since non-det is ambiguous this is an UPPER "
                        "bound on true FDP. Controlled <= q validates the machinery on real data.",
            }
        else:
            res.setdefault("conformal_selection", {})[f"q={q}"] = {"n_selected": 0}

    (OUT / "conformal_certificate.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
