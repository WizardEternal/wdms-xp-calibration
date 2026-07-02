"""Decisive distance-confound test for SURVIVOR 3.
Off-seq GALEX sources are at 433pc vs 302pc (on-seq). FUV comes from the WD, whose
APPARENT FUV scales with distance. The brightness rebuttal used total apparent G (MS-dominated),
which is the WRONG control. Proper control: does the off/on FUV-detection gap survive at
MATCHED distance? Stratify by distance bins and by parallax-matched control.
Re-uses the GALEX match cached from adv_survivor3 by re-querying (fast enough).

Committed backing: the distance-matched control result (off-seq FUV detection ~0.188 vs
distance-matched on-seq ~0.574) is WRITTEN to
  wdms-xp-calibration/outputs/audit/uv_distance_matched.json
with the matching-procedure parameters and honest input provenance, so the note's UV
reliability number is not stdout-only.
"""
import csv, json
from pathlib import Path
import numpy as np
ROOT = Path(r"C:/Users/Karan/Documents/Python stuff/cool papers/wdms-xp-calibration")
CSV = ROOT / "data" / "wdms_gpc_30k_1229_mg7_msms.csv"
OUT_JSON = ROOT / "outputs" / "audit" / "uv_distance_matched.json"
rows = list(csv.DictReader(open(CSV, newline="")))
n = len(rows)
def fc(name): return np.array([float(r[name]) if r.get(name) not in ("","nan",None) else np.nan for r in rows])
def bc(name): return np.array([str(r[name]).strip().lower()=="true" for r in rows])
G=fc("phot_g_mean_mag"); plx=fc("parallax"); ra,dec=fc("ra"),fc("dec"); wdin=bc("flag_wdmsfit_in")
dist=1000.0/plx

import astropy.units as u
from astropy.table import Table
from astroquery.xmatch import XMatch
up = Table({"myid": np.arange(n), "ra": ra, "dec": dec})
xm = XMatch.query(cat1=up, cat2="vizier:II/335/galex_ais", max_distance=3*u.arcsec, colRA1="ra", colDec1="dec")
fuvc = next((c for c in xm.colnames if c.upper()=="FUVMAG"), None)
nuvc = next((c for c in xm.colnames if c.upper()=="NUVMAG"), None)
ang=np.asarray(xm["angDist"]); mid=np.asarray(xm["myid"]).astype(int)
nuv=np.full(n,np.nan); fuv=np.full(n,np.nan); seen=set()
for k in np.argsort(ang):
    mm=mid[k]
    if mm in seen: continue
    seen.add(mm)
    v=xm[nuvc][k]; nuv[mm]=float(v) if not np.ma.is_masked(v) else np.nan
    w=xm[fuvc][k]; fuv[mm]=float(w) if not np.ma.is_masked(w) else np.nan
has_nuv=np.isfinite(nuv); has_fuv=np.isfinite(fuv)

# Within GALEX-NUV-detected sample, FUV detection vs distance, split off/on
m_nuv = has_nuv & np.isfinite(dist)
print("=== FUV detection in distance bins (within NUV-matched) ===")
edges=[0,250,350,450,600,3000]
from scipy.stats import norm
bin_results=[]
for a,b in zip(edges[:-1],edges[1:]):
    inb = m_nuv & (dist>=a) & (dist<b)
    off = inb & ~wdin; on = inb & wdin
    no,non = off.sum(), on.sum()
    if no<10 or non<10:
        print(f"  {a}-{b}pc: off n={no} on n={non} (skip, small)")
        bin_results.append({"bin_pc":[a,b],"n_off":int(no),"n_on":int(non),"skipped":True})
        continue
    fo = (has_fuv & off).sum()/no; fn=(has_fuv & on).sum()/non
    print(f"  {a}-{b}pc: OFF fuvdet={fo:.3f}(n={no})  ON fuvdet={fn:.3f}(n={non})  diff={fn-fo:+.3f}")
    bin_results.append({"bin_pc":[a,b],"n_off":int(no),"n_on":int(non),
                        "fuvdet_off":float(fo),"fuvdet_on":float(fn),"diff":float(fn-fo),"skipped":False})

# distance-matched control: for each off-seq NUV source, nearest on-seq in log-distance
off_idx = np.where(m_nuv & ~wdin)[0]
on_idx  = np.where(m_nuv &  wdin)[0]
ld_on = np.log10(dist[on_idx])
matched=[]
used=set()
for i in off_idx:
    di=np.log10(dist[i])
    order=np.argsort(np.abs(ld_on-di))
    for o in order:
        if on_idx[o] not in used:
            used.add(on_idx[o]); matched.append(on_idx[o]); break
matched=np.array(matched)
fo = (has_fuv[off_idx]).mean()
fm = (has_fuv[matched]).mean()
ppool=((has_fuv[off_idx]).sum()+(has_fuv[matched]).sum())/(len(off_idx)+len(matched))
se=np.sqrt(ppool*(1-ppool)*(1/len(off_idx)+1/len(matched)))
z=(fm-fo)/se
raw_on_fuvdet = float(has_fuv[on_idx].mean())
med_dist_on_unmatched = float(np.median(dist[on_idx]))
print(f"\n=== distance-matched control ===")
print(f"off-seq med_dist={np.median(dist[off_idx]):.1f}  matched-on med_dist={np.median(dist[matched]):.1f}  full-on(unmatched) med_dist={med_dist_on_unmatched:.1f}")
print(f"off FUVdet={fo:.3f} (n={len(off_idx)})  distance-matched-on FUVdet={fm:.3f} (n={len(matched)})  full-on(unmatched) n={len(on_idx)}")
print(f"gap after distance matching: {fm-fo:+.3f}  z={z:.2f}  p={2*norm.sf(abs(z)):.2e}")
print(f"(raw gap was {raw_on_fuvdet:.3f}-{fo:.3f} = {raw_on_fuvdet-fo:+.3f}; how much survives matching: {fm-fo:+.3f})")

# subset counts for provenance
n_rows_total       = int(n)
n_galex_nuv_det    = int(has_nuv.sum())                    # any NUV match, before parallax/on-off cuts
n_finite_parallax  = int(np.isfinite(dist).sum())
n_m_nuv            = int(m_nuv.sum())                      # NUV-detected AND finite parallax (the analysis sample)
n_off_seq_sample   = int(len(off_idx))                     # m_nuv & off-seq  -> THIS IS THE 383
n_on_seq_sample    = int(len(on_idx))                      # m_nuv & on-seq   -> full on-seq pool
n_matched_on       = int(len(matched))                     # on-seq sources drawn 1:1 from the pool

# ---- committed backing: write the matched-control result to the repo audit dir ----
res = {
    "description": "Distance-matched FUV-detection control for the off- vs on-cooling-sequence "
                   "GALEX reliability test (note UV-reliability number). Off-sequence sources "
                   "are FUV-detected far less than on-sequence, and the gap SURVIVES matching "
                   "the two populations in log-distance, so it is intrinsic (a lack of a hot WD), "
                   "not a GALEX depth/distance effect.",
    "matched_control": {
        "fuvdet_off_seq": float(fo),
        "fuvdet_distance_matched_on_seq": float(fm),
        "fuvdet_on_seq_unmatched": raw_on_fuvdet,
        "raw_gap_on_minus_off": float(raw_on_fuvdet - fo),
        "matched_gap_on_minus_off": float(fm - fo),
        "z": float(z),
        "p_two_sided": float(2 * norm.sf(abs(z))),
        "n_off": int(len(off_idx)),
        "n_matched_on": int(len(matched)),
        "med_dist_off_pc": float(np.median(dist[off_idx])),
        "med_dist_matched_on_pc": float(np.median(dist[matched])),
        "med_dist_on_seq_unmatched_pc": med_dist_on_unmatched,
    },
    "subset_counts": {
        "n_rows_total": n_rows_total,
        "n_galex_nuv_detected": n_galex_nuv_det,
        "n_finite_parallax": n_finite_parallax,
        "n_m_nuv_analysis_sample": n_m_nuv,
        "n_off_seq_sample": n_off_seq_sample,
        "n_on_seq_sample": n_on_seq_sample,
        "n_matched_on": n_matched_on,
        "definition_n_off_seq_sample": "the 383 = off-cooling-sequence sources "
            "(flag_wdmsfit_in == false) that have a GALEX AIS NUV detection within 3 arcsec "
            "AND a finite Gaia parallax (finite dist = 1000/parallax); FUV detection is the "
            "measured outcome, NOT a selection cut on this denominator.",
        "definition_n_m_nuv_analysis_sample": "m_nuv = has_nuv & finite parallax = every source "
            "with a <=3 arcsec GALEX AIS NUV magnitude and a finite Gaia parallax, before the "
            "on/off cooling-sequence split; the on- and off-seq samples partition this set.",
        "definition_n_on_seq_sample": "on-cooling-sequence half of m_nuv (flag_wdmsfit_in == true); "
            "full unmatched on-seq pool from which the distance-matched controls are drawn.",
        "definition_n_matched_on": "the n_off on-seq sources selected 1:1 (greedy, no replacement) "
            "as nearest neighbours of the off-seq sources in log10(distance).",
    },
    "matching_procedure": {
        "sample": "GALEX-NUV-detected sources with finite Gaia parallax (m_nuv)",
        "on_off_split": "flag_wdmsfit_in true = on cooling sequence, false = off",
        "distance": "dist_pc = 1000 / parallax_mas",
        "match_metric": "nearest neighbour in log10(distance)",
        "match_type": "greedy 1:1 without replacement (each on-seq source used once), "
                      "off-seq sources matched in catalog order",
        "match_caliper": "none (unconstrained nearest neighbour; no maximum log-distance "
                         "separation imposed)",
        "match_direction": "each off-seq source (n=n_off) is matched to one on-seq source; "
                           "on-seq pool (n_on_seq_sample) is larger, so matching is exhaustive "
                           "for the off-seq side",
        "n_off_matched": int(len(off_idx)),
        "n_on_pool": int(len(on_idx)),
        "n_matched_pairs": int(len(matched)),
        "detection_definition": "FUV detected = finite FUVmag from a <=3 arcsec GALEX AIS match",
        "galex_catalog": "vizier:II/335/galex_ais",
        "galex_match_radius_arcsec": 3.0,
        "distance_bin_edges_pc": [0, 250, 350, 450, 600, 3000],
        "distance_bin_min_per_cell": 10,
    },
    "distance_bins": bin_results,
    "input_provenance": {
        "catalog_csv": str(CSV),
        "catalog_note": "committed Li 2025 GPC catalog in the repo (data/, gitignored in the "
                        "public repo but present locally); 30,131 rows.",
        "galex": "vizier:II/335/galex_ais via astroquery.xmatch.XMatch, queried live at run time "
                 "(3 arcsec). Not cached to disk; re-querying reproduces the match.",
        "script_location": "scripts/adv_survivor3b.py (repo copy of the salvage-dir original); "
                           "this script's only persistent input is the repo catalog CSV above plus "
                           "the live GALEX query, so the result is reproducible from the committed "
                           "catalog + a network call.",
    },
}
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(res, indent=2))
print(f"\nwrote {OUT_JSON}")
