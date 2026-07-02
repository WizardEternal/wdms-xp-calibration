"""Severity-to-local-excess audit for the note's ~27x number.

The note reports that an s=0.02 ADDITIVE injection (a fraction s of a spectrum's
TOTAL flux, deposited blueward of 680 nm via the unit-mean half-cosine RINJ) works
out to a much larger LOCAL blue excess than Huang 2024's 2% post-correction residual.
This script recomputes that local excess directly from the real single-MS Gaia XP
templates and writes the labelled statistics to a committed JSON, so the ~27x
multiplier in the note (05-note/note.tex and 05-note/appendix.tex, App. A.1) has
backing that is not just stdout.

Definition (matches appendix.tex, App. A.1):
  * 61-pixel grid lambda = 392, 402, ..., 992 nm (10 nm spacing).
  * unnormalized shape  rtilde(lambda) = cos^2( (pi/2) * x ) for lambda <= 680 nm,
    0 past 680 nm, with x = clip[(lambda-392)/(680-392), 0, 1].
  * RINJ = rtilde / mean(rtilde)  (unit mean over the 61-pixel grid).
  * additive injection  x_t = x_s + s * mean(x_s) * RINJ.
  * local fractional excess at each pixel = (x_t - x_s)/x_s = s * mean(x_s) * RINJ / x_s.

We evaluate the local excess for every real single-MS template interpolated onto the
grid, and report three LABELLED summaries of the blue local excess, plus the implied
multiplier over a 0.02 local residual for each:
  1. per_template_band_median_392_420_then_median_across_templates:
     for each template take the median local excess over the 392-420 nm pixels,
     then the median of those per-template values (the note's headline construction).
  2. pooled_pixel_median_392_420:
     pool every (template, pixel) local-excess value over the 392-420 nm pixels and
     take the single median.
  3. median_edge_392:
     median across templates of the local excess at the single 392 nm edge pixel.

Usage:  python scripts/severity_to_local.py [--severity 0.02]
No git actions; writes outputs/audit/severity_to_local.json.
"""
import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "audit"
WAVE = np.arange(392.0, 992.0 + 1e-6, 10.0)  # 61 pixels, 392..992 nm
assert len(WAVE) == 61, f"grid is {len(WAVE)} px, expected 61"

# 392-420 nm band on the 10-nm grid = the three pixels 392, 402, 412
# (422 > 420, so it is excluded; matches the note's "392-420 nm" blue band).
BAND = (WAVE >= 392.0) & (WAVE <= 420.0)


def _read_csv_spectrum(path):
    """Read one wavelength_nm,flux[,flux_error] CSV; Angstrom->nm autoconvert."""
    w, f = [], []
    with open(path, newline="") as fh:
        r = csv.DictReader(fh)
        if not r.fieldnames:
            return np.array([]), np.array([])
        wcol = next((c for c in r.fieldnames if "wave" in c.lower()), r.fieldnames[0])
        fcol = next((c for c in r.fieldnames if c.lower() in ("flux", "flux_flambda", "flam")),
                    r.fieldnames[1] if len(r.fieldnames) > 1 else r.fieldnames[0])
        for row in r:
            try:
                w.append(float(row[wcol])); f.append(float(row[fcol]))
            except (ValueError, TypeError):
                pass
    w, f = np.array(w), np.array(f)
    if len(w) and np.nanmedian(w) > 3000:  # Angstrom -> nm
        w = w / 10.0
    return w, f


def load_ms_templates(data_dir):
    """Load real single-MS templates, interpolate onto the 61-px grid, mean-normalize.

    Same drop rules as g1_realistic.load_library: need >=5 points, full grid
    coverage, positive mean. Returns (array [n, 61], list of dropped (name, reason))."""
    loaded, dropped = [], []
    for p in sorted(glob.glob(str(Path(data_dir) / "ms*.csv"))):
        if Path(p).name == "manifest.csv":
            continue
        w, f = _read_csv_spectrum(p)
        name = Path(p).name
        if len(w) < 5:
            dropped.append((name, "too few points")); continue
        fr = np.interp(WAVE, w, f, left=np.nan, right=np.nan)
        if not np.all(np.isfinite(fr)):
            dropped.append((name, f"grid coverage {w.min():.0f}-{w.max():.0f}nm")); continue
        if fr.mean() <= 0:
            dropped.append((name, "nonpositive mean")); continue
        loaded.append(fr / fr.mean())
    return (np.array(loaded) if loaded else np.empty((0, len(WAVE)))), dropped


def riello_inject():
    """Unit-mean half-cosine RINJ (cos^2 taper, zero past 680 nm)."""
    t = np.where(WAVE <= 680.0,
                 np.cos(0.5 * np.pi * np.clip((WAVE - 392.0) / (680.0 - 392.0), 0, 1)) ** 2, 0.0)
    return t / t.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--severity", type=float, default=0.02)
    args = ap.parse_args()
    s = args.severity

    data_dir = ROOT / "data" / "real_spectra_large"
    used_dir = data_dir
    if not data_dir.exists() or not glob.glob(str(data_dir / "ms*.csv")):
        used_dir = ROOT / "data" / "real_spectra"
    MS, dropped = load_ms_templates(used_dir)
    if MS.shape[0] < 1:
        raise SystemExit(f"NO MS TEMPLATES under {used_dir}")

    RINJ = riello_inject()

    # local fractional excess per (template, pixel): s * mean(x_s) * RINJ / x_s.
    # MS is already mean-normalized so mean(x_s) == 1 per template, but keep it
    # explicit so the formula is exactly Eq. (severity) regardless of normalization.
    mean_flux = MS.mean(axis=1, keepdims=True)                      # [n,1]
    local_excess = s * mean_flux * RINJ[None, :] / MS               # [n,61]

    band_local = local_excess[:, BAND]                             # [n, 3] over 392-420
    edge_local = local_excess[:, 0]                               # 392 nm column

    # 1) per-template band median, then median across templates
    per_tmpl_band_median = np.median(band_local, axis=1)          # [n]
    stat_per_template = float(np.median(per_tmpl_band_median))

    # 2) pooled pixel median over the band
    stat_pooled = float(np.median(band_local))

    # 3) median across templates of the 392 nm edge pixel
    stat_edge = float(np.median(edge_local))

    ref_local = 0.02  # Huang 2024 ~2% post-correction local residual shortward of 400 nm

    res = {
        "description": "Local blue excess implied by an additive severity-s RINJ injection "
                       "on real single-MS Gaia XP templates, backing the note's ~27x number "
                       "(05-note/appendix.tex App. A.1).",
        "severity_s_additive_total_flux_fraction": s,
        "reference_local_residual": ref_local,
        "reference_note": "Huang 2024 (ApJS 271, 13) ~2% post-correction local residual "
                          "shortward of 400 nm; the injection is compared against this.",
        "data_dir_used": str(used_dir),
        "fell_back_to_small_dir": bool(used_dir.name == "real_spectra" and used_dir != data_dir),
        "n_templates": int(MS.shape[0]),
        "n_dropped": len(dropped),
        "grid_nm": [float(WAVE[0]), float(WAVE[-1]), 10.0, int(len(WAVE))],
        "band_392_420_nm_pixels": [float(x) for x in WAVE[BAND]],
        "RINJ_peak_over_mean_at_392": float(RINJ[0]),
        "statistics": {
            "per_template_band_median_392_420_then_median_across_templates": {
                "value": stat_per_template,
                "implied_multiplier_over_0p02_local": float(stat_per_template / ref_local),
                "as_percent": float(stat_per_template * 100.0),
            },
            "pooled_pixel_median_392_420": {
                "value": stat_pooled,
                "implied_multiplier_over_0p02_local": float(stat_pooled / ref_local),
                "as_percent": float(stat_pooled * 100.0),
            },
            "median_edge_392": {
                "value": stat_edge,
                "implied_multiplier_over_0p02_local": float(stat_edge / ref_local),
                "as_percent": float(stat_edge * 100.0),
            },
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "severity_to_local.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
