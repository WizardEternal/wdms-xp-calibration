"""G1 with REAL spectra (data-source-agnostic).

Replaces the blackbody proxy (p0_gonogo.py) with spectral TEMPLATE LIBRARIES from disk
(real Gaia XP single MS + single WD, or theoretical Koester+BT-Settl grids). Same go/no-go,
plus:
  * LEAVE-ONE-OUT (loo): each truth spectrum is fit with its generating template(s) EXCLUDED,
    so clean fits are not trivially perfect; surfaces library-sparsity false positives.
    Headline metrics reported BOTH non-LOO (optimistic) and LOO (defensible).
  * DISTRIBUTIONAL GoF: AUC of binary-fit redchi2 separating genuine vs spurious (~0.5 =
    per-object GoF cannot distinguish them), replacing the |Delta redchi2|<0.3 heuristic.
  * TEMPLATE DIAGNOSTIC: residual blue-excess (RINJ projection) already present in the MS
    templates - quantifies whether the audited systematic is pre-baked into real Gaia XP
    templates BEFORE we inject more.
  * load-drop diagnostics; falsy-zero guards.

The misspecification under test is the injected Gaia BP-band blue excess (Riello/Huang),
absent from the library by construction. Usage:
  python scripts/g1_realistic.py --data data/real_spectra [--severity 0.05 --snr 30]
Run-guarded: prints a clear NO-DATA message if templates are missing.
"""
import argparse
import csv
import glob
import json
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "p0"
WAVE = np.arange(392.0, 992.0 + 1e-6, 10.0)
RNG = np.random.default_rng(0)


def _read_csv_spectrum(path):
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
    if len(w) and np.nanmedian(w) > 3000:   # Angstrom -> nm
        w = w / 10.0
    return w, f


def load_library(data_dir, prefix, verbose=True):
    loaded, dropped = [], []
    for p in sorted(glob.glob(str(Path(data_dir) / f"{prefix}*.csv"))):
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
    if verbose and dropped:
        print(f"  [{prefix}] dropped {len(dropped)}: " + "; ".join(f"{n}({r})" for n, r in dropped[:6]))
    return (np.array(loaded) if loaded else np.empty((0, len(WAVE)))), dropped


def riello_inject():
    t = np.where(WAVE <= 680.0,
                 np.cos(0.5 * np.pi * np.clip((WAVE - 392.0) / (680.0 - 392.0), 0, 1)) ** 2, 0.0)
    return t / t.mean()


def riello_gate():
    t = np.where(WAVE <= 680.0, np.clip((680.0 - WAVE) / (680.0 - 392.0), 0, 1), 0.0)
    return t / t.mean()


RINJ, RGATE = riello_inject(), riello_gate()


def _chi2(model, x, sig):
    r = (x - model) / sig
    return float(r @ r)


def fit_single(x, sig, LIB, exclude=()):
    best = np.inf
    for k in range(LIB.shape[0]):
        if k in exclude:
            continue
        a, _ = nnls((LIB[k] / sig)[:, None], x / sig)
        best = min(best, _chi2(a[0] * LIB[k], x, sig))
    return best


def fit_binary(x, sig, MS, WD, ex_ms=-1, ex_wd=-1):
    best = np.inf; xs = x / sig
    for i in range(MS.shape[0]):
        if i == ex_ms:
            continue
        ai = MS[i] / sig
        for j in range(WD.shape[0]):
            if j == ex_wd:
                continue
            c, _ = nnls(np.column_stack([ai, WD[j] / sig]), xs)
            best = min(best, _chi2(c[0] * MS[i] + c[1] * WD[j], x, sig))
    return best


def fit_single_plus(x, sig, MS, template, ex_ms=-1):
    best = np.inf; xs = x / sig; rs = template / sig
    for i in range(MS.shape[0]):
        if i == ex_ms:
            continue
        c, _ = nnls(np.column_stack([MS[i] / sig, rs]), xs)
        best = min(best, _chi2(c[0] * MS[i] + c[1] * template, x, sig))
    return best


def bic(chi2, k, n=len(WAVE)):
    return chi2 + k * np.log(n)


def gen(LIB, n, snr, second=None, fracs=None):
    idx = RNG.integers(0, LIB.shape[0], n)
    x0 = LIB[idx].copy(); jdx = -np.ones(n, int)
    if second is not None:
        jdx = RNG.integers(0, second.shape[0], n)
        fr = RNG.uniform(*fracs, n)
        for k in range(n):
            x0[k] = LIB[idx[k]] + fr[k] * second[jdx[k]] * (LIB[idx[k]].sum() / second[jdx[k]].sum())
    sig = x0.mean(1, keepdims=True) / snr
    x = x0 + RNG.standard_normal(x0.shape) * sig
    return x, np.broadcast_to(sig, x0.shape).copy(), idx, jdx


def template_blue_residual(MS):
    """Median |projection onto RINJ| of each MS template's LOO single-fit residual,
    relative to the template mean. Large => BP-excess-shaped structure already in templates."""
    ALL = MS
    proj = []
    for i in range(MS.shape[0]):
        sig = np.ones(len(WAVE))
        # best other-template fit (amplitude) to template i
        best = (np.inf, None)
        for k in range(ALL.shape[0]):
            if k == i:
                continue
            a, _ = nnls((ALL[k])[:, None], MS[i])
            r = MS[i] - a[0] * ALL[k]
            c = float(r @ r)
            if c < best[0]:
                best = (c, r)
        if best[1] is not None:
            p = float(best[1] @ RINJ) / float(RINJ @ RINJ)
            proj.append(abs(p) / MS[i].mean())
    return round(float(np.median(proj)), 4) if proj else None


def run(data_dir, severity=0.05, snr=30.0, n=300, seed=0, loo=True, maxlib=50):
    global RNG
    RNG = np.random.default_rng(seed)
    print(f"loading templates from {data_dir} ...")
    MS, dms = load_library(data_dir, "ms")
    WD, dwd = load_library(data_dir, "wd")
    if MS.shape[0] < 3 or WD.shape[0] < 3:
        print(json.dumps({"status": "NO DATA", "data_dir": str(data_dir),
                          "n_ms": int(MS.shape[0]), "n_wd": int(WD.shape[0]),
                          "note": "Need >=3 ms*.csv and wd*.csv templates."}))
        return None
    n_ms_full, n_wd_full = MS.shape[0], WD.shape[0]
    # cap the fit library (binary fit is O(N_MS*N_WD)); subsample representatively
    if MS.shape[0] > maxlib:
        MS = MS[RNG.choice(MS.shape[0], maxlib, replace=False)]
    if WD.shape[0] > maxlib:
        WD = WD[RNG.choice(WD.shape[0], maxlib, replace=False)]
    ALL = np.vstack([MS, WD])

    xs_s, sig_s, i_s, _ = gen(MS, n, snr)
    xs_b, sig_b, i_b, j_b = gen(MS, n, snr, second=WD, fracs=(0.05, 0.30))
    xs_t = xs_s + severity * xs_s.mean(1, keepdims=True) * RINJ[None, :]

    def metrics(use_loo):
        # exclusions per spectrum (only when use_loo)
        def es(idx):  # exclude generating MS template (same index in ALL, MS first)
            return ({int(idx[k])} for k in range(n)) if use_loo else ((set() for _ in range(n)))
        exs_s = list(es(i_s)); exs_t = list(es(i_s))
        cs_s = np.array([fit_single(xs_s[k], sig_s[k], ALL, exs_s[k]) for k in range(n)])
        cb_s = np.array([fit_binary(xs_s[k], sig_s[k], MS, WD, i_s[k] if use_loo else -1) for k in range(n)])
        cs_b = np.array([fit_single(xs_b[k], sig_b[k], ALL,
                                    {int(i_b[k]), int(MS.shape[0] + j_b[k])} if use_loo else set()) for k in range(n)])
        cb_b = np.array([fit_binary(xs_b[k], sig_b[k], MS, WD,
                                    i_b[k] if use_loo else -1, j_b[k] if use_loo else -1) for k in range(n)])
        cs_t = np.array([fit_single(xs_t[k], sig_s[k], ALL, exs_t[k]) for k in range(n)])
        cb_t = np.array([fit_binary(xs_t[k], sig_s[k], MS, WD, i_s[k] if use_loo else -1) for k in range(n)])
        cr_b = np.array([fit_single_plus(xs_b[k], sig_b[k], MS, RGATE, i_b[k] if use_loo else -1) for k in range(n)])
        cr_t = np.array([fit_single_plus(xs_t[k], sig_s[k], MS, RGATE, i_s[k] if use_loo else -1) for k in range(n)])
        cro_b = np.array([fit_single_plus(xs_b[k], sig_b[k], MS, RINJ, i_b[k] if use_loo else -1) for k in range(n)])
        cro_t = np.array([fit_single_plus(xs_t[k], sig_s[k], MS, RINJ, i_s[k] if use_loo else -1) for k in range(n)])

        d_s = (cs_s - cb_s) / cs_s; d_b = (cs_b - cb_b) / cs_b; d_t = (cs_t - cb_t) / cs_t
        thr = np.quantile(d_s, 0.95)
        sel = d_t > thr
        y2 = np.r_[np.zeros(n), np.ones(sel.sum())] if sel.any() else None
        def auc(a, b):
            return round(float(roc_auc_score(y2, np.r_[a, b])), 3) if sel.any() else None
        gof_auc = auc(cb_b, cb_t[sel])  # binary-redchi2 genuine-vs-spurious (~0.5 = indistinguishable)
        gstat = lambda cb, cr: bic(cb, 4) - bic(cr, 3)
        return {
            "sanity_auc_single_vs_binary": round(float(roc_auc_score(np.r_[np.zeros(n), np.ones(n)], np.r_[d_s, d_b])), 3),
            "spurious_rate": round(float(sel.mean()), 3),
            "baseline_fpr_clean": round(float((d_s > thr).mean()), 3),
            "redchi2_binary_genuine": round(float(np.median(cb_b) / (len(WAVE) - 4)), 3),
            "redchi2_binary_spurious": round(float(np.median(cb_t[sel]) / (len(WAVE) - 4)), 3) if sel.any() else None,
            "gof_AUC_genuine_vs_spurious": gof_auc,   # ~0.5 => per-object GoF can't tell them apart
            "auc_gate_honest": auc(gstat(cb_b, cr_b), gstat(cb_t[sel], cr_t[sel])),
            "auc_gate_oracle": auc(gstat(cb_b, cro_b), gstat(cb_t[sel], cro_t[sel])),
            "auc_dchi2_perobj": auc(d_b, d_t[sel]),
        }

    res = {
        "status": "OK", "data_dir": str(data_dir),
        "n_ms_templates_used": int(MS.shape[0]), "n_wd_templates_used": int(WD.shape[0]),
        "n_ms_available": n_ms_full, "n_wd_available": n_wd_full, "maxlib": maxlib,
        "dropped": {"ms": len(dms), "wd": len(dwd)},
        "config": {"severity": severity, "snr": snr, "n": n},
        "template_blue_residual_median": template_blue_residual(MS),
        "template_blue_residual_note": "median |RINJ-projection| of MS-template LOO residuals / mean flux; "
                                       "large => BP-excess pre-baked in templates. "
                                       "Compare to injected severity.",
        "NONLOO": metrics(False),
        "LOO": metrics(True),
        "verdict_note": "Trust the LOO block. GO if (LOO) spurious_rate >> baseline_fpr, gof_AUC ~0.5 "
                        "(per-object GoF blind), auc_dchi2_perobj <=0.5 (anti-informative), and "
                        "auc_gate_honest well above 0.5. If LOO baseline_fpr or spurious on clean is high, "
                        "the library is too sparse (confound) - report, do not claim GO.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "g1_realistic.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real_spectra")
    ap.add_argument("--severity", type=float, default=0.05)
    ap.add_argument("--snr", type=float, default=30.0)
    args = ap.parse_args()
    run(ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data),
        severity=args.severity, snr=args.snr)
