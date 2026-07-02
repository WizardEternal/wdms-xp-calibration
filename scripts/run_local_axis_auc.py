"""Committed backing for the note's Section-3 local-axis AUC numbers.

The note (Sec 3) states, for the honest MULTIPLICATIVE-LOCAL blue excess x_t = x_s*(1+a*taper):
  * the per-object Delta chi2 (what the classifier scores on) is anti-informative for
    flagging the spurious singles  -> Dchi2 AUC ~ 0.0;
  * a plain binary-fit goodness-of-fit regains power as the excess grows
    -> genuine-vs-spurious reduced-chi2 AUC ~ 0.74 at 20% local, ~0.98 at 50%;
  * a model-comparison gate (full binary vs single-star + one free nuisance flux term,
    by BIC) keeps power -> gate AUC ~ 0.9 at 20% local.
Those three axes previously lived only in scratchpad logs (recheck3_auc / gof_local /
gate_local, seeds 31/32/33). This script reproduces them from the PRODUCTION fit
functions (g1_realistic.fit_single / fit_binary / fit_single_plus / gen / load_library)
so every Sec-3 AUC is reproducible from a committed script + seed.

Injection is multiplicative-local on the 0..1 blue cos^2 taper (the honest Huang-residual
form, local excess <= a), NOT the note's original additive-of-total profile. Selection is
the same 1-D Delta chi2 > 95th-percentile-of-clean cut the production pipeline uses; the
threshold is set on an independent NOISE-ONLY clean control (no injection) so the
severity-zero step is not tautological.

Axes (all under leave-one-out, the generating template excluded from every fit):
  Dchi2 axis  = AUC( d_b , d_t[sel] )                 d = (chi2_single - chi2_binary)/chi2_single
  gof axis    = AUC( cb_b , cb_t[sel] )               cb = binary-fit chi2 (reduced-chi2 up to /dof)
  gate axis   = AUC( G_b , G_t[sel] )                 G  = BIC(binary,k=4) - BIC(single+nuisance,k=3)
                                                        (positive sign: genuine binaries -> low G,
                                                         spurious -> high G; the VERIFIED-correct
                                                         convention, NOT the 1-AUC sign flip.)
label y = 0 for genuine binaries, 1 for the injection-selected spurious singles.

Amplitudes 0.02/0.05/0.10/0.20/0.50 local; seeds 31/32/33; production maxlib.
Crash-safe: writes a per-seed partial JSON the moment each seed finishes (skip-if-exists
to resume), and merges into outputs/p0/local_axis_auc.json with per-seed and mean AUCs.

Usage:
  python scripts/run_local_axis_auc.py                       # full run (seeds 31 32 33)
  python scripts/run_local_axis_auc.py --smoke               # 1 seed, a=0.20, reduced n
  python scripts/run_local_axis_auc.py --seeds 31 --amps 0.20 --n 40
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import g1_realistic as g1  # production fit_single / fit_binary / fit_single_plus / gen / load_library

WAVE = g1.WAVE
NPX = len(WAVE)
LNN = float(np.log(NPX))

# 0..1 blue cos^2 taper (the honest local shape; peaks at 1.0 at 392nm, 0 redward of 680nm).
# NOTE: this is NOT g1.RINJ (which is the unit-MEAN additive profile). Local excess <= a here.
TAPER01 = np.where(
    WAVE <= 680.0,
    np.cos(0.5 * np.pi * np.clip((WAVE - 392.0) / (680.0 - 392.0), 0, 1)) ** 2,
    0.0,
)
# gate nuisance = the production RGATE flux-correction template (unit-mean linear blue ramp).
RGATE = g1.RGATE

DEFAULT_AMPS = [0.02, 0.05, 0.10, 0.20, 0.50]
DEFAULT_SEEDS = [31, 32, 33]
OUTDIR = ROOT / "outputs" / "p0"
PARTDIR = OUTDIR / "_local_axis_parts"
MERGED = OUTDIR / "local_axis_auc.json"


def _bic(chi2, k):
    return chi2 + k * LNN


def _stats(x, sig, MS, WD, ALL, ex_ms):
    """Production-function fit triple for one spectrum under LOO (exclude template ex_ms).
    Returns (d, cb, G):
      d  = (chi2_single - chi2_binary)/chi2_single        (Delta chi2, binary support)
      cb = chi2_binary                                    (goodness-of-fit axis, raw chi2)
      G  = BIC(binary,4) - BIC(single+nuisance,3)         (model-comparison gate statistic)
    fit_single excludes ex_ms in ALL (MS block first, so the MS index maps directly);
    fit_binary / fit_single_plus exclude ex_ms on the MS axis."""
    cs = g1.fit_single(x, sig, ALL, {int(ex_ms)})
    cb = g1.fit_binary(x, sig, MS, WD, int(ex_ms))
    cr = g1.fit_single_plus(x, sig, MS, RGATE, int(ex_ms))
    d = (cs - cb) / cs
    G = _bic(cb, 4) - _bic(cr, 3)
    return d, cb, G


def run_cell(seed, amps, n=120, snr=30.0, maxlib=50):
    """One seed: build clean singles, genuine binaries, and locally-injected singles;
    fit all three with the production functions under LOO; return per-amplitude AUCs."""
    g1.RNG = np.random.default_rng(seed)
    MS0, _ = g1.load_library(ROOT / "data/real_spectra_large", "ms", verbose=False)
    WD0, _ = g1.load_library(ROOT / "data/real_spectra_large", "wd", verbose=False)
    MS = MS0[g1.RNG.choice(MS0.shape[0], maxlib, replace=False)] if MS0.shape[0] > maxlib else MS0
    WD = WD0[g1.RNG.choice(WD0.shape[0], maxlib, replace=False)] if WD0.shape[0] > maxlib else WD0
    ALL = np.vstack([MS, WD])

    # clean single-MS draws (i_s = generating MS index, used for LOO) and the noise realisation
    xs_s, sig_s, i_s, _ = g1.gen(MS, n, snr)
    # an independent NOISE-ONLY clean control for the selection threshold (no injection)
    xs_c, sig_c, i_c, _ = g1.gen(MS, n, snr)
    # genuine MS+WD binaries, production fracs 0.05-0.30 (LOO on the MS index i_b)
    xs_b, sig_b, i_b, j_b = g1.gen(MS, n, snr, second=WD, fracs=(0.05, 0.30))

    def triples(X, S, idx):
        d = np.empty(n); cb = np.empty(n); G = np.empty(n)
        for k in range(n):
            d[k], cb[k], G[k] = _stats(X[k], S[k], MS, WD, ALL, idx[k])
        return d, cb, G

    # threshold from the independent clean control
    d_c, _, _ = triples(xs_c, sig_c, i_c)
    thr = float(np.quantile(d_c, 0.95))
    # genuine binaries (the y=0 class), fixed across amplitudes
    d_b, cb_b, G_b = triples(xs_b, sig_b, i_b)

    out = {"seed": seed, "config": {"n": n, "snr": snr, "maxlib": maxlib,
                                    "n_ms": int(MS.shape[0]), "n_wd": int(WD.shape[0]),
                                    "thr_dchi2": round(thr, 6)},
           "by_amp": {}}
    for a in amps:
        xs_t = xs_s * (1.0 + a * TAPER01[None, :])
        d_t, cb_t, G_t = triples(xs_t, sig_s, i_s)  # reuse clean noise realisation + LOO indices
        sel = d_t > thr
        nsel = int(sel.sum())
        if nsel < 3:
            out["by_amp"][f"{a}"] = {"spurious": round(float(sel.mean()), 3), "n_sel": nsel,
                                     "dchi2_auc": None, "gof_redchi2_auc": None, "gate_auc": None}
            continue
        y = np.r_[np.zeros(n), np.ones(nsel)]
        dchi2_auc = float(roc_auc_score(y, np.r_[d_b, d_t[sel]]))
        gof_auc = float(roc_auc_score(y, np.r_[cb_b, cb_t[sel]]))
        gate_auc = float(roc_auc_score(y, np.r_[G_b, G_t[sel]]))
        out["by_amp"][f"{a}"] = {
            "spurious": round(float(sel.mean()), 3), "n_sel": nsel,
            "dchi2_auc": round(dchi2_auc, 3),
            "gof_redchi2_auc": round(gof_auc, 3),
            "gate_auc": round(gate_auc, 3),
        }
        print(f"  seed {seed} a={a}: spurious={out['by_amp'][f'{a}']['spurious']} "
              f"dchi2={dchi2_auc:.3f} gof={gof_auc:.3f} gate={gate_auc:.3f} (n_sel={nsel})",
              flush=True)
    return out


def merge(seeds, amps):
    parts = {}
    for s in seeds:
        p = PARTDIR / f"seed_{s}.json"
        if p.exists():
            parts[s] = json.loads(p.read_text())
    merged = {
        "description": "Section-3 local-axis AUCs from the production g1_realistic fit "
                       "functions; multiplicative-local injection x*(1+a*taper01), LOO, "
                       "clean-control threshold. Gate uses the positive BIC(bin,4)-BIC(sing+nuis,3) "
                       "convention (genuine->low, spurious->high).",
        "injection": "x_t = x_s*(1 + a*taper01), taper01 = cos^2 blue ramp in [0,1], 0 redward of 680nm",
        "seeds": seeds, "amps_local": amps,
        "per_seed": {str(s): parts[s] for s in parts},
        "mean": {},
    }
    for a in amps:
        key = f"{a}"
        vals = {m: [] for m in ("spurious", "dchi2_auc", "gof_redchi2_auc", "gate_auc")}
        for s in parts:
            cell = parts[s]["by_amp"].get(key)
            if cell is None:
                continue
            for m in vals:
                if cell.get(m) is not None:
                    vals[m].append(cell[m])
        merged["mean"][key] = {
            m: (round(float(np.mean(v)), 3) if v else None) for m, v in vals.items()
        }
        merged["mean"][key]["n_seeds"] = len(vals["gate_auc"])
    MERGED.write_text(json.dumps(merged, indent=2))
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--amps", type=float, nargs="+", default=DEFAULT_AMPS)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--snr", type=float, default=30.0)
    ap.add_argument("--maxlib", type=int, default=50)
    ap.add_argument("--smoke", action="store_true",
                    help="cheap single cell: seed 31, a=0.20, n=40")
    ap.add_argument("--outtag", default="", help="suffix for the smoke/merged file to avoid clobber")
    args = ap.parse_args()

    if args.smoke:
        args.seeds, args.amps, args.n = [31], [0.20], 40

    PARTDIR.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"local-axis AUC run: seeds={args.seeds} amps={args.amps} n={args.n} "
          f"snr={args.snr} maxlib={args.maxlib} smoke={args.smoke}", flush=True)

    for s in args.seeds:
        part = PARTDIR / f"seed_{s}{args.outtag}.json"
        if part.exists() and not args.smoke:
            print(f"seed {s}: partial exists, skipping (resume)", flush=True)
            continue
        print(f"seed {s}: fitting ...", flush=True)
        try:
            cell = run_cell(s, args.amps, n=args.n, snr=args.snr, maxlib=args.maxlib)
            part.write_text(json.dumps(cell, indent=2))
            print(f"seed {s}: wrote {part.name} ({time.time()-t0:.0f}s elapsed)", flush=True)
        except Exception as e:  # per-unit isolation so one seed failing doesn't lose the rest
            import traceback
            (PARTDIR / f"seed_{s}{args.outtag}.ERROR.txt").write_text(traceback.format_exc())
            print(f"seed {s}: FAILED -> {e!r} (logged, continuing)", flush=True)

    if args.smoke:
        # print the single smoke cell without polluting the real merged artifact
        part = PARTDIR / f"seed_{args.seeds[0]}.json"
        if part.exists():
            print("SMOKE RESULT:", json.dumps(json.loads(part.read_text())["by_amp"], indent=2), flush=True)
    else:
        merged = merge(args.seeds, args.amps)
        print("MERGED MEAN:", json.dumps(merged["mean"], indent=2), flush=True)
        print(f"wrote {MERGED}", flush=True)
    print(f"DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
