"""P0 / G1 go-no-go: does the Riello BP-excess systematic silently manufacture
spurious WD-MS candidates, and does a systematic-aware GoF gate catch what the
per-object model-comparison (Delta-chi2) cannot?

Optical analog of arXiv:2606.17098's gain-shift result. PROXY forward model
(blackbody WD + blackbody MS, projected to the 61-px XP grid) - deliberately simple
for the go/no-go; production swaps in Koester DA + BT-Settl. Uses numpy/scipy only
(no sbi/gaiaxpy/ultranest needed).

Mechanism under test (Li 2025 Sec 2.2 + Riello+2021): the Gaia G_BP (blue) flux is
*overestimated* for faint red sources. That spurious BLUE excess mimics a hot WD
companion, so a single red MS star can be mis-selected as a WD-MS binary. We test:
  (A) SPURIOUS RATE: do BP-excess-injected single stars cross the binary threshold?
  (B) GoF BLINDNESS: is the binary-fit quality of spurious binaries as good as genuine
      ones (so a per-object goodness-of-fit gate cannot tell them apart)?
  (C) THE GATE EARNS ITS COST: a systematic-aware model comparison (single+Riello vs
      binary, by BIC) flags the spurious ones where Delta-chi2 is blind.

The discriminating physics: a real WD adds a SMOOTH hot continuum across all wavelengths;
the BP systematic is a band-LOCALIZED blue artifact. Delta-chi2(single vs binary) sees
only "blue excess => binary" (blind to which); the gate models the artifact explicitly.
"""
import json
import numpy as np
from pathlib import Path
from scipy.optimize import nnls
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(0)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "p0"

WAVE = np.arange(392.0, 992.0 + 1e-6, 10.0)  # 61 px, nm (Li grid)
H, C, KB = 6.62607015e-34, 2.99792458e8, 1.380649e-23


def planck(wave_nm, T):
    """Planck B_lambda (arbitrary units), normalized to unit mean over the grid."""
    lam = wave_nm * 1e-9
    b = (2 * H * C**2 / lam**5) / (np.exp(H * C / (lam * KB * T)) - 1.0)
    return b / b.mean()


def riello_template():
    """Realistic Gaia BP-band blue excess (Riello+2021 / Li 2025 Sec 2.2): a SMOOTH
    excess over the BP wavelength range (~<680 nm), zero in RP. The physical signature
    that distinguishes it from a real WD: a WD adds a smooth hot continuum across the
    WHOLE optical (BP AND RP), whereas this systematic adds flux ONLY in the blue.
    Half-cosine taper: 1 at 392 nm -> 0 at 680 nm."""
    t = np.where(WAVE <= 680.0,
                 np.cos(0.5 * np.pi * np.clip((WAVE - 392.0) / (680.0 - 392.0), 0, 1)) ** 2,
                 0.0)
    return t / t.mean()


RIELLO = riello_template()


def riello_gate_template():
    """A DIFFERENT but plausible BP-band blue-excess shape for the GATE to use, so the
    gate does NOT know the exact injected shape (avoid the oracle confound). Linear
    blue ramp 1@392 -> 0@680 nm (vs the injected half-cosine)."""
    t = np.where(WAVE <= 680.0, np.clip((680.0 - WAVE) / (680.0 - 392.0), 0, 1), 0.0)
    return t / t.mean()


RIELLO_GATE = riello_gate_template()
# precompute blackbody basis grids
T_MS = np.linspace(3300, 6500, 28)
T_WD = np.linspace(7000, 40000, 26)
BB_MS = np.array([planck(WAVE, T) for T in T_MS])
BB_WD = np.array([planck(WAVE, T) for T in T_WD])


def _chi2(model, x, sig):
    r = (x - model) / sig
    return float(r @ r)


def fit_single(x, sig):
    """best single BB over the MS+WD temperature grid; linear amp via nnls."""
    best = (np.inf, None)
    for B in (BB_MS, BB_WD):
        for k in range(B.shape[0]):
            A = (B[k] / sig)[:, None]
            a, _ = nnls(A, x / sig)
            c = _chi2(a[0] * B[k], x, sig)
            if c < best[0]:
                best = (c, (B[k], a[0]))
    return best[0]  # chi2_single, 2 params


def fit_binary(x, sig):
    """best MS-cool + WD-hot pair; amplitudes via nnls (>=0)."""
    best = np.inf
    xs = x / sig
    for i in range(len(T_MS)):
        ai = BB_MS[i] / sig
        for j in range(len(T_WD)):
            A = np.column_stack([ai, BB_WD[j] / sig])
            coef, _ = nnls(A, xs)
            c = _chi2(coef[0] * BB_MS[i] + coef[1] * BB_WD[j], x, sig)
            if c < best:
                best = c
    return best  # chi2_binary, 4 params


def fit_single_plus_riello(x, sig, template=RIELLO):
    """single MS BB + a blue-artifact template; amps via nnls. `template` defaults to
    RIELLO_GATE in run() (mismatched, the honest gate); pass RIELLO for the oracle bound."""
    best = np.inf
    xs = x / sig
    rs = template / sig
    for i in range(len(T_MS)):
        A = np.column_stack([BB_MS[i] / sig, rs])
        coef, _ = nnls(A, xs)
        c = _chi2(coef[0] * BB_MS[i] + coef[1] * template, x, sig)
        if c < best:
            best = c
    return best  # 3 params


def bic(chi2, k, n=len(WAVE)):
    return chi2 + k * np.log(n)


def make_single_ms(n, snr):
    T = RNG.uniform(3500, 6000, n)
    x0 = np.array([planck(WAVE, t) for t in T])
    sig = x0.mean(1, keepdims=True) / snr            # (n,1) per-spectrum noise level
    x = x0 + RNG.standard_normal(x0.shape) * sig     # independent per-PIXEL noise
    return x, np.broadcast_to(sig, x0.shape).copy(), T


def make_binary(n, snr):
    """MS-dominated WD-MS: a real WD adds a SMOOTH hot continuum (small blue fraction)."""
    Tms = RNG.uniform(3500, 6000, n)
    Twd = RNG.uniform(9000, 35000, n)
    frac = RNG.uniform(0.05, 0.30, n)  # WD blue-flux fraction (subdominant)
    x0 = np.zeros((n, len(WAVE)))
    for k in range(n):
        ms = planck(WAVE, Tms[k]); wd = planck(WAVE, Twd[k])
        x0[k] = ms + frac[k] * wd * (ms.sum() / wd.sum())
    sig = x0.mean(1, keepdims=True) / snr
    x = x0 + RNG.standard_normal(x0.shape) * sig     # independent per-PIXEL noise
    return x, np.broadcast_to(sig, x0.shape).copy(), Tms, Twd, frac


def inject_riello(x_single, sig, severity):
    """Add the BP-localized blue artifact to single-star spectra (the systematic)."""
    amp = severity * x_single.mean(1, keepdims=True)
    return x_single + amp * RIELLO[None, :]


def dchi2_renorm(cs, cb):
    return (cs - cb) / cs


def run(n=500, snr=30.0, severity=0.15, seed=0):
    global RNG
    RNG = np.random.default_rng(seed)   # reseed so each severity sees the same clean pops
    OUT.mkdir(parents=True, exist_ok=True)
    # --- generate clean populations ---
    xs_s, sig_s, _ = make_single_ms(n, snr)
    xs_b, sig_b, *_ = make_binary(n, snr)

    def scores(xs, sig):
        cs = np.array([fit_single(xs[k], sig[k]) for k in range(len(xs))])
        cb = np.array([fit_binary(xs[k], sig[k]) for k in range(len(xs))])
        cr = np.array([fit_single_plus_riello(xs[k], sig[k], RIELLO_GATE) for k in range(len(xs))])  # honest (mismatched)
        cro = np.array([fit_single_plus_riello(xs[k], sig[k], RIELLO) for k in range(len(xs))])      # oracle (matched)
        return cs, cb, cr, cro

    cs_s, cb_s, cr_s, cro_s = scores(xs_s, sig_s)        # clean singles
    cs_b, cb_b, cr_b, cro_b = scores(xs_b, sig_b)        # genuine binaries

    d_s = dchi2_renorm(cs_s, cb_s)
    d_b = dchi2_renorm(cs_b, cb_b)

    # baseline: can Delta-chi2 separate clean single vs genuine binary? (sanity)
    auc_base = roc_auc_score(np.r_[np.zeros(n), np.ones(n)], np.r_[d_s, d_b])
    thr = np.quantile(d_s, 0.95)   # binary-decision threshold (5% FPR on clean singles)

    # --- inject the Riello systematic into single stars ---
    xs_t = inject_riello(xs_s, sig_s, severity)
    cs_t, cb_t, cr_t, cro_t = scores(xs_t, sig_s)
    d_t = dchi2_renorm(cs_t, cb_t)

    # (A) spurious rate
    spurious = float((d_t > thr).mean())
    baseline_fpr = float((d_s > thr).mean())
    sel = d_t > thr

    # (B) per-object GoF: binary-fit reduced chi2, genuine vs spurious
    redchi2_genuine = float(np.median(cb_b) / (len(WAVE) - 4))
    redchi2_spurious = float(np.median(cb_t[sel]) / (len(WAVE) - 4)) if sel.any() else None

    # (C) gate AUC: BIC(binary,4) - BIC(single+template,3) separates genuine vs spurious.
    # HONEST gate uses the mismatched template (cr); ORACLE uses the matched one (cro) and
    # is only an UPPER BOUND (matched template is not earned).
    def gate_auc_fa(crb, crt):
        if not sel.any():
            return None, None
        stat = lambda cb, cr: bic(cb, 4) - bic(cr, 3)   # high => looks systematic
        auc = roc_auc_score(np.r_[np.zeros(n), np.ones(sel.sum())],
                            np.r_[stat(cb_b, crb), stat(cb_t[sel], crt[sel])])
        fa = float((bic(crb, 3) < bic(cb_b, 4)).mean())  # gate flags a genuine binary
        return auc, fa
    auc_gate_honest, fa_honest = gate_auc_fa(cr_b, cr_t)
    auc_gate_oracle, _ = gate_auc_fa(cro_b, cro_t)
    auc_dchi2 = (roc_auc_score(np.r_[np.zeros(n), np.ones(sel.sum())], np.r_[d_b, d_t[sel]])
                 if sel.any() else None)

    res = {
        "config": {"n_per_class": n, "snr": snr, "severity": severity,
                   "binary_thr_dchi2renorm": round(float(thr), 4)},
        "sanity_auc_single_vs_binary": round(auc_base, 3),
        "A_spurious_rate_injected_singles": round(spurious, 3),
        "A_baseline_fpr_clean_singles": round(baseline_fpr, 3),
        "B_perobject_gof": {
            "redchi2_binary_genuine": round(redchi2_genuine, 3),
            "redchi2_binary_spurious": round(redchi2_spurious, 3) if redchi2_spurious else None,
            "indistinguishable_heuristic": (redchi2_spurious is not None and abs(redchi2_spurious - redchi2_genuine) < 0.3),
        },
        "C_gate": {
            "auc_gate_HONEST_mismatched_template": round(auc_gate_honest, 3) if auc_gate_honest else None,
            "auc_gate_ORACLE_matched_template_upperbound": round(auc_gate_oracle, 3) if auc_gate_oracle else None,
            "false_alarm_on_genuine_honest": round(fa_honest, 3) if fa_honest is not None else None,
            "auc_dchi2_perobj_genuine_vs_spurious": round(auc_dchi2, 3) if auc_dchi2 else None,
            "dchi2_note": "<0.5 => per-object Delta-chi2 is ANTI-informative (ranks spurious as MORE binary-like than genuine); not merely uninformative.",
        },
        "verdict_hint": (
            "GO if: spurious_rate >> baseline_fpr; per-object Delta-chi2 AUC <= 0.5 "
            "(blind/anti-informative); and the HONEST (mismatched-template) gate AUC is "
            "well above 0.5 (oracle is an upper bound only)."
        ),
    }
    (OUT / "gonogo.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


def sweep():
    OUT.mkdir(parents=True, exist_ok=True)
    sevs = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    rows = [run(n=400, snr=30.0, severity=s, seed=0) for s in sevs]
    table = []
    for s, r in zip(sevs, rows):
        table.append({
            "severity": s,
            "spurious_rate": r["A_spurious_rate_injected_singles"],
            "baseline_fpr": r["A_baseline_fpr_clean_singles"],
            "redchi2_binary_genuine": r["B_perobject_gof"]["redchi2_binary_genuine"],
            "redchi2_binary_spurious": r["B_perobject_gof"]["redchi2_binary_spurious"],
            "gof_indistinguishable": r["B_perobject_gof"]["indistinguishable_heuristic"],
            "auc_gate_honest": r["C_gate"]["auc_gate_HONEST_mismatched_template"],
            "auc_gate_oracle": r["C_gate"]["auc_gate_ORACLE_matched_template_upperbound"],
            "auc_dchi2_perobj": r["C_gate"]["auc_dchi2_perobj_genuine_vs_spurious"],
            "gate_false_alarm_genuine": r["C_gate"]["false_alarm_on_genuine_honest"],
        })
    (OUT / "sweep.json").write_text(json.dumps(table, indent=2))
    print(json.dumps(table, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sv = [t["severity"] for t in table]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(sv, [t["spurious_rate"] for t in table], "o-", label="spurious-binary rate")
        ax[0].plot(sv, [t["baseline_fpr"] for t in table], "k--", label="baseline FPR")
        ax[0].set(xlabel="BP-excess severity", ylabel="rate",
                  title="(A) systematic manufactures candidates")
        ax[0].legend()
        ax[1].plot(sv, [t["auc_gate_honest"] for t in table], "s-", label="gate AUC (honest, mismatched template)")
        ax[1].plot(sv, [t["auc_gate_oracle"] for t in table], "s:", color="grey", label="gate AUC (oracle upper bound)")
        ax[1].plot(sv, [t["auc_dchi2_perobj"] for t in table], "^-", label="Delta-chi2 per-object AUC")
        ax[1].axhline(0.5, color="grey", ls=":")
        ax[1].set(xlabel="BP-excess severity", ylabel="AUC (genuine vs spurious)",
                  title="(C) gate catches what Delta-chi2 misses", ylim=(0.0, 1.02))
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(OUT / "gonogo_sweep.png", dpi=150)
        print("figure ->", OUT / "gonogo_sweep.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    sweep()
