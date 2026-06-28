"""NPE + SBC calibration diagnostic on a REAL-SPECTRA forward model.

Sequel to npe_sbc.py (blackbody proxy). A referee's first ask for the calibration-audit
paper is: show the SBC coverage-collapse on a forward model grounded in REAL Gaia XP
templates, not a blackbody. This does exactly that.

FORWARD MODEL (continuous, real-template-grounded)
--------------------------------------------------
Load the 120 MS + 120 WD real Gaia XP templates (g1_realistic.load_library; 61-px WAVE
grid arange(392,992,10) nm, each template mean-normalized to 1). Build a low-dim continuous
parametrization by PCA of each library:
    MS spectrum  = mu_ms + s_ms . PC_ms      (2 PCs, >99% variance)
    WD spectrum  = mu_wd + s_wd . PC_wd       (3 PCs, ~96% variance)
This interpolates smoothly between real templates (NPE needs a smooth theta->x map) while
staying inside the real-spectrum manifold (reconstructions verified strictly positive over
the whole prior box). The prior box on each score is the empirical [min,max] spanned by the
real templates, so we never extrapolate off the real-spectrum support.

    theta = (s_ms[0], s_ms[1], s_wd[0], s_wd[1], s_wd[2], log10_frac)              # 6 params
    x     = MS(theta) + 10^log10_frac * WD(theta) * (MS.sum()/WD.sum()) + noise@SNR

The WD flux-matching factor (MS.sum()/WD.sum()) and the log-frac range [log10(0.05),
log10(0.30)] mirror g1_realistic.gen(..., fracs=(0.05,0.30)). Noise: Gaussian, sigma =
x0.mean/SNR, SNR=30 - same as gen.

SBC (Talts+2018), exactly as in npe_sbc.py:
  * CLEAN held-out sims  -> is the amortized posterior calibrated?
  * BP-CONTAMINATED sims (inject severity*mean*RINJ blue excess at inference time, the
    Riello/Huang Gaia BP systematic, ABSENT from training) -> does it silently miscalibrate?
Ranks from NON-rejecting posterior samples: post.sample(..., reject_outside_prior=False)
(sbi's default rejection sampler hangs on OOD/contaminated inputs; and to MEASURE
miscalibration we want the flow's actual draws, not ones forced into the prior).
n_post=50 -> discrete-rank uniform null for the 90% central band is 0.882 (as in npe_sbc.py).

Crash-resistant: fixed seeds, per-seed try/except, each seed persisted the moment it finishes
to outputs/p0/npe_sbc_realmodel.json, restart resumes (skips done seeds).
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import g1_realistic as g1   # load_library, WAVE, RINJ (BP systematic shape), gen

WAVE = g1.WAVE
NPIX = len(WAVE)
RINJ = g1.RINJ           # Riello/Huang BP blue-excess shape, mean-normalized
SNR = 30.0
DATA = ROOT / "data" / "real_spectra_large"

# --- build the PCA forward model from the real libraries (once, deterministic) -------------
K_MS, K_WD = 2, 3
FRAC_LO, FRAC_HI = 0.05, 0.30   # mirror g1_realistic.gen fracs


def _pca(L, k):
    mu = L.mean(0)
    Lc = L - mu
    U, S, Vt = np.linalg.svd(Lc, full_matrices=False)
    comps = Vt[:k]                 # (k, NPIX)
    scores = Lc @ comps.T          # (N, k)
    return mu, comps, scores


def build_model():
    MS, _ = g1.load_library(DATA, "ms", verbose=False)
    WD, _ = g1.load_library(DATA, "wd", verbose=False)
    assert MS.shape[0] >= 3 and WD.shape[0] >= 3, "need real templates"
    mu_ms, C_ms, S_ms = _pca(MS, K_MS)
    mu_wd, C_wd, S_wd = _pca(WD, K_WD)
    # prior box: empirical score span (+ log frac). Reconstructions stay positive over this box.
    lo = np.concatenate([S_ms.min(0), S_wd.min(0), [np.log10(FRAC_LO)]]).astype(np.float32)
    hi = np.concatenate([S_ms.max(0), S_wd.max(0), [np.log10(FRAC_HI)]]).astype(np.float32)
    M = dict(mu_ms=mu_ms, C_ms=C_ms, mu_wd=mu_wd, C_wd=C_wd,
             n_ms=MS.shape[0], n_wd=WD.shape[0])
    return M, lo, hi


NAMES = ["msPC1", "msPC2", "wdPC1", "wdPC2", "wdPC3", "log10_frac"]


def simulate(theta, M, rng, contaminate=0.0):
    """theta: (N,6) torch -> x: (N,NPIX) torch. Optional BP systematic of `contaminate` severity.

    Mirrors g1_realistic.gen flux-matching: secondary scaled to MS total flux, then *10^logfrac.
    """
    th = theta.detach().cpu().numpy()
    n = th.shape[0]
    s_ms = th[:, 0:K_MS]
    s_wd = th[:, K_MS:K_MS + K_WD]
    logf = th[:, K_MS + K_WD]
    ms = M["mu_ms"][None, :] + s_ms @ M["C_ms"]          # (n, NPIX) real-grounded MS
    wd = M["mu_wd"][None, :] + s_wd @ M["C_wd"]          # (n, NPIX) real-grounded WD
    frac = 10.0 ** logf                                  # (n,)
    # flux-match WD to MS total flux (== g1.gen: second * MS.sum()/second.sum())
    ratio = ms.sum(1) / wd.sum(1)                        # (n,)
    x0 = ms + (frac * ratio)[:, None] * wd
    if contaminate > 0:
        x0 = x0 + contaminate * x0.mean(1, keepdims=True) * RINJ[None, :]
    sig = x0.mean(1, keepdims=True) / SNR
    x = x0 + rng.standard_normal(x0.shape) * sig
    return torch.as_tensor(x, dtype=torch.float32)


def run(seed=0, n_train=8000):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    from sbi.inference import NPE
    from sbi.utils import BoxUniform
    from scipy.stats import kstest

    M, lo, hi = build_model()
    LOW = torch.as_tensor(lo)
    HIGH = torch.as_tensor(hi)
    prior = BoxUniform(low=LOW, high=HIGH)

    theta = prior.sample((n_train,))
    x = simulate(theta, M, rng)
    print(f"[seed {seed}] training NPE on {n_train} real-model sims "
          f"(n_ms={M['n_ms']} n_wd={M['n_wd']}) ...", flush=True)
    inf = NPE(prior=prior, density_estimator="nsf")
    de = inf.append_simulations(theta, x).train(show_train_summary=False)
    post = inf.build_posterior()
    # best validation log-prob: a quick 'did the flow train' check
    try:
        best_vlp = float(inf.summary["best_validation_log_prob"][-1])
    except Exception:
        best_vlp = None

    npar = len(NAMES)

    def sbc_block(contaminate, n_sbc=150, n_post=50):
        th = prior.sample((n_sbc,))
        xo = simulate(th, M, rng, contaminate=contaminate)
        ranks = np.zeros((n_sbc, npar), dtype=int)
        for i in range(n_sbc):
            try:
                s = post.sample((n_post,), x=xo[i], reject_outside_prior=False,
                                show_progress_bars=False)
            except TypeError:
                s = post.posterior_estimator.sample(
                    (n_post,), condition=xo[i].unsqueeze(0)).reshape(n_post, npar)
            s = s.detach().cpu().numpy()
            ranks[i] = (s < th[i].numpy()[None, :]).sum(0)
        pit = ranks / n_post
        ks = [float(kstest(pit[:, j], "uniform").pvalue) for j in range(npar)]
        cov90 = [round(float(np.mean((pit[:, j] >= 0.05) & (pit[:, j] <= 0.95))), 3)
                 for j in range(npar)]
        return {"ks_pvals": [round(v, 4) for v in ks], "ks_min": round(min(ks), 5),
                "coverage90_per_param": cov90,
                "mean_coverage90": round(float(np.mean(cov90)), 3)}

    print("SBC clean ...", flush=True)
    clean = sbc_block(0.0)
    print("SBC under BP systematic (sev 0.05) ...", flush=True)
    contam = sbc_block(0.05)

    res = {
        "config": {"seed": seed, "n_train": n_train, "snr": SNR, "params": NAMES,
                   "K_MS": K_MS, "K_WD": K_WD, "frac_range": [FRAC_LO, FRAC_HI],
                   "n_ms_templates": M["n_ms"], "n_wd_templates": M["n_wd"],
                   "discrete_rank_null_cov90": 0.882, "n_post": 50},
        "best_validation_log_prob": best_vlp,
        "SBC_clean": clean,
        "SBC_under_BP_systematic_sev0.05": contam,
    }
    print(f"[seed {seed}, n_train {n_train}] clean cov90={clean['mean_coverage90']} "
          f"ks_min={clean['ks_min']} | contam cov90={contam['mean_coverage90']} "
          f"ks_min={contam['ks_min']} | best_vlp={best_vlp}", flush=True)
    return res


def _summarize(runs, n_total):
    return {
        "model": "REAL-SPECTRA PCA forward model (120 MS + 120 WD Gaia XP templates)",
        "parametrization": ("theta = (msPC1,msPC2, wdPC1,wdPC2,wdPC3, log10_frac); "
                            "MS=mu_ms+s.PC_ms (2PC,>99%var), WD=mu_wd+s.PC_wd (3PC,~96%var); "
                            "prior box = empirical score span; x = MS + 10^logfrac * WD*(MS.sum/WD.sum) "
                            "+ Gaussian(SNR=30); BP systematic = severity*mean*RINJ injected at inference."),
        "runs": runs, "n_done": len(runs), "n_total": n_total,
        "discrete_rank_null_cov90": 0.882,
        "clean_cov90_across": [r["SBC_clean"]["mean_coverage90"] for r in runs],
        "contam_cov90_across": [r["SBC_under_BP_systematic_sev0.05"]["mean_coverage90"] for r in runs],
        "clean_ksmin_across": [r["SBC_clean"]["ks_min"] for r in runs],
        "contam_ksmin_across": [r["SBC_under_BP_systematic_sev0.05"]["ks_min"] for r in runs],
        "best_vlp_across": [r.get("best_validation_log_prob") for r in runs],
        "reading": ("If clean 90% coverage sits near 0.882 (rank-null) and KS is uniform, while "
                    "BP-contaminated coverage collapses well below with KS->0, the BP systematic "
                    "silently miscalibrates the amortized posterior ON THE REAL-SPECTRA model too "
                    "(referee-proofs the optical analog of arXiv:2606.17098)."),
    }


if __name__ == "__main__":
    configs = [(0, 8000), (1, 8000), (2, 8000)]
    od = ROOT / "outputs" / "p0" / "npe_sbc_realmodel"
    od.mkdir(parents=True, exist_ok=True)
    out = ROOT / "outputs" / "p0" / "npe_sbc_realmodel.json"
    runs = []
    for (s, nt) in configs:
        pf = od / f"seed{s}_n{nt}.json"
        if pf.exists():
            print(f"[resume] {pf.name} exists -> skip", flush=True)
            runs.append(json.loads(pf.read_text()))
        else:
            try:
                r = run(seed=s, n_train=nt)
                pf.write_text(json.dumps(r, indent=2))
                runs.append(r)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[seed {s} n {nt}] FAILED: {e} -- continuing", flush=True)
                continue
        out.write_text(json.dumps(_summarize(runs, len(configs)), indent=2))
        print(f"[persisted {len(runs)}/{len(configs)}]", flush=True)
    print(json.dumps(_summarize(runs, len(configs)), indent=2))
