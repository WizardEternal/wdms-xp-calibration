"""NPE + SBC calibration diagnostic (the 'calibration' in 'calibration audit').

Train an amortized neural posterior (sbi NPE) on a parametric WD-MS binary forward model
(blackbody MS + blackbody WD -> 61-px XP grid + Gaussian noise), then:
  (1) SBC (simulation-based calibration, Talts+2018) on CLEAN held-out data -> is the
      amortized posterior calibrated?
  (2) SBC UNDER the Gaia BP-band systematic (inject the blue excess into the observed
      spectra at inference time) -> does the systematic silently MISCALIBRATE the posterior?
The optical analog of arXiv:2606.17098's SBC + gain-shift-miscalibration result.

Honest by construction: reports rank-uniformity KS p-values and credible-interval coverage
for clean vs contaminated; no claim is made beyond what the numbers show.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import torch
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import p0_gonogo as p0   # planck, WAVE, RIELLO (BP systematic shape)

WAVE = p0.WAVE
NPIX = len(WAVE)
RIELLO = p0.RIELLO
SNR = 30.0
torch.manual_seed(0)
RNG = np.random.default_rng(0)

# prior: [T_ms, log10 a_ms, T_wd, log10 a_wd]  (WD subdominant)
LOW = torch.tensor([3500., -0.5, 9000., -2.5])
HIGH = torch.tensor([6000., 0.5, 35000., -0.5])


def planck_np(T):
    return p0.planck(WAVE, T)


def simulate(theta, contaminate=0.0):
    """theta: (N,4) torch -> x: (N,NPIX) torch. Optional BP systematic of `contaminate` severity."""
    th = theta.detach().cpu().numpy()
    n = th.shape[0]
    x0 = np.empty((n, NPIX))
    for k in range(n):
        ms = planck_np(th[k, 0]); wd = planck_np(th[k, 2])
        x0[k] = 10**th[k, 1] * ms + 10**th[k, 3] * wd
    if contaminate > 0:
        x0 = x0 + contaminate * x0.mean(1, keepdims=True) * RIELLO[None, :]
    sig = x0.mean(1, keepdims=True) / SNR
    x = x0 + RNG.standard_normal(x0.shape) * sig
    return torch.as_tensor(x, dtype=torch.float32)


def run(seed=0, n_train=8000):
    global RNG
    RNG = np.random.default_rng(seed)
    torch.manual_seed(seed)
    from sbi.inference import NPE
    from sbi.utils import BoxUniform
    from sbi.diagnostics import run_sbc, check_sbc

    prior = BoxUniform(low=LOW, high=HIGH)
    N_TRAIN = n_train
    theta = prior.sample((N_TRAIN,))
    x = simulate(theta)
    print(f"[seed {seed}] training NPE on {N_TRAIN} sims ...", flush=True)
    inf = NPE(prior=prior, density_estimator="nsf")
    inf.append_simulations(theta, x).train(show_train_summary=False)
    post = inf.build_posterior()

    names = ["T_ms", "log_a_ms", "T_wd", "log_a_wd"]

    from scipy.stats import kstest

    def sbc_block(contaminate, n_sbc=150, n_post=50):
        """SBC by manual ranks from NON-rejecting posterior samples. Robust to the
        OOD-leakage that makes sbi's default rejection sampler hang on contaminated
        (out-of-distribution) inputs - and statistically correct: to MEASURE
        miscalibration we want the flow's actual samples, not ones forced into the prior."""
        th = prior.sample((n_sbc,))
        xo = simulate(th, contaminate=contaminate)
        npar = len(names)
        ranks = np.zeros((n_sbc, npar), dtype=int)
        for i in range(n_sbc):
            try:
                s = post.sample((n_post,), x=xo[i], reject_outside_prior=False, show_progress_bars=False)
            except TypeError:
                s = post.posterior_estimator.sample((n_post,), condition=xo[i].unsqueeze(0)).reshape(n_post, npar)
            s = s.detach().cpu().numpy()
            ranks[i] = (s < th[i].numpy()[None, :]).sum(0)
        pit = ranks / n_post
        ks = [float(kstest(pit[:, j], "uniform").pvalue) for j in range(npar)]
        cov90 = [round(float(np.mean((pit[:, j] >= 0.05) & (pit[:, j] <= 0.95))), 3) for j in range(npar)]
        return {"ks_pvals": [round(v, 4) for v in ks], "ks_min": round(min(ks), 5),
                "coverage90_per_param": cov90, "mean_coverage90": round(float(np.mean(cov90)), 3)}

    print("SBC clean ...", flush=True)
    clean = sbc_block(0.0)
    print("SBC under BP systematic (sev 0.05) ...", flush=True)
    contam = sbc_block(0.05)

    res = {
        "config": {"seed": seed, "n_train": N_TRAIN, "snr": SNR, "params": names},
        "SBC_clean": clean,
        "SBC_under_BP_systematic_sev0.05": contam,
    }
    print(f"[seed {seed}, n_train {N_TRAIN}] clean cov90={clean['mean_coverage90']} ks_min={clean['ks_min']} "
          f"| contam cov90={contam['mean_coverage90']} ks_min={contam['ks_min']}", flush=True)
    return res


def _summarize(runs, n_total):
    return {
        "runs": runs, "n_done": len(runs), "n_total": n_total,
        "clean_cov90_across": [r["SBC_clean"]["mean_coverage90"] for r in runs],
        "contam_cov90_across": [r["SBC_under_BP_systematic_sev0.05"]["mean_coverage90"] for r in runs],
        "clean_ksmin_across": [r["SBC_clean"]["ks_min"] for r in runs],
        "contam_ksmin_across": [r["SBC_under_BP_systematic_sev0.05"]["ks_min"] for r in runs],
        "reading": ("If clean coverage rises toward 0.9 with more training (n_train=30000) while contam "
                    "stays well below, the BP systematic robustly miscalibrates the amortized posterior "
                    "(optical analog of the X-ray result) and the clean under-coverage was undertraining."),
    }


if __name__ == "__main__":
    # CRASH-RESISTANT robustness pass: each flow is persisted the moment it finishes, the
    # rolling summary is rewritten after every flow, and a restart RESUMES (skips done flows).
    # Mirrors the X-ray paper's reseed + larger-training control.
    configs = [(0, 8000), (1, 8000), (2, 8000), (0, 30000)]
    od = ROOT / "outputs" / "p0" / "npe_sbc"
    od.mkdir(parents=True, exist_ok=True)
    runs = []
    for (s, nt) in configs:
        pf = od / f"seed{s}_n{nt}.json"
        if pf.exists():                                  # resume: reuse a completed flow
            print(f"[resume] {pf.name} exists -> skip", flush=True)
            runs.append(json.loads(pf.read_text()))
        else:
            try:
                r = run(seed=s, n_train=nt)
                pf.write_text(json.dumps(r, indent=2))   # persist this flow immediately
                runs.append(r)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[seed {s} n {nt}] FAILED: {e} -- continuing", flush=True)
                continue
        # rewrite the rolling summary after every flow (partial results always on disk)
        (ROOT / "outputs" / "p0" / "npe_sbc_robust.json").write_text(json.dumps(_summarize(runs, len(configs)), indent=2))
        print(f"[persisted {len(runs)}/{len(configs)}]", flush=True)
    print(json.dumps(_summarize(runs, len(configs)), indent=2))
