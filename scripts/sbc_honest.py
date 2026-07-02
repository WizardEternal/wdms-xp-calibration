"""Honest-amplitude SBC control. The note's SBC leg injects BP as severity*x0.mean*RINJ
(0.05 of TOTAL flux as a blue bump) -> same ~29x over-scaling as the injection leg. This
re-runs the calibration collapse at HONEST multiplicative-local amplitudes (local excess <= a)
to map the SBC coverage turn-on curve, parallel to the injection-leg curve. One trained flow
(seed 0, n_train=8000), reused across all arms. Writes sbc_honest.json."""
import sys, json, warnings
from pathlib import Path
import numpy as np, torch
warnings.filterwarnings("ignore")
ROOT = Path(r"C:/Users/Karan/Documents/Python stuff/cool papers/wdms-xp-calibration")
sys.path.insert(0, str(ROOT / "scripts"))
import npe_sbc_realmodel as m
from scipy.stats import kstest

WAVE = m.WAVE
RINJ = m.RINJ                                   # unit-mean half-cosine (the note's injection profile)
# shape01: 0..1, peak 1 at 392nm -> 0 at 680nm, 0 in RP. local excess of x*(1+a*shape01) is <= a.
SHAPE01 = np.where(WAVE <= 680.0,
                   np.cos(0.5*np.pi*np.clip((WAVE-392.0)/(680.0-392.0),0,1))**2, 0.0)


def simulate_arm(theta, M, rng, kind, amp):
    th = theta.detach().cpu().numpy()
    s_ms = th[:, 0:m.K_MS]; s_wd = th[:, m.K_MS:m.K_MS+m.K_WD]; logf = th[:, m.K_MS+m.K_WD]
    ms = M["mu_ms"][None] + s_ms @ M["C_ms"]; wd = M["mu_wd"][None] + s_wd @ M["C_wd"]
    frac = 10.0**logf; ratio = ms.sum(1)/wd.sum(1)
    x0 = ms + (frac*ratio)[:, None]*wd
    if kind == "note_total":            # the note's injection: amp of TOTAL flux as a blue bump
        x0 = x0 + amp * x0.mean(1, keepdims=True) * RINJ[None]
    elif kind == "honest_local":        # honest: multiplicative, local excess <= amp
        x0 = x0 * (1.0 + amp * SHAPE01[None])
    sig = x0.mean(1, keepdims=True)/m.SNR
    return torch.as_tensor(x0 + rng.standard_normal(x0.shape)*sig, dtype=torch.float32)


def main(seed=0, n_train=8000, n_sbc=150, n_post=50):
    from sbi.inference import NPE
    from sbi.utils import BoxUniform
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    M, lo, hi = m.build_model()
    prior = BoxUniform(low=torch.as_tensor(lo), high=torch.as_tensor(hi))
    theta = prior.sample((n_train,)); x = m.simulate(theta, M, rng)
    print(f"training flow seed={seed} n_train={n_train} ...", flush=True)
    inf = NPE(prior=prior, density_estimator="nsf")
    inf.append_simulations(theta, x).train(show_train_summary=False)
    post = inf.build_posterior()
    npar = len(m.NAMES)

    def block(kind, amp):
        th = prior.sample((n_sbc,)); xo = simulate_arm(th, M, rng, kind, amp)
        ranks = np.zeros((n_sbc, npar), int)
        for i in range(n_sbc):
            s = post.sample((n_post,), x=xo[i], reject_outside_prior=False,
                            show_progress_bars=False).detach().cpu().numpy()
            ranks[i] = (s < th[i].numpy()[None]).sum(0)
        pit = ranks/n_post
        ks = [float(kstest(pit[:, j], "uniform").pvalue) for j in range(npar)]
        cov = [round(float(np.mean((pit[:, j] >= 0.05) & (pit[:, j] <= 0.95))), 3) for j in range(npar)]
        return {"mean_coverage90": round(float(np.mean(cov)), 3),
                "coverage90_per_param": cov, "ks_min": round(min(ks), 5)}

    arms = [("clean", "clean", 0.0),
            ("note_total_0.05", "note_total", 0.05),   # the note's SBC injection (reproduce 0.468)
            ("honest_local_0.02", "honest_local", 0.02),
            ("honest_local_0.05", "honest_local", 0.05),
            ("honest_local_0.10", "honest_local", 0.10),
            ("honest_local_0.20", "honest_local", 0.20),
            ("honest_local_0.50", "honest_local", 0.50)]
    res = {"config": {"seed": seed, "n_train": n_train, "discrete_rank_null_cov90": 0.882,
                      "params": m.NAMES, "note": "note injects 0.05 of TOTAL; honest is multiplicative local"}}
    for name, kind, amp in arms:
        res[name] = block(kind, amp)
        print(f"  {name:20s} cov90={res[name]['mean_coverage90']} ks_min={res[name]['ks_min']} "
              f"per_par={res[name]['coverage90_per_param']}", flush=True)
    out = ROOT / "outputs" / "p0" / "sbc_honest.json"
    out.write_text(json.dumps(res, indent=2))
    print("wrote", out, flush=True)
    return res


if __name__ == "__main__":
    main()
