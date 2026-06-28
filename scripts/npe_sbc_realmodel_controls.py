"""Adversarial controls for npe_sbc_realmodel.py.

Question the headline result must survive: is the BP-contaminated coverage collapse SPECIFIC
to the Riello/Huang BP blue-excess shape, or would ANY equal-magnitude perturbation collapse
coverage (i.e. is it just 'NPE is fragile to OOD')?

Controls (all injected at inference time at the SAME severity=0.05, same relative magnitude):
  * 'bp'    : the real RINJ blue-excess shape (the headline systematic)            -> should collapse
  * 'flat'  : a flat (grey) offset of the same mean magnitude                       -> degenerate w/ amplitude; expect mild
  * 'red'   : a RED-band shape (mirror of RINJ, excess at lambda>680nm)             -> different spectral region
  * 'noise2x': double the Gaussian noise, no shape                                  -> pure stochastic, no systematic
Reports clean + each control's 90% coverage (mean and per-param) and KS-min for one trained
flow (seed 0, n_train=8000). Self-contained, reuses the npe_sbc_realmodel forward model.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np, torch
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import npe_sbc_realmodel as m
import g1_realistic as g1
from scipy.stats import kstest

WAVE = m.WAVE
RINJ = m.RINJ
# red-band mirror of the blue RINJ: excess concentrated at long wavelengths, mean-normalized
_red = np.where(WAVE >= 680.0,
                np.cos(0.5 * np.pi * np.clip((992.0 - WAVE) / (992.0 - 680.0), 0, 1)) ** 2, 0.0)
RRED = _red / _red.mean()


def simulate_ctrl(theta, M, rng, kind="clean", severity=0.05):
    th = theta.detach().cpu().numpy(); n = th.shape[0]
    s_ms = th[:, 0:m.K_MS]; s_wd = th[:, m.K_MS:m.K_MS + m.K_WD]; logf = th[:, m.K_MS + m.K_WD]
    ms = M["mu_ms"][None] + s_ms @ M["C_ms"]; wd = M["mu_wd"][None] + s_wd @ M["C_wd"]
    frac = 10.0 ** logf; ratio = ms.sum(1) / wd.sum(1)
    x0 = ms + (frac * ratio)[:, None] * wd
    noise_mult = 1.0
    if kind == "bp":
        x0 = x0 + severity * x0.mean(1, keepdims=True) * RINJ[None]
    elif kind == "red":
        x0 = x0 + severity * x0.mean(1, keepdims=True) * RRED[None]
    elif kind == "flat":
        x0 = x0 + severity * x0.mean(1, keepdims=True)            # grey offset, same mean magnitude
    elif kind == "noise2x":
        noise_mult = 2.0
    sig = x0.mean(1, keepdims=True) / m.SNR
    x = x0 + rng.standard_normal(x0.shape) * sig * noise_mult
    return torch.as_tensor(x, dtype=torch.float32)


def main(seed=0, n_train=8000, n_sbc=150, n_post=50):
    from sbi.inference import NPE
    from sbi.utils import BoxUniform
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    M, lo, hi = m.build_model()
    prior = BoxUniform(low=torch.as_tensor(lo), high=torch.as_tensor(hi))
    theta = prior.sample((n_train,)); x = m.simulate(theta, M, rng)
    print(f"training control flow seed={seed} n_train={n_train} ...", flush=True)
    inf = NPE(prior=prior, density_estimator="nsf")
    inf.append_simulations(theta, x).train(show_train_summary=False)
    post = inf.build_posterior()
    npar = len(m.NAMES)

    def block(kind):
        th = prior.sample((n_sbc,)); xo = simulate_ctrl(th, M, rng, kind=kind)
        ranks = np.zeros((n_sbc, npar), int)
        for i in range(n_sbc):
            s = post.sample((n_post,), x=xo[i], reject_outside_prior=False,
                            show_progress_bars=False).detach().cpu().numpy()
            ranks[i] = (s < th[i].numpy()[None]).sum(0)
        pit = ranks / n_post
        ks = [float(kstest(pit[:, j], "uniform").pvalue) for j in range(npar)]
        cov = [round(float(np.mean((pit[:, j] >= 0.05) & (pit[:, j] <= 0.95))), 3) for j in range(npar)]
        return {"mean_coverage90": round(float(np.mean(cov)), 3),
                "coverage90_per_param": cov, "ks_min": round(min(ks), 5)}

    res = {"config": {"seed": seed, "n_train": n_train, "severity": 0.05,
                      "discrete_rank_null_cov90": 0.882, "params": m.NAMES},
           "clean": block("clean"), "bp": block("bp"), "red": block("red"),
           "flat": block("flat"), "noise2x": block("noise2x")}
    for k in ["clean", "bp", "red", "flat", "noise2x"]:
        print(f"  {k:8s} cov90={res[k]['mean_coverage90']} ks_min={res[k]['ks_min']} "
              f"per_par={res[k]['coverage90_per_param']}", flush=True)
    out = ROOT / "outputs" / "p0" / "npe_sbc_realmodel_controls.json"
    out.write_text(json.dumps(res, indent=2))
    print("wrote", out, flush=True)
    return res


if __name__ == "__main__":
    main()
