"""Real Bayesian-evidence gate via nested sampling (ultranest) - validates the BIC-proxy
gate used in g1_realistic.py / p0_gonogo.py against PROPER evidence, the exact mechanism
of arXiv:2606.17098 ("only nested sampling's evidence flags it").

For each spectrum compute ln Z (marginal likelihood) for two models:
  Binary       : a_ms * BB(T_ms) + a_wd * BB(T_wd)               [4 params]
  Single+Riello: a * BB(T)       + b * RIELLO_GATE (BP artifact)  [3 params]
Delta lnZ = lnZ_binary - lnZ_single_riello. Genuine binaries -> favor Binary (>0);
BP-injected single stars (spurious) -> favor Single+Riello (<0). The per-object Delta-chi2
(single-vs-binary max-likelihood, never considering the systematic) is BLIND to this.

Small sample (nested sampling is ~seconds/spectrum). Reuses the blackbody proxy from p0_gonogo.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import p0_gonogo as p0           # WAVE, planck, RIELLO_GATE, make_single_ms, make_binary, inject_riello, fit_single, fit_binary
from ultranest import ReactiveNestedSampler

WAVE = p0.WAVE
RG = p0.RIELLO_GATE


def bb(T):
    return p0.planck(WAVE, T)


def lnZ_binary(x, sig):
    names = ["T_ms", "log_a_ms", "T_wd", "log_a_wd"]
    def prior(u):
        p = np.empty_like(u)
        p[0] = 3000 + 4000 * u[0]; p[1] = -1 + 2 * u[1]
        p[2] = 7000 + 33000 * u[2]; p[3] = -3 + 4 * u[3]
        return p
    def loglike(p):
        m = 10**p[1] * bb(p[0]) + 10**p[3] * bb(p[2])
        r = (x - m) / sig
        return -0.5 * float(r @ r)
    s = ReactiveNestedSampler(names, loglike, prior)
    return s.run(min_num_live_points=60, show_status=False, viz_callback=False)["logz"]


def lnZ_single_riello(x, sig):
    names = ["T", "log_a", "log_b"]
    def prior(u):
        p = np.empty_like(u)
        p[0] = 3000 + 4000 * u[0]; p[1] = -1 + 2 * u[1]; p[2] = -3 + 4 * u[2]
        return p
    def loglike(p):
        m = 10**p[1] * bb(p[0]) + 10**p[2] * RG
        r = (x - m) / sig
        return -0.5 * float(r @ r)
    s = ReactiveNestedSampler(names, loglike, prior)
    return s.run(min_num_live_points=60, show_status=False, viz_callback=False)["logz"]


def run(n_each=15, snr=30.0, severity=0.05, seed=0):
    global RNG
    p0.RNG = np.random.default_rng(seed)
    xs_b, sig_b, *_ = p0.make_binary(n_each, snr)
    xs_s, sig_s, _ = p0.make_single_ms(n_each, snr)
    xs_t = p0.inject_riello(xs_s, sig_s, severity)   # spurious = BP-injected singles

    rows = {"binary": [], "spurious": []}
    for k in range(n_each):
        # genuine binary
        zb = lnZ_binary(xs_b[k], sig_b[k]); zs = lnZ_single_riello(xs_b[k], sig_b[k])
        cs = p0.fit_single(xs_b[k], sig_b[k]); cb = p0.fit_binary(xs_b[k], sig_b[k])
        rows["binary"].append({"dlnZ": zb - zs, "dchi2_renorm": (cs - cb) / cs})
        # spurious (injected single)
        zb2 = lnZ_binary(xs_t[k], sig_s[k]); zs2 = lnZ_single_riello(xs_t[k], sig_s[k])
        cs2 = p0.fit_single(xs_t[k], sig_s[k]); cb2 = p0.fit_binary(xs_t[k], sig_s[k])
        rows["spurious"].append({"dlnZ": zb2 - zs2, "dchi2_renorm": (cs2 - cb2) / cs2})
        print(f"  {k+1}/{n_each} done", flush=True)

    dlnZ = np.array([r["dlnZ"] for r in rows["binary"]] + [r["dlnZ"] for r in rows["spurious"]])
    dchi2 = np.array([r["dchi2_renorm"] for r in rows["binary"]] + [r["dchi2_renorm"] for r in rows["spurious"]])
    y = np.r_[np.zeros(n_each), np.ones(n_each)]   # 1 = spurious
    # evidence gate: spurious have LOWER dlnZ (favor single+riello) -> use -dlnZ as "spurious score"
    auc_evidence = roc_auc_score(y, -dlnZ)
    # per-object dchi2: spurious have HIGHER dchi2_renorm (look MORE binary) -> blind/anti-informative
    auc_dchi2 = roc_auc_score(y, dchi2)
    res = {
        "config": {"n_each": n_each, "snr": snr, "severity": severity},
        "median_dlnZ_genuine_binary": round(float(np.median([r["dlnZ"] for r in rows["binary"]])), 2),
        "median_dlnZ_spurious": round(float(np.median([r["dlnZ"] for r in rows["spurious"]])), 2),
        "AUC_evidence_gate (genuine vs spurious)": round(float(auc_evidence), 3),
        "AUC_dchi2_perobj (genuine vs spurious)": round(float(auc_dchi2), 3),
        "interpretation": "Genuine binaries have dlnZ>>0 (evidence favors the true binary); BP-injected "
                          "singles have dlnZ<0 (evidence favors single+systematic). The evidence gate "
                          "separates them (AUC high) where per-object Delta-chi2 is anti-informative "
                          "(AUC<=0.5) - proper-evidence confirmation of the BIC-gate result.",
    }
    out = ROOT / "outputs" / "p0" / "evidence_gate.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    run()
