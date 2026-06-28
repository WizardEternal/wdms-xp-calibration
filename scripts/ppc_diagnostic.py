"""Posterior-predictive check (PPC) on the NPE setup - does a PPC WARN that the
BP-systematic-contaminated posterior is wrong (caught), or is it silent?

For each test spectrum x: take the NPE posterior median theta_hat, model spectrum m(theta_hat),
observed discrepancy T_obs = chi2(x, m). Draw K noise replicates x_rep ~ m + noise; PPC
p-value = mean(chi2(x_rep, m) >= T_obs). Low p => observed misfits the posterior-predictive
=> a warning. Report AUC of (1 - p) separating CLEAN vs BP-CONTAMINATED test spectra.
  AUC ~ 0.5 => PPC is BLIND (the miscalibration is truly silent, even to a PPC).
  AUC >> 0.5 => a PPC CATCHES it (an in-the-loop GoF check would warn you).
Reuses the parametric forward model from npe_sbc.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np, torch
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import npe_sbc as M
from sklearn.metrics import roc_auc_score


def main(seed=0, n_train=8000, n_test=150, K=100):
    torch.manual_seed(seed); M.RNG = np.random.default_rng(seed)
    from sbi.inference import NPE
    from sbi.utils import BoxUniform
    prior = BoxUniform(low=M.LOW, high=M.HIGH)
    th = prior.sample((n_train,)); x = M.simulate(th)
    print("training NPE ...", flush=True)
    inf = NPE(prior=prior, density_estimator="nsf"); inf.append_simulations(th, x).train(show_train_summary=False)
    post = inf.build_posterior()

    def model_noiseless(theta_row):
        ms = M.planck_np(theta_row[0]); wd = M.planck_np(theta_row[2])
        return 10**theta_row[1] * ms + 10**theta_row[3] * wd

    def ppc_pvals(contaminate):
        tt = prior.sample((n_test,))
        xo = M.simulate(tt, contaminate=contaminate).numpy()
        pv = np.zeros(n_test)
        for i in range(n_test):
            xi = torch.as_tensor(xo[i], dtype=torch.float32)
            try:
                s = post.sample((200,), x=xi, reject_outside_prior=False, show_progress_bars=False)
            except TypeError:
                s = post.posterior_estimator.sample((200,), condition=xi.unsqueeze(0)).reshape(200, 4)
            thr = np.median(s.detach().cpu().numpy(), 0)
            m = model_noiseless(thr)
            sig = m.mean() / M.SNR
            T_obs = float(np.sum(((xo[i] - m) / sig) ** 2))
            reps = m[None, :] + M.RNG.standard_normal((K, len(m))) * sig
            T_rep = np.sum(((reps - m[None, :]) / sig) ** 2, 1)
            pv[i] = float(np.mean(T_rep >= T_obs))
        return pv

    print("PPC clean ...", flush=True); p_clean = ppc_pvals(0.0)
    print("PPC contaminated (sev 0.05) ...", flush=True); p_contam = ppc_pvals(0.05)
    # 1 - p = "warning score"; AUC separating clean (0) vs contaminated (1)
    y = np.r_[np.zeros(n_test), np.ones(n_test)]
    auc = float(roc_auc_score(y, np.r_[1 - p_clean, 1 - p_contam]))
    res = {"config": {"seed": seed, "n_train": n_train, "n_test": n_test, "K": K},
           "median_ppc_p_clean": round(float(np.median(p_clean)), 4),
           "median_ppc_p_contaminated": round(float(np.median(p_contam)), 4),
           "frac_contam_flagged_p<0.05": round(float(np.mean(p_contam < 0.05)), 3),
           "frac_clean_flagged_p<0.05": round(float(np.mean(p_clean < 0.05)), 3),
           "AUC_PPC_clean_vs_contaminated": round(auc, 3),
           "reading": "AUC~0.5 => PPC BLIND (truly silent, even a posterior-predictive check doesn't warn). "
                      "AUC>>0.5 => a PPC CATCHES the contamination (an in-the-loop GoF check would warn)."}
    (ROOT / "outputs" / "p0" / "ppc_diagnostic.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
