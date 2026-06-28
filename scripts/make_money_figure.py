"""Consolidated paper money figure - 3 panels:
(A) G1 proxy: BP systematic manufactures spurious WD-MS that per-object Delta-chi2 can't flag; gate catches.
(B) G1 REAL Gaia XP (leave-one-out): same result on real spectra, with the bootstrap error bar at the
    realistic ~2% residual.
(C) GALEX FUV-detection (WD-presence floor) across catalog subsets (Wilson 95%).
Reads only saved JSON outputs.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P0 = json.loads((ROOT / "outputs/p0/sweep.json").read_text())
RS = json.loads((ROOT / "outputs/p0/g1_realistic_sweep.json").read_text())
BS = json.loads((ROOT / "outputs/p0/g1_realistic_bootstrap.json").read_text())
CC = json.loads((ROOT / "outputs/audit/conformal_certificate.json").read_text())

fig, ax = plt.subplots(1, 3, figsize=(18, 5))

# --- (A) G1 proxy ---
sev = [r["severity"] for r in P0]
ax[0].plot(sev, [r["spurious_rate"] for r in P0], "o-", color="crimson", label="spurious rate")
ax[0].plot(sev, [r["baseline_fpr"] for r in P0], "k--", lw=1, label="baseline FPR")
ax[0].plot(sev, [r["auc_gate_honest"] for r in P0], "s-", color="seagreen", label="gate AUC (honest)")
ax[0].plot(sev, [r["auc_dchi2_perobj"] for r in P0], "^-", color="navy", label="per-object Delta-chi2 AUC")
ax[0].axhline(0.5, color="grey", ls=":", lw=1); ax[0].axvspan(0.0, 0.05, color="gold", alpha=0.18)
ax[0].set(xlabel="BP-excess severity", ylabel="rate / AUC", ylim=(-0.02, 1.05),
          title="(A) G1 - blackbody proxy\nDelta-chi2 anti-informative; gate catches")
ax[0].legend(fontsize=8, loc="center right")

# --- (B) G1 real Gaia XP (LOO) ---
sv = [r["severity"] for r in RS]
ax[1].plot(sv, [r["spurious_rate"] for r in RS], "o-", color="crimson", label="spurious rate")
ax[1].plot(sv, [r["auc_gate_honest"] for r in RS], "s-", color="seagreen", label="gate AUC (honest)")
ax[1].plot(sv, [r["auc_dchi2_perobj"] for r in RS], "^-", color="navy", label="per-object Delta-chi2 AUC")
ax[1].plot(sv, [r["gof_AUC_genuine_vs_spurious"] for r in RS], "d-", color="darkorange", label="per-object GoF AUC")
# bootstrap error bars at sev 0.02
b = lambda k: (BS[k]["mean"], BS[k]["std"])
ax[1].errorbar([0.02], [b("auc_gate_honest")[0]], yerr=[b("auc_gate_honest")[1]], fmt="s", color="seagreen", capsize=4)
ax[1].errorbar([0.02], [b("auc_dchi2_perobj")[0]], yerr=[b("auc_dchi2_perobj")[1]], fmt="^", color="navy", capsize=4)
ax[1].errorbar([0.02], [b("gof_AUC_genuine_vs_spurious")[0]], yerr=[b("gof_AUC_genuine_vs_spurious")[1]], fmt="d", color="darkorange", capsize=4)
ax[1].axhline(0.5, color="grey", ls=":", lw=1); ax[1].axvspan(0.0, 0.03, color="gold", alpha=0.18)
ax[1].set(xlabel="BP-excess severity", ylabel="rate / AUC (LOO)", ylim=(-0.02, 1.05),
          title="(B) G1 - REAL Gaia XP, leave-one-out\nsame result; err bars = 6-seed bootstrap @ 2%")
ax[1].legend(fontsize=8, loc="center right")

# --- (C) FUV floor ---
pf = CC["purity_floor_fuv"]
groups = [("low third", pf["prob_binary_low_third"]), ("mid third", pf["prob_binary_mid_third"]),
          ("high third", pf["prob_binary_high_third"]), ("WD on-seq", pf["wd_on_sequence"]),
          ("WD off-seq", pf["wd_off_sequence"]), ("ALL", pf["ALL_galex_covered"])]
frac = [g[1]["fuv_det_frac"] for g in groups]
lo = [g[1]["fuv_det_frac"] - g[1]["wilson95"][0] for g in groups]
hi = [g[1]["wilson95"][1] - g[1]["fuv_det_frac"] for g in groups]
colors = ["#9ecae1", "#4292c6", "#08519c", "seagreen", "crimson", "grey"]
y = np.arange(len(groups))
ax[2].barh(y, frac, xerr=[lo, hi], color=colors, alpha=0.85, capsize=3)
for i, g in enumerate(groups):
    ax[2].text(frac[i] + max(hi) + 0.01, i, f"n={g[1]['n_galex']}", va="center", fontsize=8)
ax[2].set(yticks=y, yticklabels=[g[0] for g in groups], xlim=(0, 0.75),
          xlabel="GALEX FUV-detection (WD-present floor)",
          title="(C) Real-data WD-presence floor (Wilson 95%)\nprob_binary informative; off-seq UV-deficient")
ax[2].invert_yaxis()

fig.tight_layout()
out = ROOT / "outputs" / "money_figure.png"
fig.savefig(out, dpi=150)
print("saved", out)
