"""
Publication figures for the WD-MS Gaia XP calibration audit.
Reads the committed output JSONs and writes Fig 1-5 as .png and .pdf.

Legends are placed ABOVE the axes (never over the data); axis labels,
titles, and legend entries are capitalized.

Run:  python scripts/make_figures.py
"""
import json
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- paths -------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
P0 = os.path.join(ROOT, "outputs", "p0")
AUDIT = os.path.join(ROOT, "outputs", "audit")
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

# ---- style -------------------------------------------------------------
# colorblind-safe (Wong 2011) palette
C = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "vermil": "#D55E00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
    "grey":   "#999999",
}
mpl.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "figure.autolayout": False,
    "savefig.bbox": "tight",
})


def load(path):
    with open(path) as f:
        return json.load(f)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", name + ".png/.pdf")


def leg_above(ax, ncol, fs=8.0, y=1.17):
    """Place a legend above the axes (clear of the title and the data)."""
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol,
              frameon=False, fontsize=fs, handlelength=1.6,
              columnspacing=1.1, borderaxespad=0.0)


def wilson(k, n, z=1.96):
    """Wilson 95% interval, returns (lo, hi)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


# =======================================================================
# Fig 1 - G1 severity sweep on real Gaia XP (LOO), with proxy comparison
# =======================================================================
def fig1():
    real = load(os.path.join(P0, "g1_realistic_sweep.json"))
    boot = load(os.path.join(P0, "g1_realistic_bootstrap.json"))
    proxy = load(os.path.join(P0, "sweep.json"))

    sev = np.array([r["severity"] for r in real])
    spur = np.array([r["spurious_rate"] for r in real])
    gate = np.array([r["auc_gate_honest"] for r in real])
    gof = np.array([r["gof_AUC_genuine_vs_spurious"] for r in real])
    dchi = np.array([r["auc_dchi2_perobj"] for r in real])

    # bootstrap errors at severity 0.02 (the Huang residual)
    s0 = 0.02
    err = {
        "spur": boot["spurious_rate"]["std"],
        "gate": boot["auc_gate_honest"]["std"],
        "gof":  boot["gof_AUC_genuine_vs_spurious"]["std"],
        "dchi": boot["auc_dchi2_perobj"]["std"],
    }

    psev = np.array([r["severity"] for r in proxy])
    pspur = np.array([r["spurious_rate"] for r in proxy])
    pgate = np.array([r["auc_gate_honest"] for r in proxy])
    pdchi = np.array([r["auc_dchi2_perobj"] for r in proxy])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 4.2))

    # ---- panel A: real spectra (LOO) ----
    axA.axhline(0.5, color=C["grey"], ls=":", lw=1.0, zorder=1)
    axA.axhline(0.05, color=C["grey"], ls="--", lw=0.9, zorder=1)
    axA.plot(sev, spur, "-o", color=C["vermil"], label="Spurious-binary rate")
    axA.plot(sev, gate, "-s", color=C["blue"], label="Model-comparison gate AUC")
    axA.plot(sev, gof, "-^", color=C["green"], label="Per-object GoF AUC")
    axA.plot(sev, dchi, "-D", color=C["orange"], label=r"Per-object $\Delta\chi^2$ AUC")

    # bootstrap error bars at sev 0.02
    i0 = int(np.where(sev == s0)[0][0])
    axA.errorbar([s0], [spur[i0]], yerr=err["spur"], color=C["vermil"], capsize=3, lw=1.4)
    axA.errorbar([s0], [gate[i0]], yerr=err["gate"], color=C["blue"], capsize=3, lw=1.4)
    axA.errorbar([s0], [gof[i0]], yerr=err["gof"], color=C["green"], capsize=3, lw=1.4)
    axA.errorbar([s0], [dchi[i0]], yerr=err["dchi"], color=C["orange"], capsize=3, lw=1.4)
    axA.axvline(s0, color=C["purple"], ls="-", lw=0.8, alpha=0.6)
    axA.text(s0 + 0.004, 0.30, "Huang+2024\nresidual ~2%", color=C["purple"],
             ha="left", va="center", fontsize=7.5)

    axA.set_xlabel("Injected BP-excess severity")
    axA.set_ylabel("Rate / ROC AUC")
    axA.set_title("Real Gaia DR3 XP spectra (leave-one-out)")
    axA.set_ylim(-0.04, 1.08)
    axA.set_xlim(0.0, 0.215)

    # ---- panel B: blackbody proxy ----
    axB.axhline(0.5, color=C["grey"], ls=":", lw=1.0, zorder=1)
    axB.axhline(0.05, color=C["grey"], ls="--", lw=0.9, zorder=1)
    axB.plot(psev, pspur, "-o", color=C["vermil"], label="Spurious-binary rate")
    axB.plot(psev, pgate, "-s", color=C["blue"], label="Model-comparison gate AUC")
    axB.plot(psev, pdchi, "-D", color=C["orange"], label=r"Per-object $\Delta\chi^2$ AUC")
    axB.axvline(s0, color=C["purple"], ls="-", lw=0.8, alpha=0.6)
    axB.set_xlabel("Injected BP-excess severity")
    axB.set_ylabel("Rate / ROC AUC")
    axB.set_title("Blackbody proxy")
    axB.set_ylim(-0.04, 1.08)
    axB.set_xlim(0.0, 0.31)

    # shared legend across the top, above both panels (all four series)
    handles, labels = axA.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig1_g1_severity_sweep")


# =======================================================================
# Fig 2 - evidence gate: dlnZ and evidence AUC vs Delta-chi2 AUC
# =======================================================================
def fig2():
    g5 = load(os.path.join(P0, "evidence_gate_sev005.json"))
    g2 = load(os.path.join(P0, "evidence_gate_sev002.json"))

    sevs = [0.02, 0.05]
    dlnZ_gen = [g2["median_dlnZ_genuine_binary"], g5["median_dlnZ_genuine_binary"]]
    dlnZ_spur = [g2["median_dlnZ_spurious"], g5["median_dlnZ_spurious"]]
    auc_ev = [g2["AUC_evidence_gate (genuine vs spurious)"],
              g5["AUC_evidence_gate (genuine vs spurious)"]]
    auc_dc = [g2["AUC_dchi2_perobj (genuine vs spurious)"],
              g5["AUC_dchi2_perobj (genuine vs spurious)"]]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.0, 4.2))

    # ---- panel A: median dlnZ, genuine vs spurious ----
    x = np.arange(len(sevs))
    w = 0.36
    axA.axhline(0.0, color=C["grey"], ls="-", lw=0.8)
    axA.bar(x - w / 2, dlnZ_gen, w, color=C["blue"], label="Genuine binary")
    axA.bar(x + w / 2, dlnZ_spur, w, color=C["vermil"], label="BP-injected spurious")
    for xi, v in zip(x - w / 2, dlnZ_gen):
        axA.text(xi, v + 0.6, f"{v:+.1f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + w / 2, dlnZ_spur):
        off = 0.6 if v >= 0 else -0.9
        axA.text(xi, v + off, f"{v:+.1f}", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=8)
    axA.set_xticks(x)
    axA.set_xticklabels([f"Sev {s:.2f}" for s in sevs])
    axA.set_ylim(-7, 28)
    axA.set_ylabel(r"Median $\Delta\ln Z$ (binary vs single+systematic)")
    axA.set_title("Nested-sampling evidence")
    leg_above(axA, ncol=2)

    # ---- panel B: evidence AUC vs Delta-chi2 AUC ----
    w2 = 0.36
    axB.axhline(0.5, color=C["grey"], ls=":", lw=1.0)
    axB.bar(x - w2 / 2, auc_ev, w2, color=C["green"], label="Evidence gate")
    axB.bar(x + w2 / 2, auc_dc, w2, color=C["orange"], label=r"Per-object $\Delta\chi^2$")
    for xi, v in zip(x - w2 / 2, auc_ev):
        axB.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + w2 / 2, auc_dc):
        axB.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    axB.set_xticks(x)
    axB.set_xticklabels([f"Sev {s:.2f}" for s in sevs])
    axB.set_ylabel("ROC AUC (genuine vs spurious)")
    axB.set_ylim(0, 1.12)
    axB.set_title("Evidence gate vs per-object score")
    leg_above(axB, ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "fig2_evidence_gate")


# =======================================================================
# Fig 3 - catalog reliability: GALEX FUV-detection fraction by subset
# =======================================================================
def fig3():
    cert = load(os.path.join(AUDIT, "conformal_certificate.json"))
    pf = cert["purity_floor_fuv"]

    # ground-truth confirmed sample (FUV-detected 91% in 127, RESULTS R3)
    confirmed = {"label": "confirmed WD-MS", "n_galex": 127,
                 "fuv_det": int(round(0.91 * 127))}

    rows = [
        ("Prob. low third",   pf["prob_binary_low_third"], C["sky"]),
        ("Prob. mid third",   pf["prob_binary_mid_third"], C["sky"]),
        ("Prob. high third",  pf["prob_binary_high_third"], C["sky"]),
        ("On WD-sequence",    pf["wd_on_sequence"], C["green"]),
        ("Off WD-sequence",   pf["wd_off_sequence"], C["vermil"]),
        ("All GALEX-covered", pf["ALL_galex_covered"], C["grey"]),
        ("Confirmed WD-MS",   confirmed, C["blue"]),
    ]

    labels, fracs, los, his, ns, cols = [], [], [], [], [], []
    for name, d, col in rows:
        n = d["n_galex"]
        k = d["fuv_det"]
        frac = k / n
        lo, hi = wilson(k, n)
        labels.append(name)
        fracs.append(frac)
        los.append(frac - lo)
        his.append(hi - frac)
        ns.append(n)
        cols.append(col)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    y = np.arange(len(labels))[::-1]
    ax.errorbar(fracs, y, xerr=[los, his], fmt="o", color="black",
                ecolor="black", capsize=3, lw=1.2, zorder=3, ms=5)
    for yi, f, col, n in zip(y, fracs, cols, ns):
        ax.plot([f], [yi], "o", color=col, ms=8, zorder=4)
        ax.text(min(f + max(his) + 0.02, 0.98), yi, f"n={n}",
                va="center", ha="left", fontsize=7.5, color=C["grey"])
    ax.axvline(cert["purity_floor_fuv"]["ALL_galex_covered"]["fuv_det_frac"],
               color=C["grey"], ls="--", lw=0.9,
               label="Subset mean (0.47)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("GALEX FUV-detection fraction (WD-present floor)")
    ax.set_xlim(0.0, 1.05)
    ax.set_title("Certified WD-present fraction by subset (Wilson 95%)")
    leg_above(ax, ncol=1, y=1.10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig3_catalog_reliability")


# =======================================================================
# Fig 4 - NPE+SBC calibration: clean vs contaminated, + diagnostic hierarchy
# =======================================================================
def fig4():
    rob = load(os.path.join(P0, "npe_sbc_robust.json"))
    ppc = load(os.path.join(P0, "ppc_diagnostic.json"))
    g2 = load(os.path.join(P0, "evidence_gate_sev002.json"))

    clean = np.array(rob["clean_cov90_across"])
    contam = np.array(rob["contam_cov90_across"])
    runs = rob["runs"]
    labels = []
    for r in runs:
        c = r["config"]
        tag = "30k" if c["n_train"] == 30000 else f"Seed {c['seed']}"
        labels.append(tag)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 4.2))

    # ---- panel A: SBC 90% coverage per flow ----
    x = np.arange(len(clean))
    w = 0.36
    NULL = 0.882  # discrete-estimator null at n_post=50
    axA.axhline(NULL, color=C["purple"], ls="-", lw=1.2,
                label="Calibrated null (0.882)")
    axA.axhline(0.90, color=C["grey"], ls=":", lw=0.9, label="Nominal 0.90")
    axA.bar(x - w / 2, clean, w, color=C["blue"], label="Clean")
    axA.bar(x + w / 2, contam, w, color=C["vermil"], label="BP systematic (sev 0.05)")
    axA.set_xticks(x)
    axA.set_xticklabels(labels)
    axA.set_ylabel("Mean SBC 90% coverage")
    axA.set_ylim(0.0, 1.0)
    axA.set_title("Amortized NPE posterior, 4 flows")
    leg_above(axA, ncol=2, fs=7.6, y=1.19)

    # ---- panel B: diagnostic hierarchy (AUC of each external check) ----
    diag = [
        (r"Per-object $\Delta\chi^2$", g2["AUC_dchi2_perobj (genuine vs spurious)"], C["orange"]),
        ("Posterior-predictive check", ppc["AUC_PPC_clean_vs_contaminated"], C["sky"]),
        ("Evidence gate", g2["AUC_evidence_gate (genuine vs spurious)"], C["green"]),
    ]
    names = [d[0] for d in diag]
    vals = [d[1] for d in diag]
    cols = [d[2] for d in diag]
    yb = np.arange(len(names))[::-1]
    axB.axvline(0.5, color=C["grey"], ls=":", lw=1.0, label="Blind (0.5)")
    axB.barh(yb, vals, color=cols, height=0.55)
    for yi, v in zip(yb, vals):
        axB.text(v + 0.015, yi, f"{v:.2f}", va="center", ha="left", fontsize=9)
    axB.set_yticks(yb)
    axB.set_yticklabels(names)
    axB.set_xlim(0.0, 1.18)
    axB.set_ylim(-0.6, len(names) - 0.4)
    axB.set_xlabel("Detection AUC (contaminated vs clean)")
    axB.set_title("Diagnostic hierarchy")
    leg_above(axB, ncol=1, y=1.19)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "fig4_npe_sbc_calibration")


# =======================================================================
# Fig 5 - SBC coverage collapse on the REAL-spectra forward model
# =======================================================================
def fig5():
    real = load(os.path.join(P0, "npe_sbc_realmodel.json"))
    rob = load(os.path.join(P0, "npe_sbc_robust.json"))

    NULL = real["discrete_rank_null_cov90"]  # 0.882 discrete-rank null

    # ---- panel A inputs: per-seed/flow coverage, both models -----------
    proxy_clean = np.array(rob["clean_cov90_across"])     # 4 flows
    proxy_contam = np.array(rob["contam_cov90_across"])
    real_clean = np.array(real["clean_cov90_across"])     # 3 seeds
    real_contam = np.array(real["contam_cov90_across"])

    # ---- panel B inputs: per-parameter coverage, real model -----------
    params = real["runs"][0]["config"]["params"]
    plabels = ["msPC1", "msPC2", "wdPC1", "wdPC2", "wdPC3", r"log$_{10}$frac"]
    clean_pp = np.array([r["SBC_clean"]["coverage90_per_param"]
                         for r in real["runs"]])                  # (3, 6)
    contam_pp = np.array([r["SBC_under_BP_systematic_sev0.05"]["coverage90_per_param"]
                          for r in real["runs"]])                 # (3, 6)
    clean_mean, clean_std = clean_pp.mean(0), clean_pp.std(0)
    contam_mean, contam_std = contam_pp.mean(0), contam_pp.std(0)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 4.2))

    # ---- panel A: mean coverage clean vs contam, both models ----------
    groups = ["Blackbody proxy", "Real templates"]
    x = np.arange(len(groups))
    w = 0.36

    axA.axhline(NULL, color=C["purple"], ls="-", lw=1.2,
                label="Discrete-rank null (0.882)")
    axA.axhline(0.90, color=C["grey"], ls=":", lw=0.9, label="Nominal 0.90")

    clean_means = [proxy_clean.mean(), real_clean.mean()]
    contam_means = [proxy_contam.mean(), real_contam.mean()]
    axA.bar(x - w / 2, clean_means, w, color=C["blue"], label="Clean",
            zorder=2)
    axA.bar(x + w / 2, contam_means, w, color=C["vermil"],
            label="BP systematic (sev 0.05)", zorder=2)

    # per-seed / per-flow scatter on top of the bars
    clean_pts = [proxy_clean, real_clean]
    contam_pts = [proxy_contam, real_contam]
    for xi, pts in zip(x - w / 2, clean_pts):
        axA.plot(np.full_like(pts, xi), pts, "o", color="black",
                 ms=4, mfc="white", mew=1.0, zorder=4)
    for xi, pts in zip(x + w / 2, contam_pts):
        axA.plot(np.full_like(pts, xi), pts, "o", color="black",
                 ms=4, mfc="white", mew=1.0, zorder=4)

    # value annotations above each bar
    for xi, v in zip(x - w / 2, clean_means):
        axA.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + w / 2, contam_means):
        axA.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    axA.set_xticks(x)
    axA.set_xticklabels(groups)
    axA.set_ylabel("Mean SBC 90% Coverage")
    axA.set_ylim(0.0, 1.0)
    axA.set_title("Coverage Collapse Under The BP Systematic")
    leg_above(axA, ncol=2, fs=7.6, y=1.19)

    # ---- panel B: per-parameter coverage, real model ------------------
    xb = np.arange(len(plabels))
    wb = 0.38
    axB.axhline(NULL, color=C["purple"], ls="-", lw=1.2,
                label="Discrete-rank null (0.882)")
    axB.bar(xb - wb / 2, clean_mean, wb, yerr=clean_std, color=C["blue"],
            label="Clean", capsize=2.5, ecolor=C["grey"], zorder=2)
    axB.bar(xb + wb / 2, contam_mean, wb, yerr=contam_std, color=C["vermil"],
            label="BP systematic (sev 0.05)", capsize=2.5, ecolor=C["grey"],
            zorder=2)
    axB.set_xticks(xb)
    axB.set_xticklabels(plabels, rotation=30, ha="right")
    axB.set_ylabel("SBC 90% Coverage (Mean Of 3 Seeds)")
    axB.set_ylim(0.0, 1.0)
    axB.set_title("Per-Parameter Signature, Real-Template Model")
    leg_above(axB, ncol=2, fs=7.6, y=1.19)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "fig5_realmodel_sbc")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    print("done.")
