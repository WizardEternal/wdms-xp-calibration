"""Severity sweep for the realistic-spectra G1 (real Gaia XP templates). Reports the
LOO (defensible) block per severity. Run after the larger template library lands:
  python scripts/g1_realistic_sweep.py --data data/real_spectra_large
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from g1_realistic import run, OUT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real_spectra_large")
    ap.add_argument("--snr", type=float, default=30.0)
    args = ap.parse_args()
    ddir = ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)

    sevs = [0.005, 0.01, 0.015, 0.02, 0.05, 0.10, 0.20]
    table = []
    for s in sevs:
        r = run(ddir, severity=s, snr=args.snr, n=300, seed=0)
        if r is None or r.get("status") != "OK":
            print(f"severity {s}: NO DATA / not OK - aborting sweep"); return
        loo = r["LOO"]
        table.append({"severity": s, "template_blue_residual": r["template_blue_residual_median"],
                      **{k: loo[k] for k in ("spurious_rate", "baseline_fpr_clean",
                         "redchi2_binary_genuine", "redchi2_binary_spurious",
                         "gof_AUC_genuine_vs_spurious", "auc_gate_honest", "auc_gate_oracle",
                         "auc_dchi2_perobj")}})
    (OUT / "g1_realistic_sweep.json").write_text(json.dumps(table, indent=2))
    print("\n=== SWEEP (LOO block) ===")
    print(json.dumps(table, indent=2))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sv = [t["severity"] for t in table]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(sv, [t["spurious_rate"] for t in table], "o-", color="crimson", label="spurious rate")
        ax.plot(sv, [t["auc_gate_honest"] for t in table], "s-", color="seagreen", label="gate AUC (honest)")
        ax.plot(sv, [t["auc_dchi2_perobj"] for t in table], "^-", color="navy", label="Delta-chi2 per-object AUC")
        ax.plot(sv, [t["gof_AUC_genuine_vs_spurious"] for t in table], "d-", color="darkorange", label="per-object GoF AUC")
        ax.axhline(0.5, color="grey", ls=":"); ax.set_ylim(0, 1.05)
        ax.set(xlabel="injected BP-excess severity", ylabel="rate / AUC (LOO)",
               title="Realistic-spectra G1 (real Gaia XP, leave-one-out)")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / "g1_realistic_sweep.png", dpi=150)
        print("figure ->", OUT / "g1_realistic_sweep.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
