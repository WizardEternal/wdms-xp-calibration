"""Real-spectra G1: SNR sweep (LOO) at fixed severity, to match the proxy SNR sweep.
  python scripts/g1_realistic_snr.py --data data/real_spectra_large
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from g1_realistic import run, OUT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real_spectra_large")
    ap.add_argument("--severity", type=float, default=0.05)
    args = ap.parse_args()
    ddir = ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
    table = []
    for snr in [10, 20, 30, 50]:
        r = run(ddir, severity=args.severity, snr=float(snr), n=300, seed=0)
        if not r or r.get("status") != "OK":
            print("NO DATA"); return
        loo = r["LOO"]
        table.append({"snr": snr, **{k: loo[k] for k in
                      ("spurious_rate", "redchi2_binary_genuine", "gof_AUC_genuine_vs_spurious",
                       "auc_gate_honest", "auc_dchi2_perobj")}})
    (OUT / "g1_realistic_snr.json").write_text(json.dumps(table, indent=2))
    print("\n=== REAL-SPECTRA SNR SWEEP (LOO, severity %.2f) ===" % args.severity)
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
