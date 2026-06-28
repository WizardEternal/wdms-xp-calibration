"""G1 robustness: does the silent-systematic regime hold across data quality (SNR)?
Reuses the validated run() from p0_gonogo (no changes to that script). Fixed low,
realistic BP-excess severity (0.05); sweep SNR. Addresses the fixed-SNR=30 caveat.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from p0_gonogo import run, OUT  # noqa: E402


def main():
    snrs = [10, 20, 30, 50, 100]
    table = []
    for s in snrs:
        r = run(n=400, snr=float(s), severity=0.05, seed=0)
        table.append({
            "snr": s,
            "spurious_rate": r["A_spurious_rate_injected_singles"],
            "baseline_fpr": r["A_baseline_fpr_clean_singles"],
            "redchi2_binary_genuine": r["B_perobject_gof"]["redchi2_binary_genuine"],
            "redchi2_binary_spurious": r["B_perobject_gof"]["redchi2_binary_spurious"],
            "gof_indistinguishable": r["B_perobject_gof"]["indistinguishable_heuristic"],
            "auc_gate_honest": r["C_gate"]["auc_gate_HONEST_mismatched_template"],
            "auc_dchi2_perobj": r["C_gate"]["auc_dchi2_perobj_genuine_vs_spurious"],
        })
    (OUT / "snr_sweep.json").write_text(json.dumps(table, indent=2))
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
