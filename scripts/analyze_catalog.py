"""Deep characterization of Li's published 30k GPC WD-MS catalog (stdlib only).

Descriptive audit using ONLY shipped columns (no external labels). Writes
outputs/audit/catalog_analysis.{json,md}. Interpretations flagged [CHECK] are
verified against the Li 2025 paper (Table 2 column defs).

Column notes (Li 2025 Table 2; the CSV has _1/_2 duplicate join cols that are equal):
  prob_binary       P_WDMS GPC score
  chi2_single_1     chi2 of single (MS or WD) fit        (3 params -> dof 58 over 61 px)
  chi2_binary_1     chi2 of WD-MS binary fit             (6 params -> dof 55)
  chi2_diff_renorm  (chi2_single - chi2_binary)/chi2_single ; >0 => binary better
  flag_MSMS         1 => classified as MS-MS (CONTAMINANT of a WD-MS catalog) [CHECK]
  flag_binary_fit   1 => poor binary fit ; flag_single_fit 1 => poor single fit
  flag_binary_mag_diff 1 => large mag diff between components
  flag_wdmsfit_in   1 => WD component lies in the WD sequence (good) [CHECK]
  q_binary          mass ratio of the MS-MS binary fit
"""
import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "wdms_gpc_30k_1229_mg7_msms.csv"
OUT = ROOT / "outputs" / "audit"


def fnum(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def fbool(x):
    return str(x).strip().lower() == "true"


def col(rows, name):
    return [r[name] for r in rows]


def fcol(rows, name):
    return [v for v in (fnum(r[name]) for r in rows) if v is not None]


def main():
    rows = list(csv.DictReader(open(CSV, newline="")))
    n = len(rows)
    OUT.mkdir(parents=True, exist_ok=True)

    flags = ["flag_MSMS", "flag_binary_fit", "flag_single_fit",
             "flag_binary_mag_diff", "flag_wdmsfit_in"]
    flag_counts = {f: sum(fbool(r[f]) for r in rows) for f in flags}

    p = fcol(rows, "prob_binary")
    cdr = fcol(rows, "chi2_diff_renorm")
    cb = fcol(rows, "chi2_binary_1")
    cs = fcol(rows, "chi2_single_1")
    DOF_B, DOF_S = 55, 58
    cb_red = [x / DOF_B for x in cb]

    msms = [fbool(r["flag_MSMS"]) for r in rows]
    p_msms = [fnum(r["prob_binary"]) for r in rows if fbool(r["flag_MSMS"])]
    p_clean = [fnum(r["prob_binary"]) for r in rows if not fbool(r["flag_MSMS"])]
    q_msms = [v for v in (fnum(r["q_binary"]) for r in rows if fbool(r["flag_MSMS"])) if v is not None]

    def frac(pred_list):
        return round(sum(1 for x in pred_list if x) / n, 4)

    summary = {
        "n_rows": n,
        "flag_true_counts": flag_counts,
        "flag_true_frac": {f: round(c / n, 4) for f, c in flag_counts.items()},
        "prob_binary": {
            "min": round(min(p), 4), "median": round(st.median(p), 4),
            "mean": round(st.mean(p), 4), "max": round(max(p), 4),
            "n_gt_0.9": sum(x > 0.9 for x in p), "n_gt_0.95": sum(x > 0.95 for x in p),
            "median_MSMS_flagged": round(st.median(p_msms), 4),
            "median_not_flagged": round(st.median(p_clean), 4),
        },
        "binary_fit_quality": {
            "chi2_binary_reduced_median": round(st.median(cb_red), 3),
            "frac_redchi2_gt_1": frac([x > 1 for x in cb_red]),
            "frac_redchi2_gt_2": frac([x > 2 for x in cb_red]),
            "frac_redchi2_gt_5": frac([x > 5 for x in cb_red]),
        },
        "chi2_diff_renorm": {
            "median": round(st.median(cdr), 4),
            "frac_lt_0_single_better": frac([x < 0 for x in cdr]),
            "min": round(min(cdr), 4),
        },
        "q_binary_MSMS": {
            "n": len(q_msms),
            "median": round(st.median(q_msms), 3) if q_msms else None,
        },
        "interpretation_CHECK": [
            "25%-level flag_MSMS => a quarter of the WD-MS catalog is internally flagged "
            "as probable MS-MS contamination [CHECK meaning of flag_MSMS=1 vs Table 2].",
            "frac chi2_diff_renorm<0 = systems where the SINGLE fit is better yet still "
            "GPC-selected as binary [CHECK: QC cut required chi2_diff_renorm>-1].",
            "All prob_binary>0.8 => shipped file is POST-cut; literal 0.8-replacement "
            "needs the pre-cut pool Li do not ship (audit purity within the 30k instead).",
        ],
    }
    (OUT / "catalog_analysis.json").write_text(json.dumps(summary, indent=2))

    md = ["# Li 30k catalog - descriptive audit (shipped columns only)\n",
          f"- rows: **{n}**\n", "\n## Flags (true)\n"]
    for f in flags:
        md.append(f"- `{f}` = true: **{flag_counts[f]}** ({flag_counts[f]/n:.1%})\n")
    md.append(f"\n## prob_binary\nmin {min(p):.3f} | median {st.median(p):.3f} | "
              f"max {max(p):.4f}; >0.9: {sum(x>0.9 for x in p)}, >0.95: {sum(x>0.95 for x in p)}\n")
    md.append(f"- median P among MS-MS-flagged: {st.median(p_msms):.3f} vs not-flagged: {st.median(p_clean):.3f}\n")
    md.append(f"\n## binary fit quality (reduced chi2 = chi2_binary/{DOF_B})\n"
              f"median {st.median(cb_red):.2f}; frac >1: {frac([x>1 for x in cb_red]):.1%}, "
              f">2: {frac([x>2 for x in cb_red]):.1%}, >5: {frac([x>5 for x in cb_red]):.1%}\n")
    md.append(f"\n## chi2_diff_renorm (>0 = binary better)\n"
              f"median {st.median(cdr):.3f}; frac<0 (single better, still selected): "
              f"{frac([x<0 for x in cdr]):.1%}; min {min(cdr):.3f}\n")
    (OUT / "catalog_analysis.md").write_text("".join(md))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
