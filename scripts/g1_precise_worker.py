"""Run ONE g1_realistic config and persist a single part-file. Crash-isolated.
Usage:
  python scripts/g1_precise_worker.py --kind boot --seed 0 --maxlib 120 --severity 0.02 --snr 30 --n 300
  python scripts/g1_precise_worker.py --kind snr  --seed 0 --maxlib 120 --severity 0.02 --snr 50 --n 300
Writes outputs/p0/_precise_parts/<tag>.json with the LOO+NONLOO blocks and config.
"""
import argparse, json, sys, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1_realistic import run, ROOT  # noqa: E402

PARTS = ROOT / "outputs" / "p0" / "_precise_parts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True)          # boot | snr | compare
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--maxlib", type=int, default=120)
    ap.add_argument("--severity", type=float, default=0.02)
    ap.add_argument("--snr", type=float, default=30.0)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    PARTS.mkdir(parents=True, exist_ok=True)
    tag = f"{a.kind}_seed{a.seed}_lib{a.maxlib}_snr{int(a.snr)}_sev{a.severity}_n{a.n}"
    out = PARTS / f"{tag}.json"
    rec = {"tag": tag, "kind": a.kind, "seed": a.seed, "maxlib": a.maxlib,
           "severity": a.severity, "snr": a.snr, "n": a.n}
    t0 = time.time()
    try:
        r = run(ROOT / "data/real_spectra_large", severity=a.severity, snr=a.snr,
                n=a.n, seed=a.seed, loo=True, maxlib=a.maxlib)
        if r is None:
            rec["status"] = "NO_DATA"
        else:
            rec["status"] = "OK"
            rec["LOO"] = r["LOO"]
            rec["NONLOO"] = r["NONLOO"]
            rec["n_ms_templates_used"] = r["n_ms_templates_used"]
            rec["n_wd_templates_used"] = r["n_wd_templates_used"]
            rec["template_blue_residual_median"] = r["template_blue_residual_median"]
    except Exception:
        rec["status"] = "ERROR"
        rec["traceback"] = traceback.format_exc()
    rec["elapsed_sec"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(rec, indent=2))
    print(f"WROTE {out.name}  status={rec['status']}  elapsed={rec['elapsed_sec']}s")


if __name__ == "__main__":
    main()
