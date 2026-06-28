"""Aggregate the per-task part files into the two headline JSONs.
Re-runnable / incremental: rebuilds from whatever parts currently exist on disk.
  outputs/p0/g1_precise_bootstrap.json   (20-seed bootstrap, maxlib=120)
  outputs/p0/g1_precise_snr.json         (snr sweep, maxlib=120)
  outputs/p0/g1_precise_compare.json     (maxlib 50 vs 120 at nominal)
Also embeds the maxlib=50 single-config comparison rows if present.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "outputs" / "p0" / "_precise_parts"
OUT = ROOT / "outputs" / "p0"

METRICS = ["spurious_rate", "baseline_fpr_clean", "gof_AUC_genuine_vs_spurious",
           "auc_gate_honest", "auc_gate_oracle", "auc_dchi2_perobj",
           "redchi2_binary_genuine", "redchi2_binary_spurious"]


def load_parts():
    parts = []
    for p in sorted(PARTS.glob("*.json")):
        try:
            parts.append(json.loads(p.read_text()))
        except Exception:
            pass
    return parts


def summ(vals):
    v = np.array([x for x in vals if x is not None], float)
    if v.size == 0:
        return None
    return {"mean": round(float(v.mean()), 4), "std": round(float(v.std(ddof=1)) if v.size > 1 else 0.0, 4),
            "min": round(float(v.min()), 4), "max": round(float(v.max()), 4),
            "n": int(v.size), "values": [round(float(x), 4) for x in v]}


def build_bootstrap(parts):
    rows = [p for p in parts if p["kind"] == "boot" and p.get("status") == "OK" and p["maxlib"] == 120]
    rows = sorted(rows, key=lambda r: r["seed"])
    out = {"_meta": {"maxlib": 120, "severity": 0.02, "snr": 30, "n": 300,
                     "n_seeds": len(rows), "seeds": [r["seed"] for r in rows],
                     "library": "full 120 MS + 120 WD real Gaia XP templates",
                     "note": "20-seed bootstrap; std is ddof=1 sample std across seeds"}}
    for m in METRICS:
        out[m] = summ([r["LOO"].get(m) for r in rows])
    # errored/missing seeds
    errs = [p for p in parts if p["kind"] == "boot" and p.get("status") != "OK"]
    out["_meta"]["errored_or_missing"] = [{"seed": e.get("seed"), "status": e.get("status")} for e in errs]
    OUT.joinpath("g1_precise_bootstrap.json").write_text(json.dumps(out, indent=2))
    return out


def build_snr(parts):
    rows = [p for p in parts if p["kind"] == "snr" and p.get("status") == "OK" and p["maxlib"] == 120]
    rows = sorted(rows, key=lambda r: r["snr"])
    out = []
    for r in rows:
        L = r["LOO"]
        out.append({"snr": r["snr"],
                    "spurious_rate": L.get("spurious_rate"),
                    "baseline_fpr_clean": L.get("baseline_fpr_clean"),
                    "redchi2_binary_genuine": L.get("redchi2_binary_genuine"),
                    "gof_AUC_genuine_vs_spurious": L.get("gof_AUC_genuine_vs_spurious"),
                    "auc_gate_honest": L.get("auc_gate_honest"),
                    "auc_dchi2_perobj": L.get("auc_dchi2_perobj")})
    payload = {"_meta": {"maxlib": 120, "severity": 0.02, "n": 300, "seed": 0}, "rows": out}
    OUT.joinpath("g1_precise_snr.json").write_text(json.dumps(payload, indent=2))
    return payload


def build_compare(parts):
    # nominal config = severity 0.02, snr 30, n 300, seed 0; compare lib 50 vs 120
    def grab(lib):
        cands = [p for p in parts if p.get("status") == "OK" and p["maxlib"] == lib
                 and p["seed"] == 0 and abs(p["snr"] - 30) < 1e-6 and abs(p["severity"] - 0.02) < 1e-9]
        return cands[0] if cands else None
    p50, p120 = grab(50), grab(120)
    out = {"_meta": {"config": "severity=0.02, snr=30, n=300, seed=0"}}
    for lib, p in [(50, p50), (120, p120)]:
        out[f"maxlib_{lib}"] = (
            {k: p["LOO"].get(k) for k in METRICS} | {"n_ms_used": p["n_ms_templates_used"],
             "n_wd_used": p["n_wd_templates_used"]} if p else None)
    if p50 and p120:
        out["delta_spurious_rate_120_minus_50"] = round(
            p120["LOO"]["spurious_rate"] - p50["LOO"]["spurious_rate"], 4)
    OUT.joinpath("g1_precise_compare.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    parts = load_parts()
    b = build_bootstrap(parts)
    s = build_snr(parts)
    c = build_compare(parts)
    print("=== BOOTSTRAP (maxlib=120) ===")
    print(json.dumps({k: b[k] for k in ["spurious_rate", "gof_AUC_genuine_vs_spurious",
                                        "auc_gate_honest", "auc_dchi2_perobj", "_meta"] if k in b}, indent=2))
    print("=== SNR SWEEP (maxlib=120) ===")
    print(json.dumps(s["rows"], indent=2))
    print("=== COMPARE lib50 vs lib120 ===")
    print(json.dumps(c, indent=2))


if __name__ == "__main__":
    main()
