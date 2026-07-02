"""Turn-on curve robustness: the injection-leg honest turn-on across 3 seeds, for error bars on
the salvage figure. Uses g1_realistic's production fit functions, real 120 templates, LOO. Honest
multiplicative-local injection x*(1+a*shape01). Writes g1_honest_multiseed.json incrementally."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(r"C:/Users/Karan/Documents/Python stuff/cool papers/wdms-xp-calibration")
sys.path.insert(0, str(ROOT / "scripts"))
import g1_realistic as g1
np = g1.np
OUT = ROOT / "outputs" / "p0" / "g1_honest_multiseed.json"
WAVE = g1.WAVE
SHAPE01 = np.where(WAVE <= 680.0, np.cos(0.5*np.pi*np.clip((WAVE-392.0)/(680.0-392.0),0,1))**2, 0.0)
SNR, N, MAXLIB = 30.0, 150, 50
AMPS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
SEEDS = [0, 1, 2]

MS0, _ = g1.load_library(ROOT/"data/real_spectra_large", "ms")
WD0, _ = g1.load_library(ROOT/"data/real_spectra_large", "wd")
print(f"templates: {MS0.shape[0]} MS, {WD0.shape[0]} WD", flush=True)

res = {"config": {"snr": SNR, "n": N, "maxlib": MAXLIB, "seeds": SEEDS, "amps_pct_local": AMPS},
       "baseline_fpr": {}, "spurious_by_amp": {f"{a}": [] for a in AMPS}}
for seed in SEEDS:
    g1.RNG = np.random.default_rng(seed)
    MS = MS0[g1.RNG.choice(MS0.shape[0], MAXLIB, replace=False)] if MS0.shape[0] > MAXLIB else MS0
    WD = WD0[g1.RNG.choice(WD0.shape[0], MAXLIB, replace=False)] if WD0.shape[0] > MAXLIB else WD0
    ALL = np.vstack([MS, WD])
    xs_s, sig_s, i_s, _ = g1.gen(MS, N, SNR)
    def dchi2(xs):
        cs = np.array([g1.fit_single(xs[k], sig_s[k], ALL, {int(i_s[k])}) for k in range(N)])
        cb = np.array([g1.fit_binary(xs[k], sig_s[k], MS, WD, int(i_s[k])) for k in range(N)])
        return (cs - cb) / cs
    d_s = dchi2(xs_s); thr = np.quantile(d_s, 0.95)
    res["baseline_fpr"][f"{seed}"] = round(float((d_s > thr).mean()), 3)
    print(f"seed {seed}: baseline={res['baseline_fpr'][f'{seed}']}", flush=True)
    for a in AMPS:
        r = round(float((dchi2(xs_s*(1.0+a*SHAPE01[None, :])) > thr).mean()), 3)
        res["spurious_by_amp"][f"{a}"].append(r)
        print(f"   a={a}: spurious={r}", flush=True)
        OUT.write_text(json.dumps(res, indent=2))
# summarize mean+/-std
res["summary"] = {f"{a}": {"mean": round(float(np.mean(res['spurious_by_amp'][f'{a}'])), 3),
                          "std": round(float(np.std(res['spurious_by_amp'][f'{a}'])), 3)} for a in AMPS}
print("summary:", json.dumps(res["summary"]), flush=True)
OUT.write_text(json.dumps(res, indent=2)); print("DONE", flush=True)
