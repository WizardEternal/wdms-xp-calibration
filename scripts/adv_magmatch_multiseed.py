"""SEED-ROBUSTNESS of the matched-damage specificity control (adv_magmatch.py).

adv_magmatch.py tuned each arm's amplitude at seed 0 so all four arms hit the SAME
mean cov90 (~0.68), then compared per-param SHAPE. The load-bearing pattern (at matched
mean coverage) is:
    bp     -> log10_frac collapses (0.09), msPC1 spared (0.81)   [specificity]
    noise  -> ~uniform across params (~0.6-0.73)                 [generic fragility flat]
    red    -> intermediate log10_frac (0.533), kills msPC1 (0.22)[location matters]

This script asks: does that pattern HOLD across seeds? We REUSE the committed tuned
amplitudes (bp 0.1066 / red 0.0234 / flat 0.0122 / noise_mult 1.6709 -- the DESIGN, do
NOT re-tune) and re-run all arms + clean at seeds 1 and 2, matching seed-0's config
(n_train=8000, n_sbc=300, n_post=50). One flow trained per seed, reused across arms;
fixed paired SBC draw set. Crash-safe: each seed persisted the moment it finishes,
skip-if-exists resume, per-seed try/except.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np, torch
warnings.filterwarnings("ignore")

ROOT = Path(r"C:/Users/Karan/Documents/Python stuff/cool papers/wdms-xp-calibration")
sys.path.insert(0, str(ROOT / "scripts"))
import npe_sbc_realmodel as m

WAVE = m.WAVE
BLUE = np.where(WAVE <= 680.0, np.cos(0.5*np.pi*np.clip((WAVE-392.0)/(680.0-392.0),0,1))**2, 0.0)
RED  = np.where(WAVE >= 680.0, np.cos(0.5*np.pi*np.clip((992.0-WAVE)/(992.0-680.0),0,1))**2, 0.0)
NPAR = len(m.NAMES)

# committed tuned amplitudes from adv_magmatch.json (seed 0) -- REUSED, NOT re-tuned.
TUNED = {"bp": 0.1066, "red": 0.0234, "flat": 0.0122, "noise": 1.6709}

OUTDIR = ROOT / "outputs" / "audit"
PARTDIR = OUTDIR / "adv_magmatch_multiseed_parts"
PARTDIR.mkdir(parents=True, exist_ok=True)

# committed seed-0 block (verbatim from adv_magmatch.json)
SEED0 = {
    "clean_per_param": [0.88, 0.907, 0.873, 0.887, 0.85, 0.837],
    "tuned_amps": {"bp": 0.1066, "red": 0.0234, "flat": 0.0122, "noise_mult": 1.6709},
    "matched_mean_cov": {"bp": 0.679, "red": 0.692, "flat": 0.664, "noise": 0.692},
    "bp_matched_per_param":    [0.81, 0.843, 0.683, 0.843, 0.863, 0.09],
    "red_matched_per_param":   [0.22, 0.88, 0.707, 0.867, 0.847, 0.533],
    "flat_matched_per_param":  [0.577, 0.81, 0.613, 0.873, 0.857, 0.273],
    "noise_matched_per_param": [0.607, 0.687, 0.69, 0.673, 0.65, 0.73],
    "source": "committed adv_magmatch.json",
}


def simulate_arm(theta, M, rng, kind, amp):
    th = theta.detach().cpu().numpy()
    s_ms = th[:, 0:m.K_MS]; s_wd = th[:, m.K_MS:m.K_MS+m.K_WD]; logf = th[:, m.K_MS+m.K_WD]
    ms = M["mu_ms"][None] + s_ms @ M["C_ms"]; wd = M["mu_wd"][None] + s_wd @ M["C_wd"]
    frac = 10.0**logf; ratio = ms.sum(1)/wd.sum(1)
    x0 = ms + (frac*ratio)[:, None]*wd
    nm = 1.0
    if kind == "bp":      x0 = x0 * (1.0 + amp*BLUE[None])
    elif kind == "red":   x0 = x0 * (1.0 + amp*RED[None])
    elif kind == "flat":  x0 = x0 * (1.0 + amp)
    elif kind == "noise": nm = amp
    sig = x0.mean(1, keepdims=True)/m.SNR
    return torch.as_tensor(x0 + rng.standard_normal(x0.shape)*sig*nm, dtype=torch.float32)


def run_seed(seed, n_train=8000, n_sbc=300, n_post=50):
    from sbi.inference import NPE
    from sbi.utils import BoxUniform
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    M, lo, hi = m.build_model()
    prior = BoxUniform(low=torch.as_tensor(lo), high=torch.as_tensor(hi))
    theta = prior.sample((n_train,)); x = m.simulate(theta, M, rng)
    print(f"training flow seed={seed} n_train={n_train} n_sbc={n_sbc} ...", flush=True)
    inf = NPE(prior=prior, density_estimator="nsf")
    inf.append_simulations(theta, x).train(show_train_summary=False)
    post = inf.build_posterior()

    th_eval = prior.sample((n_sbc,))
    th_np = th_eval.numpy()

    def cov_block(kind, amp):
        xo = simulate_arm(th_eval, M, rng, kind, amp)
        ranks = np.zeros((n_sbc, NPAR), int)
        for i in range(n_sbc):
            s = post.sample((n_post,), x=xo[i], reject_outside_prior=False,
                            show_progress_bars=False).detach().cpu().numpy()
            ranks[i] = (s < th_np[i][None]).sum(0)
        pit = ranks/n_post
        cov = [float(np.mean((pit[:, j] >= 0.05) & (pit[:, j] <= 0.95))) for j in range(NPAR)]
        return np.array(cov)

    res = {"config": {"seed": seed, "n_train": n_train, "n_sbc": n_sbc, "n_post": n_post,
                      "params": m.NAMES, "null_cov90": 0.882,
                      "reused_tuned_amps": TUNED, "note": "amps NOT re-tuned; reused from seed-0 design"},
           "clean_per_param": [round(v, 3) for v in cov_block("clean", 0.0)]}
    mean_cov = {}
    for kind in ["bp", "red", "flat", "noise"]:
        cov = cov_block(kind, TUNED[kind])
        res[f"{kind}_matched_per_param"] = [round(v, 3) for v in cov]
        mean_cov[kind] = round(float(cov.mean()), 3)
    res["matched_mean_cov"] = mean_cov

    print(f"\n=== seed {seed} MATCHED per-param (msPC1,msPC2,wdPC1,wdPC2,wdPC3,log10_frac) ===")
    print("  clean ", res["clean_per_param"])
    for k in ["bp", "red", "flat", "noise"]:
        print(f"  {k:6s}", res[f"{k}_matched_per_param"], f"(mean={mean_cov[k]})")
    print(f"  --> log10_frac: bp={res['bp_matched_per_param'][5]} "
          f"noise={res['noise_matched_per_param'][5]} red={res['red_matched_per_param'][5]}", flush=True)
    return res


def build_merged(seeds_data):
    """seeds_data: dict seed(str)->block. Compute per-arm mean per-param over available seeds."""
    arms = ["clean_per_param", "bp_matched_per_param", "red_matched_per_param",
            "flat_matched_per_param", "noise_matched_per_param"]
    merged = {
        "description": "Seed-robustness of the matched-damage specificity control. "
                       "Seed 0 = committed adv_magmatch.json; seeds 1,2 = rerun REUSING seed-0 "
                       "tuned amps (bp 0.1066/red 0.0234/flat 0.0122/noise_mult 1.6709). "
                       "n_train=8000 n_sbc=300 n_post=50 all seeds.",
        "params": m.NAMES,
        "reused_tuned_amps": TUNED,
        "per_seed": seeds_data,
        "mean_over_seeds": {},
        "log10_frac_by_seed": {},
        "msPC1_by_seed": {},
        "matched_mean_cov_by_seed": {},
    }
    avail = sorted(seeds_data.keys(), key=int)
    for arm in arms:
        stack = np.array([seeds_data[s][arm] for s in avail if arm in seeds_data[s]])
        merged["mean_over_seeds"][arm] = [round(float(v), 3) for v in stack.mean(0)]
    # log10_frac (param 5) and msPC1 (param 0) per arm per seed
    li = m.NAMES.index("log10_frac"); mi = m.NAMES.index("msPC1")
    for arm_key, label in [("bp_matched_per_param", "bp"), ("red_matched_per_param", "red"),
                           ("flat_matched_per_param", "flat"), ("noise_matched_per_param", "noise"),
                           ("clean_per_param", "clean")]:
        merged["log10_frac_by_seed"][label] = {s: seeds_data[s][arm_key][li] for s in avail if arm_key in seeds_data[s]}
        merged["msPC1_by_seed"][label] = {s: seeds_data[s][arm_key][mi] for s in avail if arm_key in seeds_data[s]}
    for s in avail:
        if "matched_mean_cov" in seeds_data[s]:
            merged["matched_mean_cov_by_seed"][s] = seeds_data[s]["matched_mean_cov"]
    return merged


def build_clean_seeds(rerun_clean):
    """Consolidate clean-arm mean cov90 per seed from committed artifacts + rerun clean arms.
    Committed clean means:
      seed 0: sbc_honest.json clean.mean_coverage90 = 0.879
      seeds 1-3: sbc_specificity_seeds.json 'clean' per-param -> mean
    rerun_clean: dict seed(str) -> clean_per_param list (this rerun, n_sbc=300 matched-control clean arm)
    """
    entries = {}
    # committed seed 0 from sbc_honest.json
    entries["sbc_honest_seed0"] = {"seed": 0, "mean_cov90": 0.879,
                                   "source": "sbc_honest.json clean.mean_coverage90"}
    # committed seeds 1-3 from sbc_specificity_seeds.json
    spec = {
        "1": [0.88, 0.86, 0.847, 0.873, 0.867, 0.893],
        "2": [0.84, 0.86, 0.847, 0.913, 0.847, 0.927],
        "3": [0.893, 0.92, 0.887, 0.907, 0.927, 0.893],
    }
    for s, pp in spec.items():
        entries[f"sbc_specificity_seed{s}"] = {"seed": int(s),
                                               "mean_cov90": round(float(np.mean(pp)), 3),
                                               "source": "sbc_specificity_seeds.json clean per-param mean"}
    # rerun clean arms (matched-control clean, this task)
    for s, pp in sorted(rerun_clean.items(), key=lambda kv: int(kv[0])):
        entries[f"adv_magmatch_multiseed_seed{s}"] = {"seed": int(s),
                                                      "mean_cov90": round(float(np.mean(pp)), 3),
                                                      "source": "adv_magmatch_multiseed rerun clean arm (n_sbc=300)"}
    vals = [e["mean_cov90"] for e in entries.values()]
    return {
        "description": "Clean-arm mean cov90 per seed, consolidated across committed SBC artifacts "
                       "and the matched-damage rerun clean arms. Null (discrete-rank) cov90 = 0.882.",
        "null_cov90": 0.882,
        "entries": entries,
        "clean_mean_cov90_values": [round(v, 3) for v in vals],
        "min": round(float(np.min(vals)), 3),
        "max": round(float(np.max(vals)), 3),
        "mean": round(float(np.mean(vals)), 3),
        "n": len(vals),
    }


if __name__ == "__main__":
    seeds_to_run = [1, 2]
    seeds_data = {"0": {**SEED0, "config": {"seed": 0, "source": "committed adv_magmatch.json",
                                            "n_train": 8000, "n_sbc": 300, "n_post": 50}}}
    # add matched_mean_cov for seed 0
    seeds_data["0"]["matched_mean_cov"] = SEED0["matched_mean_cov"]

    for seed in seeds_to_run:
        pf = PARTDIR / f"seed{seed}.json"
        if pf.exists():
            print(f"[resume] {pf.name} exists -> skip", flush=True)
            seeds_data[str(seed)] = json.loads(pf.read_text())
            continue
        try:
            r = run_seed(seed)
            pf.write_text(json.dumps(r, indent=2))
            seeds_data[str(seed)] = r
            print(f"[persisted seed {seed}] -> {pf}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[seed {seed}] FAILED: {e} -- continuing", flush=True)
            continue
        # rewrite merged after each seed (rolling)
        merged = build_merged(seeds_data)
        (OUTDIR / "adv_magmatch_multiseed.json").write_text(json.dumps(merged, indent=2))
        print(f"[rolling merged written, {len(seeds_data)} seeds]", flush=True)

    merged = build_merged(seeds_data)
    (OUTDIR / "adv_magmatch_multiseed.json").write_text(json.dumps(merged, indent=2))

    rerun_clean = {s: seeds_data[s]["clean_per_param"] for s in seeds_data
                   if s != "0" and "clean_per_param" in seeds_data[s]}
    clean_seeds = build_clean_seeds(rerun_clean)
    (OUTDIR / "clean_coverage_seeds.json").write_text(json.dumps(clean_seeds, indent=2))

    print("\n==== FINAL ====", flush=True)
    print("adv_magmatch_multiseed.json log10_frac_by_seed:", json.dumps(merged["log10_frac_by_seed"]), flush=True)
    print("adv_magmatch_multiseed.json msPC1_by_seed:", json.dumps(merged["msPC1_by_seed"]), flush=True)
    print("clean_coverage_seeds min/max/mean:", clean_seeds["min"], clean_seeds["max"], clean_seeds["mean"], flush=True)
    print("DONE", flush=True)
