# wdms-xp-calibration

This repo audits the Li et al. 2025 Gaia DR3 XP white-dwarf main-sequence (WD-MS) binary
catalog (~30,000 candidates, each with a Gaussian-process `prob_binary` but no likelihood
and no goodness-of-fit) with a simulation-based-inference calibration toolkit. It is the
optical sequel to `sbi-xray-calibration` (arXiv:2606.17098), the X-ray note where the same
toolkit caught a detector gain shift that a per-object trust score misses and the model
evidence catches.

The method injects the Gaia BP blue-flux residual (Riello et al. 2021) into real single
main-sequence XP spectra, fits single against binary, and asks which scores separate the
spurious candidates from genuine binaries. It also trains an amortized neural posterior on
the same forward model and checks the posterior calibration with SBC, and it cross-matches
the catalog against SDSS, LAMOST, and GALEX.

The audited catalog is on Zenodo (doi:10.5281/zenodo.14411002); the shipped CSV here is
byte-identical to it (MD5 66cb2ae49124096e63876e46bf74bfdb). Everything runs from a fixed
seed (0), and the committed `outputs/**/*.json` records hold the machine-readable values.

## What it finds

The question is whether the Gaia BP blue-flux residual, at the level that survives Huang
et al. 2024's correction, contaminates the Li WD-MS selection. At the realistic amplitude it
does not.

The subtlety is what "the residual" means. Huang report a consistency better than 2% in
336-400 nm, which is a local, per-wavelength number. Injecting 2% of a star's total flux into
the blue band is a different thing. On a red MS star, faint in the blue, that is a ~55% local
excess, about 27x larger. At the honest 2%-of-local amplitude the spurious rate is 0.08
against a 0.05 baseline, and the NPE 90% coverage is 0.84 against a clean 0.88. The
single-versus-binary selection only starts to fail above about a 10-20% local excess, and it
reaches a ~0.96 spurious rate near a 50% local excess, which is the raw uncorrected systematic
that Huang's correction and Li's `B<18` cut already remove. So at the realistic
post-correction residual the BP systematic does essentially nothing to the selection, and the
contamination hypothesis is a null.

Two things hold up. The first is an amplitude-resolved map of where
this kind of single-versus-binary selection starts to fail. The second is an
off-cooling-sequence UV-deficiency indicator that survives a distance control on the
GALEX-matched subset (off-sequence sources show FUV detection 0.19 against 0.50 on-sequence,
and that gap holds after matching on distance). The committed `outputs/` records are the raw
runs behind these numbers.

## Prior work

The audited catalog and the two systematics it rests on:

- **Li et al. 2025**, ApJS 279, 47 ([arXiv:2501.14494](https://arxiv.org/abs/2501.14494)), the target. ~30k WD-MS candidates from Gaia XP, an amortized Gaussian-process classifier `prob_binary` with no goodness-of-fit gate, cut at 0.8. Zenodo doi:10.5281/zenodo.14411002.
- **Riello et al. 2021**, A&A 649, A3 ([arXiv:2012.01916](https://arxiv.org/abs/2012.01916)), the Gaia EDR3 photometric validation that documents the BP-band blue-flux excess for faint red sources; the systematic injected here.
- **Huang et al. 2024**, ApJS 271, 13 ([arXiv:2401.12006](https://arxiv.org/abs/2401.12006)), an empirical correction of the Gaia DR3 XP spectra. Its stated consistency after correction is better than 2% in 336-400 nm, a local per-wavelength residual (see What it finds above for why that differs from a fraction of total flux).

SBI calibration and misspecification (the toolkit ported from the X-ray note):

- **Akbari 2026** ([arXiv:2606.17098](https://arxiv.org/abs/2606.17098)), my X-ray note. The gain-shift silent-miscalibration result and the evidence-gate-vs-per-object-score framing this work ports to the optical domain.
- **Vincent et al. 2025** ([arXiv:2510.16261](https://arxiv.org/abs/2510.16261)), the closest prior art, NPE for white-dwarf spectroscopy with SBC. I cede "SBC on WD NPE" to it and differentiate on the XP injection, the catalog audit, and the model-comparison repair.
- **Talts et al. 2018** ([arXiv:1804.06788](https://arxiv.org/abs/1804.06788)), Simulation-Based Calibration, the rank-histogram test behind the calibration check.
- **Hermans et al.**, TMLR 2022 ([arXiv:2110.06581](https://arxiv.org/abs/2110.06581)), the SBI coverage-crisis paper.
- **Ward et al. 2022** (RNPE) ([arXiv:2210.06564](https://arxiv.org/abs/2210.06564)), **Cannon et al. 2022** ([arXiv:2209.01845](https://arxiv.org/abs/2209.01845)), **Anau Montel, Alvey & Weniger 2025**, PRD 111, 083013 ([arXiv:2412.15100](https://arxiv.org/abs/2412.15100)), the robust- / misspecification-aware NPE cluster.
- **sbi toolkit (Tejero-Cantero et al. 2020)**, JOSS 5(52) 2505, the NPE+NSF software, and **UltraNest (Buchner 2021)**, JOSS 6(60) 3001 ([arXiv:2101.09604](https://arxiv.org/abs/2101.09604)), the nested-sampling engine for the evidence gate.

Conformal selection (an honest negative):

- **Jin & Candes 2023**, selection by prediction with conformal p-values ([arXiv:2210.01408](https://arxiv.org/abs/2210.01408)) and the covariate-shift-weighted form ([arXiv:2307.09291](https://arxiv.org/abs/2307.09291)); **Tibshirani et al. 2019**, conformal prediction under covariate shift ([arXiv:1904.06019](https://arxiv.org/abs/1904.06019)).
- **Ashton 2024** ([arXiv:2402.19313](https://arxiv.org/abs/2402.19313)), conformal calibration of gravitational-wave search algorithms; the astrophysics precedent for conformal selection.

Competing Gaia DR3 WD-MS / WD catalogs (this work audits one of them):

- **Rebassa-Mansergas et al. 2025** ([arXiv:2505.15895](https://arxiv.org/abs/2505.15895)), a magnitude-limited unresolved WD-MS catalog from SED fitting.
- **Nayak 2025** ([arXiv:2509.06910](https://arxiv.org/abs/2509.06910)), unresolved WD-MS via UV excess.
- **Perez-Couto et al. 2025** ([arXiv:2503.04672](https://arxiv.org/abs/2503.04672)), hidden companions via an unsupervised self-organizing map.
- **Shahaf et al. 2024** ([arXiv:2309.15143](https://arxiv.org/abs/2309.15143)), the astrometric-orbit triage census.
- **Gentile Fusillo et al. 2021** ([arXiv:2106.07669](https://arxiv.org/abs/2106.07669)), the Gaia EDR3 white-dwarf catalog.
- **Garcia-Zamora et al. 2025** ([arXiv:2505.05560](https://arxiv.org/abs/2505.05560)), a random-forest spectral classification of the Gaia 500-pc sample.

External truth for the reliability audit: **Rebassa-Mansergas et al. 2016** (SDSS WD-MS, [arXiv:1603.01017](https://arxiv.org/abs/1603.01017)), **Ren et al. 2018** (LAMOST WD-MS, [arXiv:1803.09523](https://arxiv.org/abs/1803.09523)), and GALEX (**Bianchi et al. 2017** GUVcat, [arXiv:1704.05903](https://arxiv.org/abs/1704.05903)).

## What's in here

```
src/sbiwdms/
  audit.py        shared audit library: conformal p-values, BH/FDR selection,
                  Mondrian CMD binning, Wilson intervals (used by the catalog-audit scripts)
scripts/          self-contained analyses, each writes a record under outputs/
  analyze_catalog.py      descriptive audit of the published 30k
  bp_extension.py         the B>18 extension check (outputs/audit/bp_extension.json)
  p0_gonogo.py            G1 go/no-go: BP injection on the blackbody proxy
  p0_snr_sweep.py         G1 SNR robustness sweep
  g1_realistic.py         G1 on REAL Gaia XP spectra, leave-one-out validated
  g1_realistic_snr.py / g1_realistic_sweep.py      real-spectra SNR + severity sweeps
  g1_precise_worker.py / g1_precise_aggregate.py   crash-isolated 20-seed bootstrap for
                  the injection spurious-rate (outputs/p0/g1_precise_bootstrap.json)
  evidence_gate.py        nested-sampling (UltraNest) evidence gate vs the BIC proxy
  npe_sbc.py              amortized NPE + SBC, clean vs under the injection (proxy model)
  npe_sbc_realmodel.py / npe_sbc_realmodel_controls.py   NPE+SBC on the real-template PCA
                  forward model, with the 3-seed + emulator-fidelity controls
  ppc_diagnostic.py       posterior-predictive check
  build_groundtruth.py / analyze_groundtruth.py   SDSS+LAMOST+GALEX cross-match audit
  conformal_certificate.py / conformal_fdr.py     purity floor + conformal-FDR (negative)
  make_figures.py         regenerate figures from the output JSONs
tests/        test_core.py        (conformal null-uniform, BH FDR, planck, riello, Wilson CI)
outputs/      committed JSON records behind every number (audit/ and p0/)
```

The committed `outputs/**/*.json` records let you read the numbers without running anything.
Figures are not committed in this revision; they are regenerable
from the output records via `make_figures.py`. The 28 MB Li catalog, trained checkpoints,
pulled XP spectra, and per-seed dumps are gitignored (rebuildable from Zenodo + seed).

## Reproduce

```powershell
# from the repo root, with the venv python; cap compute as needed
$env:OMP_NUM_THREADS = 4
.venv\Scripts\python.exe -m pip install -e . --no-deps      # one-time editable install

# the audited catalog is NOT committed (28 MB): pull it from Zenodo into data/
#   doi:10.5281/zenodo.14411002  ->  data/wdms_gpc_30k_1229_mg7_msms.csv

# the catalog audit (stdlib + numpy; reads the Zenodo CSV)
.venv\Scripts\python.exe scripts\analyze_catalog.py
.venv\Scripts\python.exe scripts\bp_extension.py

# BP injection: blackbody proxy, then real Gaia XP (leave-one-out)
.venv\Scripts\python.exe scripts\p0_gonogo.py
.venv\Scripts\python.exe scripts\g1_realistic.py --data-dir data\real_spectra_large
.venv\Scripts\python.exe scripts\evidence_gate.py        # UltraNest evidence gate

# calibration: amortized NPE + SBC, clean vs under the injection
.venv\Scripts\python.exe scripts\npe_sbc.py
.venv\Scripts\python.exe scripts\npe_sbc_realmodel.py

# reliability audit (needs astroquery network for the cross-matches)
.venv\Scripts\python.exe scripts\build_groundtruth.py
.venv\Scripts\python.exe scripts\analyze_groundtruth.py
.venv\Scripts\python.exe scripts\conformal_certificate.py

# tests (no network, ~2s)
.venv\Scripts\python.exe -m pytest -q
```

The Li catalog and the pulled XP spectra live under `data/` and are gitignored (the catalog
from Zenodo, the XP spectra from the Gaia archive). `sbi` and `ultranest` are needed only for
the NPE+SBC and nested-sampling scripts; the catalog audit and the BIC gate run without them.
Python 3.12, native Windows.

## Author

Karan Akbari, MSc Astrophysics, St. Xavier's College Mumbai. Background in X-ray timing and
spectral analysis of black-hole X-ray binaries (GRS 1915+105, 4U 1630-47) with
Dr. Sudip Bhattacharyya at TIFR. This is the optical sequel to the X-ray SBI calibration
note (arXiv:2606.17098).
