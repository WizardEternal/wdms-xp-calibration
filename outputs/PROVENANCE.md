# Provenance: figures and load-bearing numbers to their committed record

Every figure and every load-bearing number in the note maps to one committed JSON
record and the script that generated it. The JSONs under `outputs/` are the record;
the long jobs are not meant to be rerun to reproduce them (the seeds and configs are
fixed inside each script). Paths below are relative to the repo root.

| Note item | Committed JSON | Generating script |
|---|---|---|
| Turn-on curve (spurious rate vs local excess, injection leg) | `outputs/p0/g1_honest_multiseed.json` | `scripts/g1_honest_multiseed.py` |
| Coverage turn-on curve (NPE 90% coverage vs local excess) | `outputs/p0/sbc_honest.json` | `scripts/sbc_honest.py` |
| Section-3 local-axis AUCs | `outputs/p0/local_axis_auc.json` | `scripts/run_local_axis_auc.py` |
| Matched-damage specificity (log10_frac 0.07, matched mean cov 0.68, red msPC1 0.47) | `outputs/audit/adv_magmatch_multiseed.json` | `scripts/adv_magmatch_multiseed.py` |
| Clean coverage across seeds | `outputs/audit/clean_coverage_seeds.json` | `scripts/adv_magmatch_multiseed.py` |
| UV distance-matched control (off- vs on-sequence FUV detection) | `outputs/audit/uv_distance_matched.json` | `scripts/adv_survivor3b.py` |
| 27x severity-to-local statistic | `outputs/audit/severity_to_local.json` | `scripts/severity_to_local.py` |
| Catalog coverage fractions | `outputs/audit/catalog_fractions.json` | `scripts/catalog_fractions.py` |
| Conformal purity floor | `outputs/audit/conformal_certificate.json` | `scripts/conformal_certificate.py` |

## Convention warnings

Two records use a convention that differs from what the paper plots. Both are kept
on purpose, so read the number in the right context.

(a) The `spurious` column inside `outputs/p0/local_axis_auc.json` is computed with a
clean-control threshold. It is not the same quantity as the production turn-on curve
in `outputs/p0/g1_honest_multiseed.json`, which uses a per-seed percentile threshold
(95th percentile of the clean dchi2). The turn-on curve the paper plots is the
`g1_honest_multiseed.json` one; the `local_axis_auc.json` spurious column is the AUC
run's own control and will read a bit differently at the same amplitude.

(b) The `sev0.05 -> 0.433` coverage point in `outputs/p0/npe_sbc_realmodel.json` is the
superseded ADDITIVE-injection convention (severity as a fraction of total flux added as
a blue bump), kept for history. The multiplicative-local coverage curve the paper quotes
lives in `outputs/p0/sbc_honest.json`, where the injection is x*(1+a*shape01) so the local
excess is at most a. Do not read the additive 0.433 as the paper's coverage-at-5% number.
