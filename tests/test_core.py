"""Core unit tests - reproducibility/correctness for the overnight results.
Run: pytest -q   (from 04-code/). No network.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sbiwdms import audit  # noqa: E402
import p0_gonogo as p0  # noqa: E402
import conformal_certificate as cc  # noqa: E402


# --- conformal selection (Jin & Candes) ---
def test_conformal_pvalues_null_uniform():
    rng = np.random.default_rng(1)
    cal = rng.normal(size=2000)            # calibration negatives
    test = rng.normal(size=5000)           # test points from the SAME null dist
    p = audit.conformal_pvalues(cal, test)
    assert p.min() >= 0 and p.max() <= 1
    # null p-values are ~uniform => mean ~0.5
    assert abs(p.mean() - 0.5) < 0.03


def test_conformal_pvalues_monotone():
    cal = np.linspace(0, 1, 100)
    # higher score => smaller (more significant) one-sided p-value
    p_lo = audit.conformal_pvalues(cal, np.array([0.1]))[0]
    p_hi = audit.conformal_pvalues(cal, np.array([0.9]))[0]
    assert p_hi < p_lo


def test_bh_select_controls_fdr():
    rng = np.random.default_rng(2)
    # 200 nulls (p~U), 50 strong signals (p~0)
    p = np.concatenate([rng.uniform(size=200), rng.uniform(0, 0.001, size=50)])
    truth_null = np.r_[np.ones(200, bool), np.zeros(50, bool)]
    sel = audit.bh_select(p, fdr=0.1)
    assert len(sel) >= 50                       # recovers the signals
    fdp = truth_null[sel].mean()
    assert fdp <= 0.15                          # empirical FDP near/below target


# --- forward model (proxy) ---
def test_planck_normalized_and_bluer_when_hotter():
    b_cool = p0.planck(p0.WAVE, 4000.0)
    b_hot = p0.planck(p0.WAVE, 20000.0)
    assert np.all(b_cool > 0)
    assert abs(b_cool.mean() - 1.0) < 1e-9      # normalized to unit mean
    # bluer/redder slope: hot has relatively more blue flux
    blue = p0.WAVE < 500
    red = p0.WAVE > 900
    assert (b_hot[blue].mean() / b_hot[red].mean()) > (b_cool[blue].mean() / b_cool[red].mean())


def test_noise_is_per_pixel_independent():
    # regression test for the fixed bug (noise was one value broadcast across pixels)
    x, sig, _ = p0.make_single_ms(50, snr=20.0)
    resid_var = np.var(x - p0.planck(p0.WAVE, 5000.0) * 0, axis=1)  # not the point; check spread
    # within a spectrum, pixels must differ (independent noise) -> high per-row std
    per_row_std = x.std(axis=1)
    assert np.all(per_row_std > 0)
    # two spectra should not share an identical constant offset pattern
    assert not np.allclose(x[0] - x[0].mean(), x[1] - x[1].mean())


def test_riello_template_is_blue_only():
    # BP systematic: positive in the blue, ~zero in RP (the discriminating physics)
    assert p0.RIELLO[p0.WAVE < 500].mean() > p0.RIELLO[p0.WAVE > 800].mean()
    assert np.allclose(p0.RIELLO[p0.WAVE > 700], 0.0)
    # gate template differs from injected template (mismatch, not oracle)
    assert not np.allclose(p0.RIELLO, p0.RIELLO_GATE)


# --- stats helper ---
def test_wilson_interval_brackets_and_orders():
    lo, hi = cc.wilson(50, 100)
    assert 0 < lo < 0.5 < hi < 1
    lo2, hi2 = cc.wilson(50, 1000)               # tighter CI with more data
    assert (hi2 - lo2) < (hi - lo)
