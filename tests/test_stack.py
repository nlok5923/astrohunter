"""Tests for shift-and-stack / digital tracking (Phase 3)."""

import json

import numpy as np

from astrohunt.extract import extract_sources
from astrohunt.io import load_frame_set
from astrohunt.stack import digital_tracking, peak_snr_at, shift_and_stack
from astrohunt.synthetic import SyntheticConfig, generate


def _faint_set(tmp_path, flux=700, seed=4):
    cfg = SyntheticConfig(seed=seed, asteroid_flux=flux, star_flux_range=(2e3, 1.5e4))
    paths, truth_path = generate(cfg, str(tmp_path))
    return cfg, paths, json.load(open(truth_path))


def test_stack_recovers_faint_mover(tmp_path):
    cfg, paths, truth = _faint_set(tmp_path, flux=700)
    fset = load_frame_set(paths)
    af0 = truth["asteroid"]["frames"][0]
    ax, ay = af0["x"], af0["y"]

    # single-frame: the faint mover is below the detection threshold
    s0 = extract_sources(fset.frames[0].data, fwhm=3.2, threshold_sigma=5.0)
    d = (np.hypot(np.asarray(s0["x"]) - ax, np.asarray(s0["y"]) - ay)
         if len(s0) else np.array([99.0]))
    assert d.min() > 3.0, "faint mover should NOT be detected in a single frame"

    snr_single = peak_snr_at(fset.frames[0].data, ax, ay)
    stack, _t0 = shift_and_stack(
        fset.frames, cfg.asteroid_rate_arcsec_min, cfg.asteroid_pa_deg
    )
    snr_stack = peak_snr_at(stack, ax, ay)
    assert snr_stack > 1.5 * snr_single, (snr_single, snr_stack)  # ~sqrt(N) gain

    ss = extract_sources(stack, fwhm=3.2, threshold_sigma=5.0)
    d2 = (np.hypot(np.asarray(ss["x"]) - ax, np.asarray(ss["y"]) - ay)
          if len(ss) else np.array([99.0]))
    assert d2.min() < 3.0, "mover should be recovered in the stack"


def test_digital_tracking_finds_velocity(tmp_path):
    cfg, paths, truth = _faint_set(tmp_path, flux=800)
    fset = load_frame_set(paths)
    af0 = truth["asteroid"]["frames"][0]
    ax, ay = af0["x"], af0["y"]

    rates = np.array([0.8, 1.1, 1.4, 1.6, 1.9, 2.2])   # includes true 1.6
    pas = np.array([25.0, 40.0, 55.0, 70.0, 85.0])     # includes true 55
    hits = digital_tracking(fset.frames, rates, pas, threshold_sigma=6.0)
    assert hits, "grid search should produce hits"

    top = hits[0]
    assert np.hypot(top.x - ax, top.y - ay) < 4.0           # top hit at the mover
    assert abs(top.rate - cfg.asteroid_rate_arcsec_min) <= 0.3
    assert abs(top.pa - cfg.asteroid_pa_deg) <= 15.0
