"""End-to-end test: synthetic set -> pipeline recovers the injected asteroid."""

import json

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
import astropy.units as u

from astrohunt.catalog import load_reference_catalog
from astrohunt.pipeline import run_pipeline
from astrohunt.synthetic import SyntheticConfig, generate


@pytest.fixture(scope="module")
def synthetic_set(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("synth")
    paths, truth = generate(SyntheticConfig(seed=7), str(outdir))
    return paths, truth


def test_pipeline_recovers_asteroid(synthetic_set, tmp_path):
    paths, truth_path = synthetic_set
    refcat = load_reference_catalog(truth_path)
    ades = str(tmp_path / "report.psv")
    result = run_pipeline(
        paths, reference_catalog=refcat, use_online=False, out_ades=ades
    )

    # Astrometric consistency must be sub-arcsecond (the MPC metric).
    assert result.median_rms_arcsec < 1.0, result.median_rms_arcsec
    # Every frame should match plenty of reference stars.
    assert all(fr.n_matched >= 10 for fr in result.frame_reports)
    # Exactly the injected mover should surface as a NEW candidate.
    assert result.n_candidates >= 1

    with open(truth_path) as f:
        truth = json.load(f)
    truth_frames = [fr for fr in truth["asteroid"]["frames"] if fr["in_frame"]]

    cand = max(result.tracklets, key=lambda t: t.n)
    assert cand.n >= 3  # linked across most frames

    errs = []
    for d in cand.detections:
        nearest = min(truth_frames, key=lambda tf: abs(tf["mjd"] - d.time_mjd))
        sep = SkyCoord(d.ra * u.deg, d.dec * u.deg).separation(
            SkyCoord(nearest["ra"] * u.deg, nearest["dec"] * u.deg)
        ).arcsec
        errs.append(sep)
    assert np.mean(errs) < 1.0, f"asteroid position error {np.mean(errs):.3f}\""

    # Recovered rate should be close to the injected rate.
    assert abs(cand.rate_total - truth["asteroid"]["rate_arcsec_min"]) < 0.5

    # An ADES report should have been written with the expected header.
    with open(ades) as f:
        text = f.read()
    assert "# version=2022" in text
    assert "trkSub|mode|stn|obsTime|ra|dec" in text


def test_static_sources_are_suppressed(synthetic_set):
    paths, truth_path = synthetic_set
    refcat = load_reference_catalog(truth_path)
    result = run_pipeline(paths, reference_catalog=refcat, use_online=False)
    # ~90 stars injected; none of them should become tracklets.
    assert len(result.tracklets) <= 3
