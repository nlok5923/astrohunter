"""Tests for the Phase-1.5 features: plate-solving, validation, vetting, review."""

import json

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

from astrohunt.catalog import load_reference_catalog
from astrohunt.extract import extract_sources
from astrohunt.io import load_frame
from astrohunt.pipeline import run_pipeline
from astrohunt.platesolve import solve_wcs
from astrohunt.synthetic import SyntheticConfig, _add_psf, generate
from astrohunt.validate import validate_psv_text
from astrohunt.vet import vet_sources


# --- plate-solving ----------------------------------------------------------

def test_plate_solve_recovers_wcs(tmp_path):
    paths, truth_path = generate(SyntheticConfig(seed=11), str(tmp_path))
    truth = json.load(open(truth_path))
    refcat = load_reference_catalog(truth_path)
    frame = load_frame(paths[0])
    srcs = extract_sources(frame.data)

    tc = truth["center"]
    # deliberately coarse, offset pointing hint
    res = solve_wcs(srcs["x"], srcs["y"], refcat.ra, refcat.dec,
                    tc["ra"] + 0.03, tc["dec"] - 0.02, ref_mag=refcat.mag)
    assert res.n_inliers >= 30

    ny, nx = frame.data.shape
    ra_t, dec_t = frame.wcs.all_pix2world([nx / 2], [ny / 2], 0)
    ra_s, dec_s = res.wcs.all_pix2world([nx / 2], [ny / 2], 0)
    sep = SkyCoord(ra_t * u.deg, dec_t * u.deg).separation(
        SkyCoord(ra_s * u.deg, dec_s * u.deg)
    ).arcsec
    assert sep.max() < 1.0


def test_pipeline_autosolves_headerless(tmp_path):
    paths, truth_path = generate(SyntheticConfig(seed=5), str(tmp_path), strip_wcs=True)
    refcat = load_reference_catalog(truth_path)
    tc = json.load(open(truth_path))["center"]
    result = run_pipeline(
        paths, reference_catalog=refcat, use_online=False,
        hint_radec=(tc["ra"], tc["dec"]),
    )
    assert result.median_rms_arcsec < 1.0
    assert result.n_candidates >= 1
    # the auto-solve should have recorded a plate-solve note per frame
    assert any("plate-solved" in w for w in result.warnings)


# --- ADES validation --------------------------------------------------------

def test_ades_validation_pass():
    good = ("# version=2017\n"
            "trkSub|mode|stn|obsTime|ra|dec\n"
            "aH1|CCD|568|2026-06-08T10:00:00.000Z|187.5|+12.0\n")
    r = validate_psv_text(good)
    assert r.ok, r.errors
    assert r.n_rows == 1


def test_ades_validation_catches_errors():
    bad = ("# version=2017\n"
           "trkSub|mode|stn|obsTime|ra|dec\n"
           "aH1|CCD|568|notatime|400.0|+12.0\n")
    r = validate_psv_text(bad)
    assert not r.ok
    assert any("obsTime" in e for e in r.errors)
    assert any("ra=" in e or "range" in e for e in r.errors)


def test_strict_xsd_validation_passes():
    import pytest

    pytest.importorskip("lxml")
    from astrohunt.ades import ObsMetadata, build_ades_xml
    from astrohunt.detect import Detection, Tracklet
    from astrohunt.validate import validate_xsd

    dets = [Detection(frame=i, time_mjd=61199.4 + i * 0.01, x=100, y=100,
                      ra=187.5 + i * 0.001, dec=12.0 + i * 0.001, mag=19.5, flux=1e3)
            for i in range(4)]
    trk = Tracklet(detections=dets, rate_total=1.6, rms_arcsec=0.3, pa_deg=55)
    meta = ObsMetadata(obs_code="568", observer="J Doe", measurer="J Doe", submitter="J Doe")
    r = validate_xsd(build_ades_xml([trk], meta))
    assert r.ok, r.errors


def test_ades_validation_flags_placeholder_obscode():
    psv = ("# version=2017\n"
           "trkSub|mode|stn|obsTime|ra|dec\n"
           "aH1|CCD|XXX|2026-06-08T10:00:00.000Z|187.5|+12.0\n")
    r = validate_psv_text(psv)
    assert r.ok  # placeholder is a warning, not a hard error
    assert any("placeholder" in w for w in r.warnings)


# --- real-bogus vetting -----------------------------------------------------

def test_vet_rejects_sharp_spike():
    img = np.full((60, 60), 100.0)
    _add_psf(img, 20, 20, 3.0e4, 3.2)   # a real PSF star
    img[40, 40] += 6.0e4                 # a 1-pixel cosmic ray / hot pixel
    srcs = Table({"x": [20.0, 40.0], "y": [20.0, 40.0]})
    real, _stats = vet_sources(img, srcs, psf_fwhm=3.2)
    assert bool(real[0]) is True    # star kept
    assert bool(real[1]) is False   # spike rejected


def test_vetting_keeps_real_sources(tmp_path):
    paths, truth_path = generate(
        SyntheticConfig(seed=3, n_cosmic_rays=12), str(tmp_path)
    )
    frame = load_frame(paths[0])
    srcs = extract_sources(frame.data)
    _real, stats = vet_sources(frame.data, srcs)
    assert stats.n_real >= 85   # the ~90 injected stars survive vetting


# --- visual review ----------------------------------------------------------

def test_html_review_written(tmp_path):
    paths, truth_path = generate(SyntheticConfig(seed=7), str(tmp_path))
    refcat = load_reference_catalog(truth_path)
    html = str(tmp_path / "review.html")
    result = run_pipeline(
        paths, reference_catalog=refcat, use_online=False, out_html=html
    )
    assert result.html_path
    text = open(html).read()
    assert "data:image/gif;base64" in text     # blink animation embedded
    assert "NEW CANDIDATE" in text
    # interactive app: accept/reject + ADES export wired in
    assert "exportAccepted" in text
    assert "function buildPSV" in text
    assert "localStorage" in text
