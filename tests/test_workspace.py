"""Astrometrica-parity features: click-to-measure, WCS refinement, MPC 80-column
report and the interactive workspace (all offline, on synthetic frames)."""

import json

import numpy as np
import pytest

from astrohunt.catalog import load_reference_catalog
from astrohunt.extract import extract_sources
from astrohunt.io import load_frame
from astrohunt.measure import measure_at
from astrohunt.mpcreport import (
    ReportHeader, ReportObject, ReportObs, build_report, check_report,
    designation_columns, pack_number, pack_provisional,
)
from astrohunt.platesolve import refine_wcs
from astrohunt.synthetic import SyntheticConfig, generate
from astrohunt.workspace import Workspace, make_settings


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("ws")
    paths, truth = generate(SyntheticConfig(seed=7, n_cosmic_rays=4), str(outdir))
    return str(outdir), paths, truth


def test_obs_line_matches_astrometrica_format():
    # verbatim line from an IASC-published Astrometrica report (ExtraDigit=1)
    ref = "     NEW0001  C2019 10 24.58871204 11 48.498+14 24 22.67         21.0 R      F51"
    ra = (4 + 11 / 60 + 48.498 / 3600) * 15
    dec = 14 + 24 / 60 + 22.67 / 3600
    text = build_report(
        [ReportObject("NEW0001", [ReportObs(58780.588712, ra, dec, 21.0, "R")])],
        ReportHeader(cod="F51"), extra_digit=True)
    lines = text.splitlines()
    assert lines[0] == "COD F51" and lines[-1] == "----- end -----"
    assert lines[1] == ref
    # standard-precision variant keeps every field in its column
    std = build_report(
        [ReportObject("NEW0001", [ReportObs(58780.588712, ra, dec, 21.0, "R")])],
        ReportHeader(cod="F51")).splitlines()[1]
    assert len(std) == 80 and std[14] == "C" and std[70] == "R" and std[77:] == "F51"
    assert std[15:32] == "2019 10 24.58871 " and std[32:44] == "04 11 48.50 "


def test_report_checks_and_empty_form():
    hdr = ReportHeader(cod="F51", net="Gaia DR3")
    obs = [ReportObs(59000.1 + 0.01 * k, 10.0 + 0.001 * k, -5.0, 20.0, "G") for k in range(4)]
    assert check_report(build_report([ReportObject("ABC0001", obs)], hdr)) == []
    short = check_report(build_report([ReportObject("ABC0001", obs[:2])], hdr))
    assert any("only 2 observation" in p for p in short)
    empty = build_report([], hdr)
    assert "No moving objects detected" in empty and check_report(empty) == []
    with pytest.raises(ValueError):
        designation_columns("TOOLONG01")


def test_designation_packing():
    assert pack_number(1234) == "01234"
    assert pack_number(406714) == "e6714"
    assert pack_number(620000) == "~0000"
    assert pack_provisional("2020 AB1") == "K20A01B"
    assert pack_provisional("2007 TA418") == "K07Tf8A"
    assert designation_columns("406714") == "e6714       "
    assert designation_columns("2019 PK5") == "     K19P05K"
    assert designation_columns("ABC0001") == "     ABC0001"


def test_measure_at_recovers_position(synth):
    _dir, paths, truth_path = synth
    frame = load_frame(paths[0])
    with open(truth_path) as f:
        ast = json.load(f)["asteroid"]["frames"][0]
    m = measure_at(frame.data, ast["x"] + 2.0, ast["y"] - 1.5, fwhm_guess=3.2)
    assert m.ok and m.snr > 20
    assert abs(m.x - ast["x"]) < 0.15 and abs(m.y - ast["y"]) < 0.15
    assert abs(m.fwhm - 3.2) < 0.4
    assert not measure_at(frame.data, 12.0, 12.0, fwhm_guess=3.2).ok   # empty sky


def test_refine_wcs_fixes_a_bad_header(synth):
    _dir, paths, truth_path = synth
    frame = load_frame(paths[0])
    cat = load_reference_catalog(truth_path)
    srcs = extract_sources(frame.data)
    bad = frame.wcs.deepcopy()
    bad.wcs.crpix = bad.wcs.crpix + np.array([23.0, -17.0])      # ~35" pointing error
    a = np.radians(0.6)
    bad.wcs.pc = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    fixed, n, rms_px = refine_wcs(srcs["x"], srcs["y"], bad, cat.ra, cat.dec, cat.mag)
    assert n >= 50 and rms_px < 0.3
    x, y = fixed.all_world2pix(cat.ra, cat.dec, 0)
    tx, ty = frame.wcs.all_world2pix(cat.ra, cat.dec, 0)
    assert np.median(np.hypot(x - tx, y - ty)) < 0.3


def test_workspace_end_to_end(synth):
    folder, _paths, truth_path = synth
    ws = Workspace.from_folder(
        folder, make_settings("ps1", use_online=False, static_tol_arcsec=2.0,
                              link_tol_arcsec=2.0, max_rate_arcsec_min=10.0))
    info = ws.reduce(load_reference_catalog(truth_path))
    assert info["reduced"] and info["median_rms_arcsec"] < 0.3
    assert len(ws.render_png(0)) > 1000

    tracklets = ws.detect()
    assert len(tracklets) == 1                    # the mover, not the cosmic rays
    trk = tracklets[0]
    with open(truth_path) as f:
        truth = json.load(f)["asteroid"]
    assert trk["n"] == 4 and abs(trk["rate"] - truth["rate_arcsec_min"]) < 0.1

    # manual path: click near the object in every frame, as a human would
    manual = [ws.measure(m["frame"], m["x"] + 1.2, m["y"] - 0.8) for m in trk["measurements"]]
    assert all(m["ok"] for m in manual)
    pred = ws.predict(manual[:2], 3)
    assert abs(pred["x"] - manual[3]["x"]) < 1.0 and abs(pred["y"] - manual[3]["y"]) < 1.0

    rep = ws.report([{"designation": "BIM0001", "measurements": manual}])
    lines = rep["text"].splitlines()
    assert lines[0] == "COD F51" and lines[-1] == "----- end -----"
    obs = [ln for ln in lines if ln.startswith("     BIM0001")]
    assert len(obs) == 4 and all(len(ln) == 80 for ln in obs)
    assert rep["problems"] == []
    assert "No moving objects detected" in ws.report([])["text"]


def test_workspace_api(synth):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from astrohunt.webapi import app

    folder, _paths, truth_path = synth
    c = TestClient(app)
    assert "astrohunt workspace" in c.get("/").text
    assert "ps1" in c.get("/api/presets").json()
    r = c.post("/api/ws/open", json={"path": folder, "preset": "ps1",
                                     "settings": {"use_online": False}})
    assert r.status_code == 200, r.text
    ws_id = r.json()["id"]
    assert c.post(f"/api/ws/{ws_id}/measure", json={"frame": 0, "x": 5, "y": 5}).status_code == 422
    r = c.post(f"/api/ws/{ws_id}/reduce", json={"ref_catalog_path": truth_path})
    assert r.status_code == 200 and r.json()["reduced"], r.text
    assert c.get(f"/api/ws/{ws_id}/frame/0.png").headers["content-type"] == "image/png"
    assert c.post("/api/ws/open", json={"path": "/no/such/folder"}).status_code == 422
    assert c.get("/api/ws/nope").status_code == 404
