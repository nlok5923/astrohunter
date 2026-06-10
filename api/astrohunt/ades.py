"""MPC report generation: ADES (PSV) and legacy 80-column (obs80).

Research finding: ADES is now the dominant submission format; the hard acceptance
gate is server-side validation against submit.xsd (the MPC performs the
PSV->XML conversion and validation themselves). We therefore emit standards-shaped
ADES PSV with the required columns and ISO-8601 UTC timestamps, plus a best-effort
obs80 line for the legacy channel. This does NOT submit anything — it produces a
file a human reviews before submitting.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from xml.dom import minidom

import numpy as np
from astropy.coordinates import Angle
from astropy.time import Time
import astropy.units as u

# The current IAU ADES submission schema (submit.xsd) requires version "2022".
ADES_VERSION = "2022"


@dataclass
class ObsMetadata:
    obs_code: str = "XXX"          # MPC observatory code (XXX = unassigned/placeholder)
    observer: str = "Unknown"
    measurer: str = "Unknown"
    telescope_design: str = "reflector"
    aperture_m: float = 0.5        # metres; required by ADES — set your real aperture
    detector: str = "CCD"
    ast_cat: str = "Gaia3"         # astrometric reference catalogue (ADES code)
    band: str = "G"
    submitter: str = "Unknown"
    extra: dict = field(default_factory=dict)


# ADES PSV columns we populate (subset of the standard, sufficient for submission).
_ADES_COLUMNS = [
    "trkSub", "mode", "stn", "obsTime", "ra", "dec",
    "rmsRA", "rmsDec", "mag", "band", "astCat", "remarks",
]
ADES_COLUMNS = _ADES_COLUMNS   # public alias


def _iso_utc(mjd: float) -> str:
    t = Time(mjd, format="mjd", scale="utc")
    s = t.isot  # 'YYYY-MM-DDTHH:MM:SS.sss'
    if "." not in s:
        s += ".000"
    # ADES wants fractional seconds; keep millisecond precision and a trailing Z.
    head, frac = s.split(".")
    return f"{head}.{frac[:3]:<03}Z"


def _trksub(index: int) -> str:
    # Up to 8 chars; deterministic per-run temporary tracklet id.
    return f"aH{index:05d}"[:8]


def tracklet_to_ades_rows(tracklet, meta: ObsMetadata, index: int):
    rows = []
    trk = _trksub(index)
    for d in tracklet.detections:
        rms = f"{tracklet.rms_arcsec:.2f}" if np.isfinite(tracklet.rms_arcsec) else ""
        mag = f"{d.mag:.1f}" if np.isfinite(d.mag) else ""
        rows.append({
            "trkSub": trk,
            "mode": "CCD",
            "stn": meta.obs_code,
            "obsTime": _iso_utc(d.time_mjd),
            "ra": f"{d.ra:.6f}",
            "dec": f"{d.dec:+.6f}",
            "rmsRA": rms,
            "rmsDec": rms,
            "mag": mag,
            "band": meta.band,
            "astCat": meta.ast_cat,
            "remarks": tracklet.known_name or "",
        })
    return rows


def ades_metadata_lines(meta: ObsMetadata) -> list[str]:
    """The PSV header block (# blocks / ! key-values)."""
    return [
        f"# version={ADES_VERSION}",
        "# observatory",
        f"! mpcCode {meta.obs_code}",
        "# submitter",
        f"! name {meta.submitter}",
        "# observers",
        f"! name {meta.observer}",
        "# measurers",
        f"! name {meta.measurer}",
        "# telescope",
        f"! design {meta.telescope_design}",
        f"! aperture {meta.aperture_m or 0.5}",
        f"! detector {meta.detector}",
    ]


def ades_psv_text(tracklets, meta: ObsMetadata) -> str:
    lines = list(ades_metadata_lines(meta))
    lines.append("|".join(_ADES_COLUMNS))
    idx = 1
    for trk in tracklets:
        for row in tracklet_to_ades_rows(trk, meta, idx):
            lines.append("|".join(str(row[c]) for c in _ADES_COLUMNS))
        idx += 1
    return "\n".join(lines) + "\n"


def write_ades_psv(tracklets, meta: ObsMetadata, path: str) -> str:
    """Write an ADES PSV file for all (candidate) tracklets. Returns the path."""
    with open(path, "w") as f:
        f.write(ades_psv_text(tracklets, meta))
    return path


# --- legacy 80-column (obs80) ----------------------------------------------

def _obs80_date(mjd: float) -> str:
    t = Time(mjd, format="mjd", scale="utc")
    y, mo, d, h, mi, s = t.ymdhms
    frac_day = (h + mi / 60.0 + s / 3600.0) / 24.0
    day = d + frac_day
    return f"{y:4d} {mo:02d} {day:08.5f}"


def tracklet_to_obs80(tracklet, meta: ObsMetadata, index: int):
    lines = []
    desig = _trksub(index)  # temporary designation in cols 6-12
    for k, d in enumerate(tracklet.detections):
        ra_str = Angle(d.ra, u.deg).to_string(unit=u.hour, sep=" ", precision=2, pad=True)
        dec_str = Angle(d.dec, u.deg).to_string(
            unit=u.deg, sep=" ", precision=1, pad=True, alwayssign=True
        )
        date_str = _obs80_date(d.time_mjd)
        disc = "*" if k == 0 else " "
        mag = f"{d.mag:4.1f}" if np.isfinite(d.mag) else "    "
        line = (
            f"{'':5}"            # cols 1-5: number (blank, unnumbered)
            f"{desig:<7.7}"      # cols 6-12: temporary designation
            f"{disc}"            # col 13: discovery asterisk
            f"{' '}"             # col 14: note 1
            f"{'C'}"             # col 15: note 2 (CCD)
            f"{date_str:<17.17}" # cols 16-32: date
            f"{ra_str:<12.12}"   # cols 33-44: RA
            f"{dec_str:<12.12}"  # cols 45-56: Dec
            f"{'':9}"            # cols 57-65: blank
            f"{mag:<5.5} "       # cols 66-71: mag + band
            f"{meta.band:1.1}"
            f"{'':5}"            # cols 72-77: blank
            f"{meta.obs_code:>3.3}"  # cols 78-80: obs code
        )
        lines.append(line[:80])
    return lines


def write_obs80(tracklets, meta: ObsMetadata, path: str) -> str:
    lines = []
    idx = 1
    for trk in tracklets:
        lines.extend(tracklet_to_obs80(trk, meta, idx))
        idx += 1
    with open(path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return path


# --- ADES XML (the MPC's preferred representation) -------------------------

def _sub(parent, tag, text):
    el = ET.SubElement(parent, tag)
    el.text = str(text)
    return el


def build_ades_xml(tracklets, meta: ObsMetadata) -> str:
    """Build an ADES XML document string (preference order XML > PSV > obs80)."""
    ades = ET.Element("ades", {"version": ADES_VERSION})
    block = ET.SubElement(ades, "obsBlock")

    ctx = ET.SubElement(block, "obsContext")
    obs = ET.SubElement(ctx, "observatory")
    _sub(obs, "mpcCode", meta.obs_code)
    _sub(ET.SubElement(ctx, "submitter"), "name", meta.submitter)
    _sub(ET.SubElement(ctx, "observers"), "name", meta.observer)
    _sub(ET.SubElement(ctx, "measurers"), "name", meta.measurer)
    tel = ET.SubElement(ctx, "telescope")
    _sub(tel, "design", meta.telescope_design)
    _sub(tel, "aperture", meta.aperture_m or 0.5)   # required, positive
    _sub(tel, "detector", meta.detector)

    data = ET.SubElement(block, "obsData")
    idx = 1
    for trk in tracklets:
        for row in tracklet_to_ades_rows(trk, meta, idx):
            opt = ET.SubElement(data, "optical")
            _sub(opt, "trkSub", row["trkSub"])
            _sub(opt, "mode", row["mode"])
            _sub(opt, "stn", row["stn"])
            _sub(opt, "obsTime", row["obsTime"])
            _sub(opt, "ra", row["ra"])
            _sub(opt, "dec", row["dec"].lstrip("+"))
            if row["rmsRA"]:
                _sub(opt, "rmsRA", row["rmsRA"])
            if row["rmsDec"]:
                _sub(opt, "rmsDec", row["rmsDec"])
            _sub(opt, "astCat", row["astCat"])
            if row["mag"]:
                _sub(opt, "mag", row["mag"])
                _sub(opt, "band", row["band"])
        idx += 1

    rough = ET.tostring(ades, encoding="unicode")
    return minidom.parseString(rough).toprettyxml(indent="  ")


def write_ades_xml(tracklets, meta: ObsMetadata, path: str) -> str:
    with open(path, "w") as f:
        f.write(build_ades_xml(tracklets, meta))
    return path
