"""Reference catalogues and known-object matching.

Online services (Gaia DR3 for astrometric reference, SkyBoT for known asteroids)
are used when available, but every call degrades gracefully to None/empty so the
pipeline still runs fully offline on synthetic or archival data.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class ReferenceCatalog:
    ra: np.ndarray
    dec: np.ndarray
    mag: np.ndarray
    source: str = "unknown"

    def __len__(self):
        return len(self.ra)


def load_reference_catalog(path: str) -> ReferenceCatalog:
    """Load a JSON reference catalogue: {"stars": [{"ra","dec","mag"}, ...]}."""
    with open(path) as f:
        d = json.load(f)
    stars = d.get("stars", d) if isinstance(d, dict) else d
    ra = np.array([s["ra"] for s in stars], float)
    dec = np.array([s["dec"] for s in stars], float)
    mag = np.array([s.get("mag", np.nan) for s in stars], float)
    return ReferenceCatalog(ra, dec, mag, source=str(path))


def gaia_cone(ra_deg, dec_deg, radius_arcmin=12.0, mag_limit=20.5) -> ReferenceCatalog | None:
    """Fetch Gaia DR3 reference stars in a cone. Returns None on any failure."""
    try:
        from astroquery.gaia import Gaia
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        Gaia.ROW_LIMIT = 50000
        c = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            job = Gaia.cone_search_async(c, radius=radius_arcmin * u.arcmin)
            tbl = job.get_results()
        m = np.asarray(tbl["phot_g_mean_mag"], float)
        keep = np.isfinite(m) & (m <= mag_limit)
        return ReferenceCatalog(
            np.asarray(tbl["ra"], float)[keep],
            np.asarray(tbl["dec"], float)[keep],
            m[keep],
            source="GaiaDR3",
        )
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"Gaia query failed ({e}); falling back.")
        return None


def skybot_cone(ra_deg, dec_deg, epoch_mjd, radius_arcmin=15.0):
    """Query SkyBoT (IMCCE) for known solar-system objects. None on failure."""
    try:
        from astroquery.imcce import Skybot
        from astropy.coordinates import SkyCoord
        from astropy.time import Time
        import astropy.units as u

        c = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return Skybot.cone_search(c, radius_arcmin * u.arcmin, Time(epoch_mjd, format="mjd"))
    except Exception:
        return None  # nothing found OR offline -> treat as no known objects


def identify_tracklet(tracklet, skybot_table, tol_arcsec=12.0):
    """Return the name of a matching known object, or None (candidate)."""
    if skybot_table is None or len(skybot_table) == 0:
        return None
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    mid = tracklet.detections[len(tracklet.detections) // 2]
    pos = SkyCoord(mid.ra * u.deg, mid.dec * u.deg)
    known = SkyCoord(skybot_table["RA"], skybot_table["DEC"])
    sep = pos.separation(known).arcsec
    j = int(np.argmin(sep))
    if sep[j] <= tol_arcsec:
        return str(skybot_table["Name"][j])
    return None
