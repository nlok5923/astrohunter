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


def _cache_dir():
    import os

    d = os.path.join(os.path.expanduser("~"), ".cache", "astrohunt")
    os.makedirs(d, exist_ok=True)
    return d


def _apply_pm(ra, dec, pmra, pmdec, epoch_year, ref_epoch=2016.0):
    """Propagate Gaia positions (epoch 2016.0) to the observation epoch."""
    dt = float(epoch_year) - ref_epoch
    pmra = np.where(np.isfinite(pmra), pmra, 0.0)     # mas/yr, includes cos(dec)
    pmdec = np.where(np.isfinite(pmdec), pmdec, 0.0)
    dec2 = dec + pmdec * dt / 3.6e6
    ra2 = ra + pmra * dt / 3.6e6 / np.clip(np.cos(np.radians(dec)), 1e-6, None)
    return ra2 % 360.0, dec2


def _tap_csv(url, query, timeout):
    """Run a synchronous TAP/ADQL query, return columns as float arrays."""
    import csv
    import io
    import urllib.parse
    import urllib.request

    body = urllib.parse.urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query}).encode()
    req = urllib.request.Request(url, data=body, headers={"User-Agent": "astrohunt"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        text = r.read().decode("utf-8", "replace")
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 2 or len(rows[0]) < 6:
        raise RuntimeError("no stars returned")

    def num(v):
        try:
            return float(v)
        except ValueError:
            return np.nan

    arr = np.array([[num(v) for v in row[:6]] for row in rows[1:] if len(row) >= 6], float)
    return tuple(arr[:, k] for k in range(6))


def _gaia_esa(ra_deg, dec_deg, radius_deg, mag_limit, timeout=60):
    q = (
        "SELECT ra, dec, pmra, pmdec, phot_g_mean_mag, phot_rp_mean_mag "
        "FROM gaiadr3.gaia_source WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {ra_deg:.6f}, {dec_deg:.6f}, {radius_deg:.5f})) "
        f"AND phot_g_mean_mag < {mag_limit:.2f}"
    )
    return _tap_csv("https://gea.esac.esa.int/tap-server/tap/sync", q, timeout)


def _gaia_vizier(ra_deg, dec_deg, radius_deg, mag_limit, server, timeout=30):
    q = (
        'SELECT RA_ICRS, DE_ICRS, pmRA, pmDE, Gmag, RPmag FROM "I/355/gaiadr3" '
        "WHERE 1=CONTAINS(POINT('ICRS', RA_ICRS, DE_ICRS), "
        f"CIRCLE('ICRS', {ra_deg:.6f}, {dec_deg:.6f}, {radius_deg:.5f})) "
        f"AND Gmag < {mag_limit:.2f}"
    )
    return _tap_csv(f"https://{server}/TAPVizieR/tap/sync", q, timeout)


def _gaia_vizier_tsv(ra_deg, dec_deg, radius_deg, mag_limit, server, timeout=40):
    """Classic VizieR 'asu-tsv' interface — served by mirrors that have no TAP."""
    import urllib.parse
    import urllib.request

    params = [
        ("-source", "I/355/gaiadr3"), ("-c", f"{ra_deg:.6f} {dec_deg:+.6f}"),
        ("-c.rd", f"{radius_deg:.5f}"), ("-out.max", "unlimited"),
        ("-out", "RA_ICRS,DE_ICRS,pmRA,pmDE,Gmag,RPmag"), ("Gmag", f"<{mag_limit:.2f}"),
    ]
    url = f"https://{server}/viz-bin/asu-tsv?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "astrohunt"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        text = r.read().decode("utf-8", "replace")
    rows = []
    for ln in text.splitlines():
        parts = ln.split("\t")
        if len(parts) < 6 or ln.startswith("#"):
            continue
        try:
            ra, dec = float(parts[0]), float(parts[1])
        except ValueError:
            continue          # column names / units / dashes
        vals = [ra, dec]
        for v in parts[2:6]:
            try:
                vals.append(float(v))
            except ValueError:
                vals.append(np.nan)
        rows.append(vals)
    if not rows:
        raise RuntimeError("no stars returned")
    arr = np.array(rows, float)
    return tuple(arr[:, k] for k in range(6))


VIZIER_TSV_SERVERS = ["vizier.cfa.harvard.edu", "vizier.cds.unistra.fr"]
VIZIER_SERVERS = ["tapvizier.cds.unistra.fr", "vizier.nao.ac.jp", "vizier.china-vo.org"]


def gaia_cone(ra_deg, dec_deg, radius_arcmin=12.0, mag_limit=20.5,
              epoch_year=None, cache=True) -> ReferenceCatalog | None:
    """Fetch Gaia DR3 reference stars in a cone. Returns None on any failure.

    Tries the ESA Gaia archive first, then the VizieR mirrors (Harvard,
    Strasbourg, Tokyo, Beijing), each with a hard timeout — so one server being down doesn't block data reduction.
    Positions are propagated to `epoch_year` with Gaia proper motions, and the
    raw download is cached on disk so re-opening a field works offline.
    """
    import os

    radius_deg = radius_arcmin / 60.0
    key = f"gaiadr3_{ra_deg:.4f}_{dec_deg:+.4f}_{radius_arcmin:.1f}_{mag_limit:.1f}.npz"
    path = os.path.join(_cache_dir(), key)
    data = None
    source = "GaiaDR3"
    if cache and os.path.exists(path):
        try:
            z = np.load(path)
            data = tuple(z[k] for k in ("ra", "dec", "pmra", "pmdec", "g", "rp"))
            source = "GaiaDR3 (cached)"
        except Exception:  # noqa: BLE001
            data = None
    if data is None:
        errors = []
        attempts = [("ESA Gaia archive",
                     lambda: _gaia_esa(ra_deg, dec_deg, radius_deg, mag_limit, timeout=30))]
        attempts += [(f"VizieR {srv}", (lambda srv=srv: _gaia_vizier_tsv(
            ra_deg, dec_deg, radius_deg, mag_limit, srv))) for srv in VIZIER_TSV_SERVERS]
        attempts += [(f"VizieR {srv}", (lambda srv=srv: _gaia_vizier(
            ra_deg, dec_deg, radius_deg, mag_limit, srv))) for srv in VIZIER_SERVERS]
        for name, fn in attempts:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data = fn()
                if len(data[0]) == 0:
                    raise RuntimeError("no stars returned")
                source = f"GaiaDR3 via {name}"
                break
            except Exception as e:  # noqa: BLE001
                errors.append(f"{name}: {e}")
                data = None
        if data is None:
            warnings.warn("Gaia query failed (" + "; ".join(errors) + "); falling back.")
            return None
        if cache:
            try:
                np.savez(path, ra=data[0], dec=data[1], pmra=data[2], pmdec=data[3],
                         g=data[4], rp=data[5])
            except Exception:  # noqa: BLE001
                pass

    ra, dec, pmra, pmdec, g, _rp = data
    if epoch_year is not None:
        ra, dec = _apply_pm(ra, dec, pmra, pmdec, epoch_year)
    keep = np.isfinite(g) & (g <= mag_limit) & np.isfinite(ra) & np.isfinite(dec)
    return ReferenceCatalog(ra[keep], dec[keep], g[keep], source=source)


def skybot_cone(ra_deg, dec_deg, epoch_mjd, radius_arcmin=15.0, location="500"):
    """Query SkyBoT (IMCCE) for known solar-system objects. None on failure."""
    try:
        from astroquery.imcce import Skybot
        from astropy.coordinates import SkyCoord
        from astropy.time import Time
        import astropy.units as u

        Skybot.TIMEOUT = 60
        c = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tbl = Skybot.cone_search(c, radius_arcmin * u.arcmin,
                                     Time(epoch_mjd, format="mjd"), location=location)
        tbl.meta["query_mjd"] = float(epoch_mjd)   # positions are for this instant
        return tbl
    except Exception:
        return None  # nothing found OR offline -> treat as no known objects


def identify_tracklet(tracklet, skybot_table, tol_arcsec=25.0):
    """Return the name of a matching known object, or None (candidate).

    The SkyBoT position is propagated to the tracklet's own epoch with the
    catalogued motion before comparing, and the motion itself must roughly agree.
    """
    if skybot_table is None or len(skybot_table) == 0:
        return None
    import astropy.units as u

    t = skybot_table
    mid = tracklet.detections[len(tracklet.detections) // 2]
    ra = np.asarray(u.Quantity(t["RA"]).to_value(u.deg), float)
    dec = np.asarray(u.Quantity(t["DEC"]).to_value(u.deg), float)
    rr, dr = _col(t, "RA_rate"), _col(t, "DEC_rate")
    have_rate = rr is not None and dr is not None
    if have_rate:
        ra_rate = np.asarray(u.Quantity(rr).to_value(u.arcsec / u.hour), float)
        dec_rate = np.asarray(u.Quantity(dr).to_value(u.arcsec / u.hour), float)
        dt_h = (mid.time_mjd - float(t.meta.get("query_mjd", mid.time_mjd))) * 24.0
        cosd = np.clip(np.cos(np.radians(dec)), 1e-6, None)
        ra = ra + ra_rate * dt_h / 3600.0 / cosd
        dec = dec + dec_rate * dt_h / 3600.0
    cosm = np.cos(np.radians(mid.dec))
    sep = np.hypot(((ra - mid.ra + 180.0) % 360.0 - 180.0) * cosm, dec - mid.dec) * 3600.0
    # nearest object whose catalogued motion also agrees with the measured one
    # (SkyBoT ephemerides can be several arcsec off, so position alone is weak)
    for j in np.argsort(sep):
        if sep[j] > tol_arcsec:
            return None
        if not have_rate or tracklet.rate_total <= 0:
            break
        dv = np.hypot(ra_rate[j] / 60.0 - tracklet.rate_ra, dec_rate[j] / 60.0 - tracklet.rate_dec)
        if dv <= max(0.08, 0.2 * tracklet.rate_total):
            break
    else:
        return None
    return str(t["Name"][j]).strip()

def _col(table, *names):
    for n in names:
        if n in table.colnames:
            return table[n]
    return None


def known_positions(skybot_table, frames, solutions=None) -> list[dict]:
    """Per-frame positions of the known objects in a SkyBoT table.

    SkyBoT returns each object's position at one epoch plus its apparent motion
    (arcsec/h); we propagate linearly to every frame time — accurate to well
    under an arcsecond over the ~1 h an image set spans — and convert to pixels.
    """
    if skybot_table is None or len(skybot_table) == 0:
        return []
    import astropy.units as u

    t = skybot_table
    ra = np.asarray(u.Quantity(t["RA"]).to_value(u.deg), float)
    dec = np.asarray(u.Quantity(t["DEC"]).to_value(u.deg), float)
    rr, dr = _col(t, "RA_rate"), _col(t, "DEC_rate")
    ra_rate = np.asarray(u.Quantity(rr).to_value(u.arcsec / u.hour), float) if rr is not None else np.zeros(len(t))
    dec_rate = np.asarray(u.Quantity(dr).to_value(u.arcsec / u.hour), float) if dr is not None else np.zeros(len(t))
    vmag = _col(t, "V")
    num = _col(t, "Number")
    typ = _col(t, "Type")
    unc = _col(t, "posunc")
    epoch_mjd = float(t.meta.get("query_mjd", frames[0].mjd))

    out = []
    for i in range(len(t)):
        pos = []
        inside = False
        for f in frames:
            dt_h = (f.mjd - epoch_mjd) * 24.0
            d = dec[i] + dec_rate[i] * dt_h / 3600.0
            r = ra[i] + ra_rate[i] * dt_h / 3600.0 / max(np.cos(np.radians(dec[i])), 1e-6)
            if solutions is not None:
                x, y = solutions[f.index].sky2pix(r, d)
                x, y = float(x[0]), float(y[0])
            else:
                x, y = (float(v) for v in f.wcs.all_world2pix(r, d, 0))
            ny, nx = f.shape
            inside = inside or (0 <= x < nx and 0 <= y < ny)
            pos.append({"frame": f.index, "ra": float(r), "dec": float(d), "x": x, "y": y})
        if not inside:
            continue
        number = None
        try:
            n = num[i] if num is not None else None
            if n is not None and not np.ma.is_masked(n) and int(n) > 0:
                number = int(n)
        except Exception:  # noqa: BLE001
            number = None

        def _f(col):
            try:
                v = float(u.Quantity(col[i]).value)
                return v if np.isfinite(v) else None
            except Exception:  # noqa: BLE001
                return None

        out.append({
            "name": str(t["Name"][i]).strip(), "number": number,
            "type": str(typ[i]) if typ is not None else None,
            "vmag": _f(vmag) if vmag is not None else None,
            "pos_unc_arcsec": _f(unc) if unc is not None else None,
            "rate_arcsec_min": float(np.hypot(ra_rate[i], dec_rate[i]) / 60.0),
            "pa": float(np.degrees(np.arctan2(ra_rate[i], dec_rate[i])) % 360.0),
            "positions": pos,
        })
    return out
