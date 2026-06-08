"""Astrometric reduction via a least-squares plate-constants (LSPC) solution.

The MPC explicitly requires a *proper* LSPC solution and forbids deriving
positions by eye or chart overlay. We project reference-star sky positions to
standard (tangent-plane) coordinates and least-squares fit a polynomial mapping
from pixel coordinates, then invert to get RA/Dec for any source. The fit RMS is
the sub-arcsecond "consistency" metric the MPC actually cares about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u


# --- gnomonic (tangent plane) projection -----------------------------------

def radec_to_standard(ra, dec, ra0, dec0):
    """Project (ra, dec) deg -> standard coords (xi, eta) in radians."""
    ra = np.radians(np.asarray(ra, float))
    dec = np.radians(np.asarray(dec, float))
    ra0r, dec0r = np.radians(ra0), np.radians(dec0)
    cosc = np.sin(dec0r) * np.sin(dec) + np.cos(dec0r) * np.cos(dec) * np.cos(ra - ra0r)
    xi = np.cos(dec) * np.sin(ra - ra0r) / cosc
    eta = (np.cos(dec0r) * np.sin(dec) - np.sin(dec0r) * np.cos(dec) * np.cos(ra - ra0r)) / cosc
    return xi, eta


def standard_to_radec(xi, eta, ra0, dec0):
    """Invert standard coords (radians) -> (ra, dec) deg."""
    xi = np.asarray(xi, float)
    eta = np.asarray(eta, float)
    ra0r, dec0r = np.radians(ra0), np.radians(dec0)
    rho = np.sqrt(xi ** 2 + eta ** 2)
    c = np.arctan(rho)
    sinc, cosc = np.sin(c), np.cos(c)
    safe = np.where(rho == 0, 1.0, rho)
    dec = np.arcsin(cosc * np.sin(dec0r) + np.where(rho == 0, 0.0, eta * sinc * np.cos(dec0r) / safe))
    ra = ra0r + np.arctan2(
        xi * sinc, rho * np.cos(dec0r) * cosc - eta * np.sin(dec0r) * sinc
    )
    return np.degrees(ra) % 360.0, np.degrees(dec)


def _design(x, y, order):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    cols = [np.ones_like(x), x, y]
    if order >= 2:
        cols += [x * x, x * y, y * y]
    if order >= 3:
        cols += [x ** 3, x ** 2 * y, x * y ** 2, y ** 3]
    return np.vstack(cols).T


def _circmean(ra):
    a = np.radians(np.asarray(ra, float))
    return np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360.0


@dataclass
class PlateSolution:
    ra0: float
    dec0: float
    order: int
    cx: np.ndarray            # pixel -> xi coefficients
    cy: np.ndarray            # pixel -> eta coefficients
    rms_arcsec: float         # consistency of the fit (the MPC metric)
    n_stars: int
    mag_zeropoint: float = np.nan   # cat_mag = mag_inst + zp, if calibrated

    def pix2sky(self, x, y):
        A = _design(x, y, self.order)
        xi = A @ self.cx
        eta = A @ self.cy
        ra, dec = standard_to_radec(xi, eta, self.ra0, self.dec0)
        return ra, dec

    def calibrate_mag(self, mag_inst):
        if not np.isfinite(self.mag_zeropoint):
            return np.full(np.shape(mag_inst), np.nan)
        return np.asarray(mag_inst, float) + self.mag_zeropoint


def match_to_catalog(det_x, det_y, wcs, cat_ra, cat_dec, tol_arcsec=3.0):
    """Match detected (x, y) to catalogue stars using the header WCS as a guess.

    Returns boolean mask over detections, the matched catalogue indices, and
    on-sky separations (arcsec).
    """
    ra_g, dec_g = wcs.all_pix2world(np.asarray(det_x), np.asarray(det_y), 0)
    det = SkyCoord(ra_g * u.deg, dec_g * u.deg)
    cat = SkyCoord(np.asarray(cat_ra) * u.deg, np.asarray(cat_dec) * u.deg)
    idx, sep, _ = det.match_to_catalog_sky(cat)
    sep_arcsec = sep.arcsec
    good = sep_arcsec < tol_arcsec
    return good, idx, sep_arcsec


def solve_plate(x, y, ra, dec, ra0=None, dec0=None, order=1):
    """Least-squares plate-constants fit from matched stars -> PlateSolution."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ra = np.asarray(ra, float)
    dec = np.asarray(dec, float)
    if ra0 is None:
        ra0 = _circmean(ra)
    if dec0 is None:
        dec0 = float(np.mean(dec))

    n_terms = _design(np.array([0.0]), np.array([0.0]), order).shape[1]
    if len(x) < n_terms:
        raise ValueError(
            f"Need >= {n_terms} reference stars for an order-{order} fit, got {len(x)}."
        )

    xi, eta = radec_to_standard(ra, dec, ra0, dec0)
    A = _design(x, y, order)
    cx, *_ = np.linalg.lstsq(A, xi, rcond=None)
    cy, *_ = np.linalg.lstsq(A, eta, rcond=None)

    dxi = xi - A @ cx
    deta = eta - A @ cy
    res_rad = np.sqrt(dxi ** 2 + deta ** 2)
    rms_arcsec = float(np.degrees(np.sqrt(np.mean(res_rad ** 2))) * 3600.0)

    return PlateSolution(ra0, dec0, order, cx, cy, rms_arcsec, len(x))


def solve_frame(detections, wcs, cat_ra, cat_dec, cat_mag=None,
                order=1, match_tol_arcsec=3.0):
    """End-to-end per-frame astrometric calibration.

    Matches detected stars to the reference catalogue, runs the LSPC fit, and
    (if catalogue magnitudes are supplied) derives a simple photometric
    zeropoint. Returns (PlateSolution, n_matched).
    """
    det_x = np.asarray(detections["x"], float)
    det_y = np.asarray(detections["y"], float)
    good, idx, sep = match_to_catalog(det_x, det_y, wcs, cat_ra, cat_dec, match_tol_arcsec)
    if good.sum() < 3:
        raise ValueError(
            f"Only {int(good.sum())} star matches within {match_tol_arcsec}\"; "
            f"cannot solve plate. Check WCS guess / catalogue overlap."
        )

    mx = det_x[good]
    my = det_y[good]
    mra = np.asarray(cat_ra)[idx[good]]
    mdec = np.asarray(cat_dec)[idx[good]]
    sol = solve_plate(mx, my, mra, mdec, order=order)

    if cat_mag is not None:
        cm = np.asarray(cat_mag, float)[idx[good]]
        inst = np.asarray(detections["mag_inst"], float)[good]
        valid = np.isfinite(cm) & np.isfinite(inst)
        if valid.sum() >= 3:
            sol.mag_zeropoint = float(np.median(cm[valid] - inst[valid]))
    return sol, int(good.sum())
