"""Moving-object detection: static-source suppression + tracklet linking.

A real asteroid appears at a *different* sky position in each frame, moving in a
near-straight line at near-constant rate over the short (~1 hour) span. Stars and
galaxies stay fixed. We remove the fixed sources, then link the survivors into
tracklets consistent with uniform linear motion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u


@dataclass
class Detection:
    frame: int
    time_mjd: float
    x: float
    y: float
    ra: float            # deg
    dec: float           # deg
    mag: float
    flux: float


@dataclass
class Tracklet:
    detections: list[Detection]
    rate_ra: float = 0.0       # arcsec/min (great-circle component, includes cos dec)
    rate_dec: float = 0.0      # arcsec/min
    rate_total: float = 0.0    # arcsec/min
    rms_arcsec: float = 0.0    # straight-line fit residual
    pa_deg: float = 0.0        # position angle of motion
    known_name: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.detections)


def suppress_static(frames_dets, tol_arcsec=2.0, min_frames=None):
    """Drop detections that recur at the same sky position across many frames."""
    all_d = [d for fr in frames_dets for d in fr]
    n_frames = len(frames_dets)
    if min_frames is None:
        min_frames = max(2, n_frames - 1)
    if not all_d:
        return []

    coords = SkyCoord([d.ra for d in all_d] * u.deg, [d.dec for d in all_d] * u.deg)
    moving = []
    for i, d in enumerate(all_d):
        sep = coords[i].separation(coords).arcsec
        frames_present = {all_d[j].frame for j in np.where(sep < tol_arcsec)[0]}
        if len(frames_present) < min_frames:
            moving.append(d)
    return moving


def _great_circle_arcsec(ra1, dec1, ra2, dec2):
    cosd = np.cos(np.radians((dec1 + dec2) / 2.0))
    dra = (ra2 - ra1) * cosd
    ddec = dec2 - dec1
    return np.hypot(dra, ddec) * 3600.0


def _fit_tracklet(dets):
    t = np.array([d.time_mjd for d in dets])
    t0 = t.min()
    tm = (t - t0) * 1440.0  # minutes
    ra = np.array([d.ra for d in dets])
    dec = np.array([d.dec for d in dets])

    A = np.vstack([np.ones_like(tm), tm]).T
    cra, *_ = np.linalg.lstsq(A, ra, rcond=None)
    cdec, *_ = np.linalg.lstsq(A, dec, rcond=None)
    ra_fit = A @ cra
    dec_fit = A @ cdec

    cosd = np.cos(np.radians(dec.mean()))
    res = np.hypot((ra - ra_fit) * cosd, dec - dec_fit) * 3600.0
    rms = float(np.sqrt(np.mean(res ** 2)))

    rate_ra = float(cra[1] * 3600.0 * cosd)    # arcsec/min
    rate_dec = float(cdec[1] * 3600.0)
    rate_total = float(np.hypot(rate_ra, rate_dec))
    pa = float((np.degrees(np.arctan2(rate_ra, rate_dec))) % 360.0)
    return rate_ra, rate_dec, rate_total, rms, pa


def link_tracklets(
    moving,
    max_rate_arcsec_min=25.0,
    min_rate_arcsec_min=0.05,
    tol_arcsec=2.5,
    min_points=3,
):
    """Link moving detections into linear-motion tracklets.

    Seeds a velocity from every pair of detections in the two earliest frames,
    predicts positions in the remaining frames, and gathers inliers (a light
    RANSAC). Works well for the small detection counts in 4-image sets.
    """
    by_frame: dict[int, list[Detection]] = {}
    for d in moving:
        by_frame.setdefault(d.frame, []).append(d)
    frames = sorted(by_frame)
    if len(frames) < min(min_points, 2):
        return []

    candidates = []
    seed_a = by_frame[frames[0]]
    seed_b_frames = frames[1:]
    for da in seed_a:
        for fb in seed_b_frames:
            for db in by_frame[fb]:
                dt = (db.time_mjd - da.time_mjd) * 1440.0
                if dt <= 0:
                    continue
                rate = _great_circle_arcsec(da.ra, da.dec, db.ra, db.dec) / dt
                if not (min_rate_arcsec_min <= rate <= max_rate_arcsec_min):
                    continue
                vra = (db.ra - da.ra) / dt      # deg/min
                vdec = (db.dec - da.dec) / dt

                inliers = [da]
                for fc in frames:
                    if fc == da.frame:
                        continue
                    tc = by_frame[fc][0].time_mjd
                    dtc = (tc - da.time_mjd) * 1440.0
                    pra = da.ra + vra * dtc
                    pdec = da.dec + vdec * dtc
                    best, best_sep = None, tol_arcsec
                    for dc in by_frame[fc]:
                        s = _great_circle_arcsec(pra, pdec, dc.ra, dc.dec)
                        if s < best_sep:
                            best, best_sep = dc, s
                    if best is not None:
                        inliers.append(best)

                # one detection per frame, earliest first
                seen, uniq = {}, []
                for d in sorted(inliers, key=lambda d: d.time_mjd):
                    if d.frame not in seen:
                        seen[d.frame] = True
                        uniq.append(d)
                if len(uniq) >= min_points:
                    candidates.append(uniq)

    # Deduplicate overlapping candidates: prefer more points, then lower RMS.
    scored = []
    for dets in candidates:
        rate_ra, rate_dec, rate_total, rms, pa = _fit_tracklet(dets)
        scored.append((len(dets), -rms, dets, (rate_ra, rate_dec, rate_total, rms, pa)))
    scored.sort(reverse=True)

    used_ids = set()
    tracklets = []
    for npts, _negrms, dets, fit in scored:
        key = frozenset((d.frame, round(d.x, 1), round(d.y, 1)) for d in dets)
        ids = set((d.frame, round(d.x, 2), round(d.y, 2)) for d in dets)
        if ids & used_ids:
            continue
        used_ids |= ids
        rate_ra, rate_dec, rate_total, rms, pa = fit
        tracklets.append(
            Tracklet(
                detections=dets,
                rate_ra=rate_ra,
                rate_dec=rate_dec,
                rate_total=rate_total,
                rms_arcsec=rms,
                pa_deg=pa,
            )
        )
    return tracklets
