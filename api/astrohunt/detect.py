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
    snr: float = float("nan")
    fwhm: float = float("nan")      # px
    fit_rms: float = float("nan")
    source: str = "auto"            # "auto" (pipeline) or "manual" (clicked)


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


def _tangent_arcsec(ra, dec, ra0, dec0):
    """Small-field tangent-plane offsets (arcsec) about (ra0, dec0)."""
    ra = np.asarray(ra, float)
    dec = np.asarray(dec, float)
    dra = (ra - ra0 + 180.0) % 360.0 - 180.0
    x = dra * np.cos(np.radians(dec0)) * 3600.0
    y = (dec - dec0) * 3600.0
    return np.column_stack([x, y])


def suppress_static(frames_dets, tol_arcsec=2.0, min_frames=None):
    """Drop detections that recur at the same sky position across many frames."""
    from scipy.spatial import cKDTree

    all_d = [d for fr in frames_dets for d in fr]
    n_frames = len(frames_dets)
    if min_frames is None:
        min_frames = max(2, n_frames - 1)
    if not all_d:
        return []

    ra0 = all_d[0].ra
    dec0 = float(np.mean([d.dec for d in all_d]))
    xy = _tangent_arcsec([d.ra for d in all_d], [d.dec for d in all_d], ra0, dec0)
    frame_of = np.array([d.frame for d in all_d])
    tree = cKDTree(xy)
    moving = []
    for i, nbrs in enumerate(tree.query_ball_point(xy, tol_arcsec)):
        if len(set(frame_of[nbrs].tolist())) < min_frames:
            moving.append(all_d[i])
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
    max_mag_spread=None,
):
    """Link moving detections into linear-motion tracklets.

    Seeds a velocity from every pair of detections in two different frames
    (KD-tree limited to the allowed rate range), predicts positions in the
    remaining frames and gathers inliers (a light RANSAC). Seeding from every
    frame pair means an object missing from the first frame is still found.
    `max_mag_spread` (mag) optionally rejects sets whose brightness is wildly
    inconsistent — a real asteroid barely changes over an hour.
    """
    from scipy.spatial import cKDTree

    by_frame: dict[int, list[Detection]] = {}
    for d in moving:
        by_frame.setdefault(d.frame, []).append(d)
    frames = sorted(by_frame)
    if len(frames) < max(min(min_points, 2), 2):
        return []

    ra0 = moving[0].ra
    dec0 = float(np.mean([d.dec for d in moving]))
    xy = {f: _tangent_arcsec([d.ra for d in by_frame[f]], [d.dec for d in by_frame[f]],
                             ra0, dec0) for f in frames}
    tree = {f: cKDTree(xy[f]) for f in frames}
    tmin = {f: by_frame[f][0].time_mjd * 1440.0 for f in frames}   # minutes

    candidates = {}
    # a seed pair (fa, fb) can only yield >= min_points if enough frames remain
    for ia, fa in enumerate(frames):
        if len(frames) - ia < min_points:
            break
        for fb in frames[ia + 1:]:
            dt = tmin[fb] - tmin[fa]
            if dt <= 0:
                continue
            pairs = tree[fa].query_ball_tree(tree[fb], max_rate_arcsec_min * dt)
            for ka, nbrs in enumerate(pairs):
                pa = xy[fa][ka]
                for kb in nbrs:
                    v = (xy[fb][kb] - pa) / dt          # arcsec/min
                    if np.hypot(v[0], v[1]) < min_rate_arcsec_min:
                        continue
                    members = {fa: ka, fb: kb}
                    for fc in frames:
                        if fc in members:
                            continue
                        pred = pa + v * (tmin[fc] - tmin[fa])
                        dist, kc = tree[fc].query(pred, distance_upper_bound=tol_arcsec)
                        if np.isfinite(dist):
                            members[fc] = int(kc)
                    if len(members) >= min_points:
                        key = tuple(sorted(members.items()))
                        candidates[key] = [by_frame[f][k] for f, k in key]

    # Deduplicate overlapping candidates: prefer more points, then lower RMS.
    scored = []
    for dets in candidates.values():
        rate_ra, rate_dec, rate_total, rms, pa = _fit_tracklet(dets)
        if rms > tol_arcsec:
            continue
        if not (min_rate_arcsec_min <= rate_total <= max_rate_arcsec_min):
            continue
        if max_mag_spread is not None:
            mags = np.array([d.mag for d in dets], float)
            mags = mags[np.isfinite(mags)]
            if mags.size >= 2 and (mags.max() - mags.min()) > max_mag_spread:
                continue
        scored.append((len(dets), -rms, id(dets), dets,
                       (rate_ra, rate_dec, rate_total, rms, pa)))
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)

    used_ids = set()
    tracklets = []
    for _npts, _negrms, _id, dets, fit in scored:
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
