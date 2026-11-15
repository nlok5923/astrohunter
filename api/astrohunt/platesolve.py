"""Asterism (triangle) plate-solving for FITS frames lacking a WCS.

Real survey frames usually ship a WCS, but raw amateur/archival frames often
don't. This implements the core idea behind astrometry.net: build rotation/
scale/translation-invariant triangle descriptors from the brightest detected
stars and from a reference catalogue, match them, vote on star correspondences,
and fit a TAN WCS from the agreed pairs. A coarse pointing hint (telescope RA/Dec)
keeps it fast and robust; a fully blind search would need an all-sky index.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points
import astropy.units as u
from scipy.spatial import cKDTree

from .astrometry import radec_to_standard


@dataclass
class SolveResult:
    wcs: WCS
    n_inliers: int
    rms_arcsec: float
    matched_det: np.ndarray   # detected-star indices used
    matched_cat: np.ndarray   # catalogue-star indices used


def _triangle_descriptors(xy, max_pts, min_ratio=0.12):
    """Return (descriptors[T,2], vertex_indices[T,3], longest_side[T]).

    Descriptor = (mid/long, short/long) of the side lengths — invariant to
    rotation, translation, uniform scale, and reflection. Vertices are ordered
    by the length of their opposite side so correspondences line up. Thin/
    near-collinear triangles (short/long < min_ratio) are dropped as unstable.
    """
    pts = np.asarray(xy, float)[:max_pts]
    n = len(pts)
    descs, idxs, longest_sides = [], [], []
    for combo in itertools.combinations(range(n), 3):
        p = pts[list(combo)]
        # side opposite vertex v = distance between the other two vertices
        s = np.array([
            np.hypot(*(p[1] - p[2])),
            np.hypot(*(p[0] - p[2])),
            np.hypot(*(p[0] - p[1])),
        ])
        order = np.argsort(s)          # ascending opposite-side length
        s_sorted = s[order]
        longest = s_sorted[2]
        if longest <= 1e-6 or s_sorted[0] / longest < min_ratio:
            continue
        descs.append((s_sorted[1] / longest, s_sorted[0] / longest))
        idxs.append([combo[order[0]], combo[order[1]], combo[order[2]]])
        longest_sides.append(longest)
    return np.asarray(descs), np.asarray(idxs), np.asarray(longest_sides)


def solve_wcs(
    det_x,
    det_y,
    ref_ra,
    ref_dec,
    hint_ra,
    hint_dec,
    ref_mag=None,
    n_bright_det: int = 35,
    n_bright_cat: int = 70,
    match_eps: float = 0.02,
    field_radius_deg: float = 0.5,
    min_matches: int = 6,
    verify_tol_arcsec: float = 5.0,
) -> SolveResult:
    """Solve a TAN WCS from detected star pixels + a reference catalogue."""
    det_x = np.asarray(det_x, float)
    det_y = np.asarray(det_y, float)
    ref_ra = np.asarray(ref_ra, float)
    ref_dec = np.asarray(ref_dec, float)

    # Project catalogue near the hint into a planar tangent frame (arcsec).
    xi, eta = radec_to_standard(ref_ra, ref_dec, hint_ra, hint_dec)
    cat_xy = np.column_stack([np.degrees(xi) * 3600.0, np.degrees(eta) * 3600.0])
    within = np.hypot(cat_xy[:, 0], cat_xy[:, 1]) < field_radius_deg * 3600.0
    cat_idx_all = np.where(within)[0]
    cat_xy = cat_xy[within]
    cra = ref_ra[within]
    cdec = ref_dec[within]
    if ref_mag is not None:
        order = np.argsort(np.asarray(ref_mag, float)[within])
        cat_xy = cat_xy[order]
        cra = cra[order]
        cdec = cdec[order]
        cat_idx_all = cat_idx_all[order]

    if len(det_x) < 3 or len(cat_xy) < 3:
        raise ValueError("Not enough stars to plate-solve.")

    det_xy = np.column_stack([det_x, det_y])
    d_desc, d_idx, d_long = _triangle_descriptors(det_xy, n_bright_det)
    c_desc, c_idx, c_long = _triangle_descriptors(cat_xy, n_bright_cat)
    if len(d_desc) == 0 or len(c_desc) == 0:
        raise ValueError("Could not build triangle descriptors.")

    # First pass: collect candidate triangle matches with their implied scale
    # (det longest side in px / cat longest side in arcsec). Correct matches all
    # share one scale; false matches scatter — so we keep only the dominant scale.
    tree = cKDTree(c_desc)
    cand = []   # (scale, det_triple, cat_triple)
    for t, desc in enumerate(d_desc):
        for m in tree.query_ball_point(desc, match_eps):
            if c_long[m] > 0:
                cand.append((d_long[t] / c_long[m], d_idx[t], c_idx[m]))
    if not cand:
        raise ValueError("No triangle matches; check hint/scale.")

    scales = np.array([c[0] for c in cand])
    logs = np.log(scales)
    hist, edges = np.histogram(logs, bins=60)
    b = int(np.argmax(hist))
    lo, hi = edges[max(0, b - 1)], edges[min(len(edges) - 1, b + 2)]
    in_peak = (logs >= lo) & (logs <= hi)
    s0 = float(np.exp(np.median(logs[in_peak])))
    keep = np.abs(np.log(scales / s0)) < np.log(1.06)

    votes: dict[tuple[int, int], int] = {}
    for k, (_scale, dtri, ctri) in enumerate(cand):
        if not keep[k]:
            continue
        for r in range(3):
            key = (int(dtri[r]), int(ctri[r]))
            votes[key] = votes.get(key, 0) + 1
    if not votes:
        raise ValueError("No scale-consistent triangle matches.")

    # Greedy one-to-one assignment by descending votes.
    pairs = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    used_d, used_c, corr = set(), set(), []
    for (di, ci), _v in pairs:
        if di in used_d or ci in used_c:
            continue
        used_d.add(di)
        used_c.add(ci)
        corr.append((di, ci))
    if len(corr) < min_matches:
        raise ValueError(f"Only {len(corr)} correspondences (<{min_matches}).")

    corr = np.array(corr)
    px = det_x[corr[:, 0]]
    py = det_y[corr[:, 0]]
    world = SkyCoord(cra[corr[:, 1]] * u.deg, cdec[corr[:, 1]] * u.deg)
    proj = SkyCoord(hint_ra * u.deg, hint_dec * u.deg)

    wcs = fit_wcs_from_points((px, py), world, proj_point=proj, projection="TAN")

    # Verify by reprojecting the full catalogue and counting inliers.
    cat_all = SkyCoord(cra * u.deg, cdec * u.deg)
    cx, cy = wcs.world_to_pixel(cat_all)
    det_tree = cKDTree(det_xy)
    # crude pixel scale to convert tol
    scale = np.median(np.abs(np.diag(wcs.pixel_scale_matrix))) * 3600.0  # arcsec/px
    tol_px = verify_tol_arcsec / max(scale, 1e-6)
    dists, _ = det_tree.query(np.column_stack([cx, cy]))
    n_inliers = int(np.sum(dists < tol_px))

    resid = dists[dists < tol_px] * scale
    rms = float(np.sqrt(np.mean(resid ** 2))) if len(resid) else float("nan")

    if n_inliers < min_matches:
        raise ValueError(
            f"WCS verification weak ({n_inliers} inliers). Solve unreliable."
        )

    return SolveResult(
        wcs=wcs,
        n_inliers=n_inliers,
        rms_arcsec=rms,
        matched_det=corr[:, 0],
        matched_cat=cat_idx_all[corr[:, 1]],
    )


def solve_fits(in_path, ref_catalog, hint_radec=None, out_path=None,
               fwhm=3.5, threshold_sigma=5.0, **kwargs):
    """Plate-solve a FITS file and write the WCS into its header.

    `ref_catalog` is a ReferenceCatalog. `hint_radec` overrides the header
    pointing hint. Returns (SolveResult, out_path).
    """
    from astropy.io import fits

    from .extract import extract_sources

    with fits.open(in_path) as hdul:
        h = hdul[0]
        if h.data is None:
            for cand in hdul:
                if cand.data is not None and cand.data.ndim >= 2:
                    h = cand
                    break
        data = np.asarray(h.data, float)
        header = h.header.copy()

    hint = hint_radec or hint_from_header(header)
    if hint is None:
        raise ValueError(
            "No pointing hint: pass hint_radec or add RA/DEC keywords to the header."
        )

    srcs = extract_sources(data, fwhm=fwhm, threshold_sigma=threshold_sigma)
    res = solve_wcs(
        srcs["x"], srcs["y"], ref_catalog.ra, ref_catalog.dec,
        hint[0], hint[1], ref_mag=ref_catalog.mag, **kwargs,
    )

    header.update(res.wcs.to_header())
    out_path = out_path or in_path
    fits.writeto(out_path, data.astype(np.float32), header, overwrite=True)
    return res, out_path


def hint_from_header(header):
    """Best-effort pointing hint from common telescope keywords. None if absent."""
    for ra_key, dec_key in (("CRVAL1", "CRVAL2"), ("RA", "DEC"),
                            ("OBJCTRA", "OBJCTDEC"), ("RA_OBJ", "DEC_OBJ")):
        if ra_key in header and dec_key in header:
            ra, dec = header[ra_key], header[dec_key]
            try:
                if isinstance(ra, str) and (":" in ra or " " in ra):
                    c = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
                    return float(c.ra.deg), float(c.dec.deg)
                return float(ra), float(dec)
            except Exception:
                continue
    return None


# --- refining an approximate header WCS ("automatic reference star match") ----

def _offset_peak(dx, dy, limit, bin_px):
    n = int(2 * limit // bin_px)
    H, xe, ye = np.histogram2d(dx.ravel(), dy.ravel(), bins=n,
                               range=[[-limit, limit], [-limit, limit]])
    # smooth over 2x2 bins so a peak straddling a bin edge isn't split
    H2 = H[:-1, :-1] + H[1:, :-1] + H[:-1, 1:] + H[1:, 1:]
    i, j = np.unravel_index(int(H2.argmax()), H2.shape)
    return float(H2[i, j]), float(xe[i + 1]), float(ye[j + 1]), float(H2.mean())


def refine_wcs(det_x, det_y, wcs, cat_ra, cat_dec, cat_mag=None, *,
               n_bright=400, max_shift_px=400.0, final_tol_px=2.5):
    """Lock an approximate WCS onto the reference catalogue.

    Survey/observatory headers often carry a WCS that is good to a few tens of
    arcseconds and a fraction of a degree in rotation — too coarse for nearest-
    neighbour star matching. We (1) find the pixel shift (and, if needed, a
    small rotation or a completely different orientation) that makes the most
    detected stars coincide with projected catalogue stars, then (2) iterate
    match -> linear fit with a shrinking tolerance, and (3) return a clean TAN
    WCS fitted to the matched pairs.

    Returns (wcs, n_matched, rms_px). Raises ValueError if no orientation gives
    a significant match.
    """
    from astropy.coordinates import SkyCoord
    from astropy.wcs.utils import fit_wcs_from_points
    from scipy.spatial import cKDTree
    import astropy.units as u

    det_x = np.asarray(det_x, float)
    det_y = np.asarray(det_y, float)
    cat_ra = np.asarray(cat_ra, float)
    cat_dec = np.asarray(cat_dec, float)
    order = np.argsort(cat_mag) if cat_mag is not None else np.arange(len(cat_ra))
    bright = order[:n_bright]
    dx_ = det_x[:n_bright]
    dy_ = det_y[:n_bright]

    cx, cy = wcs.all_world2pix(cat_ra, cat_dec, 0)
    x0 = float(np.median(det_x))
    y0 = float(np.median(det_y))

    def rotated(ang_deg, parity):
        a = np.radians(ang_deg)
        rx, ry = cx - x0, cy - y0
        if parity < 0:
            rx = -rx
        return (x0 + rx * np.cos(a) - ry * np.sin(a), y0 + rx * np.sin(a) + ry * np.cos(a))

    def score(px, py, bin_px=6.0):
        ddx = dx_[:, None] - px[bright][None, :]
        ddy = dy_[:, None] - py[bright][None, :]
        return _offset_peak(ddx, ddy, max_shift_px, bin_px)

    def significant(pk, mean):
        # Poisson chance level of the busiest cell, with a healthy margin
        return pk >= max(8.0, mean + 7.0 * np.sqrt(mean) + 4.0)

    trials = [(0.0, 1)]
    trials += [(a, 1) for a in np.arange(-3.0, 3.01, 0.25) if a != 0.0]
    best = None
    for ang, par in trials:
        px, py = rotated(ang, par)
        pk, sx, sy, mean = score(px, py)
        if best is None or pk > best[0]:
            best = (pk, ang, par, sx, sy, mean)
    if not significant(best[0], best[5]):       # header orientation is wrong: scan everything
        for par in (1, -1):
            for ang in np.arange(0.0, 360.0, 0.5):
                px, py = rotated(ang, par)
                pk, sx, sy, mean = score(px, py)
                if pk > best[0]:
                    best = (pk, ang, par, sx, sy, mean)
    if not significant(best[0], best[5]):
        raise ValueError(
            "Automatic reference-star match failed: the image stars do not line up "
            "with the catalogue in any orientation. Check the field centre / pixel scale."
        )

    _pk, ang, par, sx, sy, _mean = best
    px, py = rotated(ang, par)
    px = px + sx
    py = py + sy

    # iterate: match detections to the predicted catalogue pixels, fit affine
    tree_tol = [12.0, 6.0, final_tol_px, final_tol_px]
    A = None
    mi = mj = None
    for tol in tree_tol:
        tree = cKDTree(np.column_stack([px, py]))
        dist, j = tree.query(np.column_stack([det_x, det_y]), distance_upper_bound=tol)
        ok = np.isfinite(dist)
        if ok.sum() < 6:
            raise ValueError("Automatic reference-star match failed: too few stars matched.")
        mi, mj = np.where(ok)[0], j[ok]
        # affine map: catalogue header-pixels (cx, cy) -> true detector pixels
        G = np.column_stack([np.ones(len(mj)), cx[mj], cy[mj]])
        ax, *_ = np.linalg.lstsq(G, det_x[mi], rcond=None)
        ay, *_ = np.linalg.lstsq(G, det_y[mi], rcond=None)
        A = (ax, ay)
        Gall = np.column_stack([np.ones(len(cx)), cx, cy])
        px, py = Gall @ ax, Gall @ ay

    res = np.hypot(det_x[mi] - px[mj], det_y[mi] - py[mj])
    keep = res <= max(3.0 * np.median(res), 0.5)
    mi, mj = mi[keep], mj[keep]
    sky = SkyCoord(cat_ra[mj] * u.deg, cat_dec[mj] * u.deg)
    new_wcs = fit_wcs_from_points((det_x[mi], det_y[mi]), sky, projection="TAN")
    fx, fy = new_wcs.all_world2pix(cat_ra[mj], cat_dec[mj], 0)
    rms_px = float(np.sqrt(np.mean((fx - det_x[mi]) ** 2 + (fy - det_y[mi]) ** 2)))
    del A
    return new_wcs, int(len(mi)), rms_px
