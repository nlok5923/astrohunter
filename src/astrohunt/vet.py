"""Real-bogus vetting: reject artifacts before they become false tracklets.

This is the CPU-only, GPU-free precursor to a CNN real-bogus classifier (the
Phase-2 upgrade). It scores each detection on cheap morphological features and
rejects the obvious artifacts a 4-image set throws up:

  * cosmic rays / hot pixels -> far too sharp/narrow vs the image PSF
  * satellite or trail streaks -> highly elongated
  * blends / extended junk    -> far too broad

The same feature vectors can later train an sklearn model (or feed a CNN); a
helper is provided to auto-label them from a known catalogue (SkyBoT/truth).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.stats import sigma_clipped_stats


@dataclass
class VetStats:
    psf_fwhm: float
    n_input: int
    n_real: int
    fwhm: np.ndarray
    elong: np.ndarray
    sharp: np.ndarray
    valid: np.ndarray


def _cutout(data, x, y, half):
    h, w = data.shape
    xi0, yi0 = int(round(x)) - half, int(round(y)) - half
    xi1, yi1 = xi0 + 2 * half + 1, yi0 + 2 * half + 1
    if xi0 < 0 or yi0 < 0 or xi1 > w or yi1 > h:
        return None
    return data[yi0:yi1, xi0:xi1].astype(float)


def source_features(data, xs, ys, box_half: int = 5, bg: float | None = None):
    """Compute (fwhm, elongation, sharpness, npix, valid) per source via moments."""
    if bg is None:
        _, bg, _ = sigma_clipped_stats(data, sigma=3.0)

    n = len(xs)
    fwhm = np.full(n, np.nan)
    elong = np.full(n, np.nan)
    sharp = np.full(n, np.nan)
    npix = np.zeros(n, int)
    valid = np.zeros(n, bool)

    for i, (x, y) in enumerate(zip(xs, ys)):
        c = _cutout(data, x, y, box_half)
        if c is None:
            continue
        cc = np.clip(c - bg, 0, None)
        tot = cc.sum()
        if tot <= 0:
            continue
        ny, nx = cc.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        xc = (xx * cc).sum() / tot
        yc = (yy * cc).sum() / tot
        mxx = ((xx - xc) ** 2 * cc).sum() / tot
        myy = ((yy - yc) ** 2 * cc).sum() / tot
        mxy = ((xx - xc) * (yy - yc) * cc).sum() / tot

        tr = mxx + myy
        det = mxx * myy - mxy ** 2
        disc = max(tr ** 2 / 4 - det, 0.0)
        l1 = max(tr / 2 + np.sqrt(disc), 1e-6)
        l2 = max(tr / 2 - np.sqrt(disc), 1e-6)

        peak = cc.max()
        fwhm[i] = 2.3548 * np.sqrt(max(tr / 2.0, 1e-6))
        elong[i] = np.sqrt(l1 / l2)
        sharp[i] = peak / tot
        npix[i] = int((cc > 0.5 * peak).sum())
        valid[i] = True

    return fwhm, elong, sharp, npix, valid


def vet_sources(
    data,
    srcs,
    psf_fwhm: float | None = None,
    fwhm_lo: float = 0.45,
    fwhm_hi: float = 2.6,
    elong_max: float = 2.6,
    min_npix: int = 2,
):
    """Return (real_mask, VetStats). `srcs` is an extract.extract_sources table."""
    xs = np.asarray(srcs["x"], float)
    ys = np.asarray(srcs["y"], float)
    fwhm, elong, sharp, npix, valid = source_features(data, xs, ys)

    if psf_fwhm is None:
        roundish = valid & np.isfinite(fwhm) & (elong < 1.6)
        ref = fwhm[roundish] if roundish.sum() >= 5 else fwhm[valid]
        psf_fwhm = float(np.nanmedian(ref)) if ref.size else 3.0

    real = (
        valid
        & np.isfinite(fwhm)
        & (fwhm >= fwhm_lo * psf_fwhm)
        & (fwhm <= fwhm_hi * psf_fwhm)
        & (elong <= elong_max)
        & (npix >= min_npix)
    )
    stats = VetStats(psf_fwhm, len(xs), int(real.sum()), fwhm, elong, sharp, valid)
    return real, stats


def build_training_features(data, srcs, real_ra, real_dec, wcs, tol_px=2.0):
    """Auto-label detections against a known catalogue: feature matrix + labels.

    Positives are detections coincident with a known source (star/asteroid);
    everything else is treated as bogus. Returns (X[n,3], y[n]) for an sklearn
    classifier — the bootstrap path toward a learned (eventually CNN) vetter.
    """
    xs = np.asarray(srcs["x"], float)
    ys = np.asarray(srcs["y"], float)
    fwhm, elong, sharp, npix, valid = source_features(data, xs, ys)

    kx, ky = wcs.world_to_pixel_values(real_ra, real_dec)
    known = np.column_stack([kx, ky])
    labels = np.zeros(len(xs), int)
    for i, (x, y) in enumerate(zip(xs, ys)):
        d = np.hypot(known[:, 0] - x, known[:, 1] - y)
        if d.min() <= tol_px:
            labels[i] = 1
    X = np.column_stack([fwhm, elong, sharp])
    keep = valid & np.all(np.isfinite(X), axis=1)
    return X[keep], labels[keep]
