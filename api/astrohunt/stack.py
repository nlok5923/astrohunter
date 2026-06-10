"""Shift-and-stack / synthetic tracking (digital tracking) — Phase 3.

A faint mover can sit below the single-frame detection threshold yet still be
recoverable: shift each frame along a trial velocity so the object aligns, then
co-add. The aligned signal stays put while the noise averages down, so SNR grows
~sqrt(N) over N frames (stars, moving at the wrong rate, smear away). Sweeping a
grid of (rate, position-angle) hypotheses is "digital tracking"; the research
flagged this as the main route to objects a few magnitudes fainter — though with
only ~4 frames the realistic gain is modest (sqrt(4) = 2x, ~0.75 mag).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import shift as ndshift

from .extract import extract_sources


def velocity_to_pixels_per_min(wcs, center_radec, rate_arcsec_min, pa_deg):
    """Convert an on-sky rate+PA to a pixel-space velocity vector (px/min)."""
    ra0, dec0 = center_radec
    pa = np.radians(pa_deg)
    dra = (rate_arcsec_min * np.sin(pa)) / 3600.0 / np.cos(np.radians(dec0))
    ddec = (rate_arcsec_min * np.cos(pa)) / 3600.0
    x0, y0 = wcs.world_to_pixel_values(ra0, dec0)
    x1, y1 = wcs.world_to_pixel_values(ra0 + dra, dec0 + ddec)
    return float(x1 - x0), float(y1 - y0)


def shift_and_stack(frames, rate_arcsec_min, pa_deg, combine="mean",
                    subtract_static=True):
    """Align frames along a trial velocity and co-add. Returns (stack, t0_mjd).

    The per-pixel median across frames models the static sky (stars persist at
    a fixed pixel; a mover does not). Subtracting it before stacking is what
    stops bright stars from smearing into streaks that drown a faint mover.
    """
    frames = sorted(frames, key=lambda f: f.mjd)
    t0 = frames[0].mjd
    f0 = frames[0]
    ny, nx = f0.data.shape
    ra0, dec0 = f0.wcs.all_pix2world(nx / 2.0, ny / 2.0, 0)
    vx, vy = velocity_to_pixels_per_min(f0.wcs, (float(ra0), float(dec0)),
                                        rate_arcsec_min, pa_deg)

    data = [f.data for f in frames]
    if subtract_static:
        static = np.median(np.array(data), axis=0)
        data = [d - static for d in data]

    layers = []
    for f, d in zip(frames, data):
        dt = (f.mjd - t0) * 1440.0  # minutes
        cval = 0.0 if subtract_static else float(sigma_clipped_stats(d, sigma=3.0)[1])
        # shift so a mover at the trial velocity lands back on its t0 position
        layers.append(ndshift(d, shift=(-vy * dt, -vx * dt), order=1,
                              mode="constant", cval=cval))
    cube = np.array(layers)
    stack = np.median(cube, axis=0) if combine == "median" else cube.mean(axis=0)
    return stack, t0


def peak_snr_at(image, x, y, box=4):
    """Local peak SNR at (x, y), using a robust background in a surrounding box."""
    h, w = image.shape
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - box), min(w, xi + box + 1)
    y0, y1 = max(0, yi - box), min(h, yi + box + 1)
    cut = image[y0:y1, x0:x1]
    if cut.size == 0:
        return 0.0
    _, med, std = sigma_clipped_stats(image, sigma=3.0)
    return float((cut.max() - med) / (std if std > 0 else 1.0))


@dataclass
class StackHit:
    rate: float
    pa: float
    x: float
    y: float
    snr: float
    ra: float
    dec: float


def digital_tracking(frames, rates, pas, threshold_sigma=5.0, fwhm=3.2,
                     dedup_px=4.0):
    """Sweep a (rate, PA) grid, stack each, and collect detections in the stacks.

    Returns StackHits sorted by SNR. A faint mover surfaces at the hypothesis
    matching its true motion; spurious smears are weak and rate-inconsistent.
    """
    frames = sorted(frames, key=lambda f: f.mjd)
    f0 = frames[0]
    hits: list[StackHit] = []
    for rate in rates:
        for pa in pas:
            stack, _t0 = shift_and_stack(frames, rate, pa)
            srcs = extract_sources(stack, fwhm=fwhm, threshold_sigma=threshold_sigma,
                                   refine=True)
            for i in range(len(srcs)):
                x, y = float(srcs["x"][i]), float(srcs["y"][i])
                snr = peak_snr_at(stack, x, y)
                ra, dec = f0.wcs.all_pix2world(x, y, 0)
                hits.append(StackHit(rate, pa, x, y, snr, float(ra), float(dec)))

    hits.sort(key=lambda h: h.snr, reverse=True)
    # de-duplicate nearby hits (keep the highest-SNR hypothesis per location)
    kept: list[StackHit] = []
    for h in hits:
        if all(np.hypot(h.x - k.x, h.y - k.y) > dedup_px for k in kept):
            kept.append(h)
    return kept
