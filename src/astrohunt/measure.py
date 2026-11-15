"""Manual object measurement — the Astrometrica "click on the object" workflow.

Given a rough pixel position (a mouse click), find the nearest point source,
fit a 2-D Gaussian PSF + flat background to it and report what Astrometrica's
Object Verification window shows: sub-pixel centroid, FWHM, peak SNR, fit RMS
and an aperture flux. The caller turns the centroid into RA/Dec with the frame's
plate solution and the flux into a magnitude with the frame's zero point.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass
class Measurement:
    x: float                 # 0-based pixel centroid (FITS array coords)
    y: float
    fwhm: float              # px (geometric mean of the two axes)
    elongation: float        # major/minor axis ratio
    peak: float              # fitted amplitude above background (ADU)
    background: float        # local sky level (ADU)
    sky_sigma: float         # local sky noise per pixel (ADU)
    snr: float               # peak / sky noise (what Astrometrica reports)
    flux: float              # background-subtracted aperture sum (ADU)
    flux_snr: float          # aperture flux / its noise
    fit_rms: float           # PSF-fit residual RMS as a fraction of the peak
    ok: bool                 # fit converged on a plausible point source
    note: str = ""

    def to_dict(self):
        d = asdict(self)
        return {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                for k, v in d.items()}


def local_sky(data, x, y, r_in, r_out):
    """Sigma-clipped median + robust sigma in an annulus around (x, y)."""
    h, w = data.shape
    x0, x1 = max(0, int(x - r_out)), min(w, int(x + r_out) + 2)
    y0, y1 = max(0, int(y - r_out)), min(h, int(y + r_out) + 2)
    cut = data[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r = np.hypot(xx - x, yy - y)
    vals = cut[(r >= r_in) & (r <= r_out) & np.isfinite(cut)]
    if vals.size < 8:
        vals = cut[np.isfinite(cut)]
    if vals.size == 0:
        return 0.0, 1.0
    med = np.median(vals)
    sig = 1.4826 * np.median(np.abs(vals - med))
    for _ in range(3):  # clip stars/neighbours out of the annulus
        keep = np.abs(vals - med) < 3.0 * max(sig, 1e-9)
        if keep.sum() < 8:
            break
        vals = vals[keep]
        med = np.median(vals)
        sig = 1.4826 * np.median(np.abs(vals - med))
    if not np.isfinite(sig) or sig <= 0:
        sig = float(np.std(vals)) or 1.0
    return float(med), float(sig)


def aperture_flux(data, x, y, radius, bg):
    """Background-subtracted sum in a circular aperture (with edge sub-sampling)."""
    h, w = data.shape
    x0, x1 = max(0, int(np.floor(x - radius - 1))), min(w, int(np.ceil(x + radius + 2)))
    y0, y1 = max(0, int(np.floor(y - radius - 1))), min(h, int(np.ceil(y + radius + 2)))
    cut = data[y0:y1, x0:x1]
    if cut.size == 0:
        return float("nan"), 0.0
    sub = 5
    offs = (np.arange(sub) + 0.5) / sub - 0.5
    yy, xx = np.mgrid[y0:y1, x0:x1]
    weight = np.zeros(cut.shape, float)
    for oy in offs:
        for ox in offs:
            weight += (np.hypot(xx + ox - x, yy + oy - y) <= radius)
    weight /= sub * sub
    good = np.isfinite(cut)
    flux = float(np.sum(np.where(good, cut - bg, 0.0) * weight))
    return flux, float(weight[good].sum())


def _gauss(p, xx, yy):
    amp, x0, y0, sx, sy, theta, bg = p
    ct, st = np.cos(theta), np.sin(theta)
    xr = (xx - x0) * ct + (yy - y0) * st
    yr = -(xx - x0) * st + (yy - y0) * ct
    return bg + amp * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))


def measure_at(data, x, y, *, fwhm_guess=3.5, search_radius=None,
               aperture_radius=None, snap=True) -> Measurement:
    """Measure the point source nearest to a clicked position.

    `search_radius` (px) limits how far the centroid may snap from the click;
    `aperture_radius` (px) is the photometric aperture (default 1.5 x FWHM guess,
    matching how the frame's zero point is derived).
    """
    h, w = data.shape
    fwhm_guess = float(max(fwhm_guess, 1.2))
    if search_radius is None:
        search_radius = max(3.0, 1.2 * fwhm_guess)
    if aperture_radius is None:
        aperture_radius = max(2.0, 1.5 * fwhm_guess)
    r_in = max(aperture_radius + 2.0, 2.5 * fwhm_guess)
    r_out = r_in + max(6.0, 2.0 * fwhm_guess)

    def fail(note):
        return Measurement(float(x), float(y), float("nan"), float("nan"), float("nan"),
                           float("nan"), float("nan"), 0.0, float("nan"), 0.0,
                           float("nan"), False, note)

    if not (0 <= x < w and 0 <= y < h):
        return fail("outside image")

    bg, sig = local_sky(data, x, y, r_in, r_out)

    # snap to the brightest (lightly smoothed) pixel near the click
    px, py = float(x), float(y)
    if snap:
        sr = int(np.ceil(search_radius))
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(1, xi - sr), min(w - 1, xi + sr + 1)
        y0, y1 = max(1, yi - sr), min(h - 1, yi + sr + 1)
        if x1 <= x0 or y1 <= y0:
            return fail("too close to the image edge")
        best = -np.inf
        for yy_ in range(y0, y1):
            for xx_ in range(x0, x1):
                if np.hypot(xx_ - x, yy_ - y) > search_radius:
                    continue
                v = np.nanmean(data[yy_ - 1:yy_ + 2, xx_ - 1:xx_ + 2])
                if v > best:
                    best, px, py = v, float(xx_), float(yy_)

    half = int(np.ceil(max(4.0, 1.6 * fwhm_guess)))
    xi, yi = int(round(px)), int(round(py))
    x0, x1 = max(0, xi - half), min(w, xi + half + 1)
    y0, y1 = max(0, yi - half), min(h, yi + half + 1)
    cut = np.asarray(data[y0:y1, x0:x1], float)
    if cut.shape[0] < 5 or cut.shape[1] < 5:
        return fail("too close to the image edge")
    good = np.isfinite(cut)
    if good.sum() < 0.6 * cut.size:
        return fail("masked / blank pixels")
    yy, xx = np.mgrid[y0:y1, x0:x1]
    xx = xx.astype(float)
    yy = yy.astype(float)

    amp0 = float(np.nanmax(cut) - bg)
    if not np.isfinite(amp0) or amp0 <= 0:
        return fail("no signal above background")
    s0 = fwhm_guess / 2.3548
    p0 = [amp0, px, py, s0, s0, 0.0, bg]
    lo = [0.0, px - half, py - half, 0.3, 0.3, -np.pi, -np.inf]
    hi = [np.inf, px + half, py + half, 3.0 * half, 3.0 * half, np.pi, np.inf]

    def resid(p):
        return (_gauss(p, xx, yy) - cut)[good]

    try:
        fit = least_squares(resid, p0, bounds=(lo, hi), max_nfev=200)
        p = fit.x
    except Exception:  # noqa: BLE001
        return fail("PSF fit failed")

    amp, cx, cy, sx, sy, _theta, fbg = p
    fwhm = 2.3548 * float(np.sqrt(sx * sy))
    elong = float(max(sx, sy) / max(min(sx, sy), 1e-6))
    res = resid(p)
    fit_rms = float(np.sqrt(np.mean(res ** 2)) / amp) if amp > 0 else float("nan")

    flux, npix = aperture_flux(data, cx, cy, aperture_radius, bg)
    # noise: sky in the aperture + Poisson of the source (unit gain assumed)
    var = npix * sig ** 2 + max(flux, 0.0)
    flux_snr = float(flux / np.sqrt(var)) if var > 0 else 0.0
    snr = float(amp / sig) if sig > 0 else 0.0

    note = ""
    ok = True
    if np.hypot(cx - x, cy - y) > search_radius + 2.0:
        ok, note = False, "fit wandered away from the click"
    elif snr < 3.0:
        ok, note = False, "too faint (SNR < 3)"
    elif fwhm < 0.9:
        ok, note = False, "sharper than a star — hot pixel / cosmic ray?"
    elif fwhm > 3.0 * fwhm_guess:
        ok, note = False, "much broader than a star"
    elif elong > 2.5:
        note = "elongated — trail or blend?"

    return Measurement(float(cx), float(cy), fwhm, elong, float(amp), float(bg),
                       float(sig), snr, float(flux), flux_snr, fit_rms, ok, note)


def estimate_fwhm(data, xs, ys, n=40, fwhm_guess=3.5):
    """Median fitted FWHM of the brightest unsaturated sources (image seeing)."""
    vals = []
    for x, y in list(zip(xs, ys))[:n]:
        m = measure_at(data, float(x), float(y), fwhm_guess=fwhm_guess, snap=False)
        if m.ok and m.elongation < 1.5 and np.isfinite(m.fwhm):
            vals.append(m.fwhm)
    return float(np.median(vals)) if len(vals) >= 3 else float(fwhm_guess)
