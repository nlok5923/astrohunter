"""Generate synthetic 4-image asteroid-search sets for testing and demos.

Produces FITS frames with a valid TAN WCS, Gaussian-PSF stars, realistic noise,
and one injected moving point source, plus a truth.json (the star reference
catalogue + the asteroid's per-frame positions). Fully offline and reproducible.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS


@dataclass
class SyntheticConfig:
    n_frames: int = 4
    size: int = 512                  # pixels (square)
    pixel_scale_arcsec: float = 1.5
    center_ra: float = 187.5
    center_dec: float = 12.0
    n_stars: int = 90
    fwhm_px: float = 3.2
    background: float = 120.0
    read_noise: float = 6.0
    gain: float = 1.0
    star_flux_range: tuple = (2.0e3, 8.0e4)
    star_mag_zeropoint: float = 25.0
    asteroid_flux: float = 1.2e4
    asteroid_rate_arcsec_min: float = 1.6   # total on-sky rate
    asteroid_pa_deg: float = 55.0           # direction of motion
    cadence_min: float = 15.0
    start_iso: str = "2026-06-08T10:00:00"
    n_cosmic_rays: int = 0          # per frame, at random positions (test vetting)
    cosmic_ray_flux: float = 5.0e4
    seed: int = 7


def _make_wcs(cfg: SyntheticConfig) -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [cfg.size / 2.0 + 0.5, cfg.size / 2.0 + 0.5]
    scale = cfg.pixel_scale_arcsec / 3600.0
    w.wcs.cdelt = [-scale, scale]
    w.wcs.crval = [cfg.center_ra, cfg.center_dec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def _add_psf(img, x, y, flux, fwhm):
    sigma = fwhm / 2.3548
    h, wd = img.shape
    r = int(np.ceil(4 * sigma))
    xi0, xi1 = max(0, int(x) - r), min(wd, int(x) + r + 1)
    yi0, yi1 = max(0, int(y) - r), min(h, int(y) + r + 1)
    if xi0 >= xi1 or yi0 >= yi1:
        return
    yy, xx = np.mgrid[yi0:yi1, xi0:xi1]
    g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    g *= flux / (2 * np.pi * sigma ** 2)
    img[yi0:yi1, xi0:xi1] += g


def generate(cfg: SyntheticConfig, outdir: str, strip_wcs: bool = False):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    wcs = _make_wcs(cfg)
    t0 = Time(cfg.start_iso)

    # Reference stars: random pixel positions -> sky via WCS (the "true" catalogue).
    margin = 16
    sx = rng.uniform(margin, cfg.size - margin, cfg.n_stars)
    sy = rng.uniform(margin, cfg.size - margin, cfg.n_stars)
    sflux = rng.uniform(*cfg.star_flux_range, cfg.n_stars)
    sra, sdec = wcs.all_pix2world(sx, sy, 0)
    smag = cfg.star_mag_zeropoint - 2.5 * np.log10(sflux)

    # Asteroid motion (arcsec/min split into RA*cos(dec) and Dec components).
    pa = np.radians(cfg.asteroid_pa_deg)
    rate = cfg.asteroid_rate_arcsec_min
    cosd = np.cos(np.radians(cfg.center_dec))
    v_ra = (rate * np.sin(pa)) / 3600.0 / cosd   # deg/min in RA
    v_dec = (rate * np.cos(pa)) / 3600.0          # deg/min in Dec
    a_ra0 = cfg.center_ra + rng.uniform(-0.02, 0.02)
    a_dec0 = cfg.center_dec + rng.uniform(-0.02, 0.02)

    truth = {
        "center": {"ra": cfg.center_ra, "dec": cfg.center_dec},
        "pixel_scale_arcsec": cfg.pixel_scale_arcsec,
        "stars": [{"ra": float(r), "dec": float(d), "mag": float(m)}
                  for r, d, m in zip(sra, sdec, smag)],
        "asteroid": {"rate_arcsec_min": rate, "pa_deg": cfg.asteroid_pa_deg, "frames": []},
    }

    paths = []
    for i in range(cfg.n_frames):
        t = t0 + TimeDelta(i * cfg.cadence_min * 60.0, format="sec")
        dt = i * cfg.cadence_min
        a_ra = a_ra0 + v_ra * dt
        a_dec = a_dec0 + v_dec * dt
        ax, ay = wcs.all_world2pix(a_ra, a_dec, 0)

        img = np.full((cfg.size, cfg.size), cfg.background, dtype=float)
        for x, y, fl in zip(sx, sy, sflux):
            _add_psf(img, x, y, fl, cfg.fwhm_px)
        in_frame = (0 <= ax < cfg.size) and (0 <= ay < cfg.size)
        if in_frame:
            _add_psf(img, float(ax), float(ay), cfg.asteroid_flux, cfg.fwhm_px)

        # Poisson photon noise + Gaussian read noise.
        img = rng.poisson(np.clip(img, 0, None)).astype(float)
        img += rng.normal(0.0, cfg.read_noise, img.shape)

        # Cosmic-ray streaks: sharp, elongated, at random positions per frame.
        for _ in range(cfg.n_cosmic_rays):
            cx = rng.integers(10, cfg.size - 10)
            cy = rng.integers(10, cfg.size - 10)
            length = rng.integers(3, 6)
            ang = rng.uniform(0, np.pi)
            for step in range(length):
                px = int(cx + step * np.cos(ang))
                py = int(cy + step * np.sin(ang))
                if 0 <= px < cfg.size and 0 <= py < cfg.size:
                    img[py, px] += cfg.cosmic_ray_flux * (1.0 - 0.1 * step)

        if strip_wcs:
            from astropy.io.fits import Header

            hdr = Header()
            # Keep a coarse pointing hint (as a real telescope header would).
            hdr["RA"] = cfg.center_ra
            hdr["DEC"] = cfg.center_dec
        else:
            hdr = wcs.to_header()
        hdr["DATE-OBS"] = t.isot
        hdr["EXPTIME"] = 60.0
        hdr["OBJECT"] = "synthetic_field"
        hdr["TELESCOP"] = "astrohunt-synth"
        hdr["GAIN"] = cfg.gain
        path = os.path.join(outdir, f"frame_{i:02d}.fits")
        fits.writeto(path, img.astype(np.float32), hdr, overwrite=True)
        paths.append(path)

        truth["asteroid"]["frames"].append({
            "frame": i, "time_iso": t.isot, "mjd": float(t.mjd),
            "ra": float(a_ra), "dec": float(a_dec),
            "x": float(ax), "y": float(ay), "in_frame": bool(in_frame),
        })

    with open(os.path.join(outdir, "truth.json"), "w") as f:
        json.dump(truth, f, indent=2)
    return paths, os.path.join(outdir, "truth.json")
