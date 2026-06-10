"""Source extraction with sub-pixel centroiding (photutils, CPU-only)."""

from __future__ import annotations

import warnings

import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from photutils.centroids import centroid_com, centroid_sources
from photutils.detection import DAOStarFinder

try:
    from photutils.utils.exceptions import NoDetectionsWarning
except Exception:  # pragma: no cover
    NoDetectionsWarning = Warning


def extract_sources(
    data: np.ndarray,
    fwhm: float = 3.5,
    threshold_sigma: float = 5.0,
    refine: bool = True,
    max_sources: int | None = 2000,
) -> Table:
    """Detect point sources and return sub-pixel centroids + brightness.

    Returns a table with columns: id, x, y, flux, peak, mag_inst, snr.
    Centroid precision drives downstream astrometric accuracy, so we refine
    DAOStarFinder peaks with a centre-of-mass fit on small cutouts.
    """
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    if not np.isfinite(std) or std <= 0:
        std = float(np.std(data)) or 1.0

    finder = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoDetectionsWarning)
        tbl = finder(data - median)
    if tbl is None or len(tbl) == 0:
        return Table(
            names=["id", "x", "y", "flux", "peak", "mag_inst", "snr"],
            dtype=[int, float, float, float, float, float, float],
        )

    x = np.asarray(tbl["xcentroid"], dtype=float)
    y = np.asarray(tbl["ycentroid"], dtype=float)

    if refine and len(x):
        box = int(2 * round(fwhm) + 1)
        try:
            xr, yr = centroid_sources(
                data - median, x, y, box_size=box, centroid_func=centroid_com
            )
            good = np.isfinite(xr) & np.isfinite(yr)
            x = np.where(good, xr, x)
            y = np.where(good, yr, y)
        except Exception:
            pass  # keep DAOStarFinder centroids if refinement fails

    flux = np.asarray(tbl["flux"], dtype=float)
    peak = np.asarray(tbl["peak"], dtype=float)
    mag_inst = -2.5 * np.log10(np.clip(flux, 1e-9, None))
    snr = peak / std

    out = Table()
    out["id"] = np.arange(len(x))
    out["x"] = x
    out["y"] = y
    out["flux"] = flux
    out["peak"] = peak
    out["mag_inst"] = mag_inst
    out["snr"] = snr

    out.sort("flux")
    out.reverse()  # brightest first
    if max_sources is not None and len(out) > max_sources:
        out = out[:max_sources]
    out["id"] = np.arange(len(out))
    return out
