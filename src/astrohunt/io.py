"""FITS ingestion: load a set of frames of the same field with WCS + timestamps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS


@dataclass
class Frame:
    """A single calibrated science frame."""

    index: int
    path: str
    data: np.ndarray
    wcs: WCS
    time: Time           # mid-exposure time (UTC)
    exptime: float       # seconds
    header: fits.Header

    @property
    def mjd(self) -> float:
        return float(self.time.mjd)

    @property
    def shape(self):
        return self.data.shape


@dataclass
class FrameSet:
    frames: list[Frame]

    def __len__(self):
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    @property
    def center_radec(self):
        """RA/Dec (deg) of the field centre, from the first frame's WCS."""
        f = self.frames[0]
        ny, nx = f.data.shape
        ra, dec = f.wcs.all_pix2world(nx / 2.0, ny / 2.0, 0)
        return float(ra), float(dec)

    @property
    def pixel_scale_arcsec(self) -> float:
        """Approximate pixel scale (arcsec/pixel) from the first frame's WCS."""
        f = self.frames[0]
        try:
            from astropy.wcs.utils import proj_plane_pixel_scales

            scales = proj_plane_pixel_scales(f.wcs)  # deg/pixel
            return float(np.mean(scales) * 3600.0)
        except Exception:
            return 1.0

    @property
    def span_minutes(self) -> float:
        mjds = [f.mjd for f in self.frames]
        return (max(mjds) - min(mjds)) * 1440.0


def _read_time(header: fits.Header) -> Time:
    """Best-effort mid-exposure time. Adds half the exposure to DATE-OBS."""
    for key in ("DATE-OBS", "DATE_OBS", "MJD-OBS", "DATE"):
        if key in header:
            val = header[key]
            try:
                t = Time(val, format="mjd") if key == "MJD-OBS" else Time(val)
            except Exception:
                continue
            exptime = float(header.get("EXPTIME", header.get("EXPOSURE", 0.0)) or 0.0)
            return t + TimeDelta(exptime / 2.0, format="sec")
    raise ValueError("No usable observation-time keyword (DATE-OBS/MJD-OBS) in header.")


def load_frame(path: str | Path, index: int = 0, hdu: int = 0) -> Frame:
    path = str(path)
    with fits.open(path) as hdul:
        h = hdul[hdu]
        # Fall back to the first HDU that actually has image data.
        if h.data is None:
            for cand in hdul:
                if cand.data is not None and cand.data.ndim >= 2:
                    h = cand
                    break
        data = np.asarray(h.data, dtype=float)
        header = h.header
        wcs = WCS(header)
        if not wcs.has_celestial:
            raise ValueError(
                f"{path}: no celestial WCS in header. astrohunt needs a plate "
                f"solution (run a plate-solver first, or use synthetic data)."
            )
        time = _read_time(header)
        exptime = float(header.get("EXPTIME", header.get("EXPOSURE", 0.0)) or 0.0)
    return Frame(
        index=index,
        path=path,
        data=data,
        wcs=wcs.celestial,
        time=time,
        exptime=exptime,
        header=header,
    )


def load_frame_set(paths) -> FrameSet:
    frames = [load_frame(p, index=i) for i, p in enumerate(paths)]
    # Order by time so "frame 0" is genuinely the earliest exposure.
    frames.sort(key=lambda f: f.mjd)
    for i, f in enumerate(frames):
        f.index = i
    if len({f.shape for f in frames}) > 1:
        raise ValueError("All frames must share the same image dimensions.")
    return FrameSet(frames)
