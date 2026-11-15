"""Interactive workspace — the Astrometrica session, as a library.

One Workspace = one image set loaded in memory. It mirrors the Astrometrica
workflow the IASC guides teach:

    load images  ->  Data Reduction  ->  blink  ->  (Moving Object Detection |
    Known Object Overlay)  ->  click an object in each image to measure it  ->
    MPC report

Settings play the role of an Astrometrica .cfg file (see PRESETS["ps1"]).
The web API (webapi.py) is a thin wrapper over this class.
"""

from __future__ import annotations

import io as _io
import os
import shutil
import tempfile
import uuid
import zipfile
from dataclasses import asdict, dataclass, field

import numpy as np
from astropy.stats import sigma_clipped_stats

from . import catalog
from .io import load_frame_set
from .measure import measure_at
from .mpcreport import ReportHeader, ReportObject, ReportObs, build_report, check_report
from .pipeline import reduce_frames

FITS_EXT = (".fits", ".fit", ".fts", ".fits.fz", ".fits.gz", ".fit.gz")


@dataclass
class Settings:
    """The equivalent of an Astrometrica configuration (.cfg) file."""

    preset: str = "generic"
    # --- report header -----------------------------------------------------
    obs_code: str = "XXX"
    contact: str = ""
    observers: str = ""
    measurers: str = ""
    telescope: str = ""
    net: str = "Gaia DR3"
    band: str = "G"
    ack: str = ""
    ac2: str = ""
    auto_ack: bool = False              # add Astrometrica's "ACK MPCReport file updated ..." line
    extra_digit: bool = False           # 0.000001 d / 0.001 s / 0.01" precision in the report
    # --- data reduction ----------------------------------------------------
    fwhm_px: float | None = None        # None = measure the seeing per image
    threshold_sigma: float = 4.0
    plate_order: int = 1
    max_sources: int = 8000
    mag_limit: float = 21.0
    use_online: bool = True
    # --- moving object detection --------------------------------------------
    static_tol_arcsec: float = 1.0
    link_tol_arcsec: float = 0.8
    min_rate_arcsec_min: float = 0.03
    max_rate_arcsec_min: float = 5.0
    min_points: int = 3
    max_mag_spread: float = 1.0
    saturation: float = 60000.0         # ADU; brighter peaks are not measured
    # --- display -------------------------------------------------------------
    flip_x: bool = False
    flip_y: bool = True                 # FITS row 0 at the bottom (north up for a normal WCS)

    def header(self) -> ReportHeader:
        return ReportHeader(cod=self.obs_code, con=self.contact, obs=self.observers,
                            mea=self.measurers, tel=self.telescope, ack=self.ack,
                            ac2=self.ac2, net=self.net)


PRESETS: dict[str, dict] = {
    "generic": {},
    # Pan-STARRS 1 image sets distributed by IASC (the PS1.cfg equivalent).
    "ps1": {
        "preset": "ps1",
        "obs_code": "F51",
        "telescope": "1.8-m f/4.4 Ritchey-Chretien + CCD",
        "observers": "J. Bulger, T. Lowe, A. Schultz, M. Willman",
        "measurers": "T. Vorobjov, PS1 Science Consortium",
        "band": "G",
        "net": "Gaia DR3",
        "plate_order": 2,
        "auto_ack": True,
        "extra_digit": True,
    },
    "ps2": {
        "preset": "ps2",
        "obs_code": "F52",
        "telescope": "1.8-m f/4.4 Ritchey-Chretien + CCD",
        "observers": "J. Bulger, T. Lowe, A. Schultz, M. Willman",
        "measurers": "T. Vorobjov, PS1 Science Consortium",
        "band": "G",
        "net": "Gaia DR3",
        "plate_order": 2,
        "auto_ack": True,
        "extra_digit": True,
    },
}


def make_settings(preset: str = "generic", **overrides) -> Settings:
    vals = dict(PRESETS.get(preset, {}))
    vals.update({k: v for k, v in overrides.items() if v is not None})
    known = Settings.__dataclass_fields__
    return Settings(**{k: v for k, v in vals.items() if k in known})


def _finite(v):
    return None if (v is None or (isinstance(v, float) and not np.isfinite(v))) else v


def find_fits(folder: str) -> list[str]:
    out = []
    for root, _dirs, files in os.walk(folder):
        for fn in sorted(files):
            if fn.lower().endswith(FITS_EXT) and not fn.startswith("."):
                out.append(os.path.join(root, fn))
    return sorted(out)


class Workspace:
    def __init__(self, paths: list[str], settings: Settings | None = None,
                 name: str = "", tmpdir: str | None = None):
        if len(paths) < 2:
            raise ValueError("An image set needs at least 2 FITS frames of the same field.")
        self.id = uuid.uuid4().hex[:12]
        self.name = name or os.path.basename(os.path.dirname(os.path.abspath(paths[0])))
        self.tmpdir = tmpdir
        self.settings = settings or Settings()
        self.fset = load_frame_set(paths, allow_no_wcs=True)
        self.reduction = None
        self.tracklets: list = []
        self.known_table = None
        self.warnings: list[str] = []
        self._stats = {}
        self._stack = None            # (image, label) of the last Track & Stack

    # --- construction helpers -------------------------------------------------
    @classmethod
    def from_folder(cls, folder: str, settings: Settings | None = None):
        folder = os.path.abspath(os.path.expanduser(folder))
        if os.path.isfile(folder) and folder.lower().endswith(".zip"):
            return cls.from_zip(folder, settings)
        if not os.path.isdir(folder):
            raise ValueError(f"{folder}: not a folder or .zip file.")
        paths = find_fits(folder)
        if not paths:
            raise ValueError(f"No FITS images found in {folder}.")
        return cls(paths, settings, name=os.path.basename(folder.rstrip("/")))

    @classmethod
    def from_zip(cls, zip_path: str, settings: Settings | None = None, name: str = ""):
        td = tempfile.mkdtemp(prefix="astrohunt-ws-")
        with zipfile.ZipFile(zip_path) as z:
            for info in z.infolist():
                base = os.path.basename(info.filename)
                if info.is_dir() or not base.lower().endswith(FITS_EXT) or base.startswith("."):
                    continue
                with z.open(info) as src, open(os.path.join(td, base), "wb") as dst:
                    shutil.copyfileobj(src, dst)   # flat extract: no zip-slip paths
        paths = find_fits(td)
        if not paths:
            shutil.rmtree(td, ignore_errors=True)
            raise ValueError("The zip contains no FITS images.")
        nm = name or os.path.splitext(os.path.basename(zip_path))[0]
        return cls(paths, settings, name=nm, tmpdir=td)

    def close(self):
        if self.tmpdir:
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    # --- info -----------------------------------------------------------------
    def frame(self, i: int):
        if not 0 <= i < len(self.fset):
            raise IndexError(f"frame {i} out of range")
        return self.fset.frames[i]

    def info(self) -> dict:
        frames = []
        for f in self.fset:
            fr = None
            if self.reduction is not None:
                r = self.reduction.frame_reports[f.index]
                sol = self.reduction.solutions[f.index]
                fr = {
                    "n_sources": r.n_sources, "n_matched": r.n_matched,
                    "n_vetoed": r.n_vetoed, "rms_arcsec": _finite(round(r.rms_arcsec, 4)),
                    "zeropoint": _finite(round(r.mag_zeropoint, 3)),
                    "zp_scatter": _finite(round(float(sol.mag_scatter), 3)),
                    "fwhm_px": _finite(round(r.fwhm_px, 2)),
                    "fwhm_arcsec": _finite(round(r.fwhm_px * self.pixel_scale, 2)),
                }
            frames.append({
                "index": f.index, "file": os.path.basename(f.path),
                "width": int(f.shape[1]), "height": int(f.shape[0]),
                "time_utc": f.time.utc.isot, "mjd": f.mjd, "exptime": f.exptime,
                "has_wcs": f.wcs is not None, "reduction": fr,
            })
        center = None
        if self.fset.frames[0].wcs is not None:
            center = [round(c, 6) for c in self.fset.center_radec]
        return {
            "id": self.id, "name": self.name, "frames": frames,
            "pixel_scale": _finite(round(self.pixel_scale, 4)),
            "center": center,
            "span_minutes": round(self.fset.span_minutes, 2),
            "reduced": self.reduction is not None,
            "catalog": getattr(self.reduction.refcat, "source", None) if self.reduction else None,
            "n_catalog": len(self.reduction.refcat) if self.reduction else 0,
            "median_rms_arcsec": _finite(round(self.reduction.median_rms_arcsec, 4))
            if self.reduction else None,
            "settings": asdict(self.settings),
            "warnings": self.warnings,
        }

    @property
    def pixel_scale(self) -> float:
        if self.fset.frames[0].wcs is None:
            return float("nan")
        return self.fset.pixel_scale_arcsec

    # --- display ----------------------------------------------------------------
    def _frame_stats(self, key, data):
        if key not in self._stats:
            sub = data[::4, ::4]
            _mean, med, std = sigma_clipped_stats(sub[np.isfinite(sub)], sigma=3.0, maxiters=5)
            self._stats[key] = (float(med), float(std) or 1.0)
        return self._stats[key]

    def render_png(self, i, lo_sigma: float = -2.0, hi_sigma: float = 12.0,
                   stretch: str = "linear") -> bytes:
        """8-bit greyscale PNG of frame i (or the last stack with i == 'stack')."""
        from PIL import Image

        if i == "stack":
            if self._stack is None:
                raise ValueError("No stacked image yet — run Track & Stack first.")
            data = self._stack[0]
        else:
            data = self.frame(int(i)).data
        med, std = self._frame_stats(i if i != "stack" else id(data), data)
        lo, hi = med + lo_sigma * std, med + hi_sigma * std
        norm = np.clip((np.nan_to_num(data, nan=med) - lo) / max(hi - lo, 1e-9), 0, 1)
        if stretch == "asinh":
            q = 10.0
            norm = np.arcsinh(norm * q) / np.arcsinh(q)
        elif stretch == "sqrt":
            norm = np.sqrt(norm)
        img = Image.fromarray((norm * 255.0 + 0.5).astype(np.uint8), "L")
        buf = _io.BytesIO()
        img.save(buf, format="PNG", compress_level=3)
        return buf.getvalue()

    # --- data reduction -----------------------------------------------------------
    def reduce(self, reference_catalog=None, hint_radec=None) -> dict:
        s = self.settings
        self.reduction = reduce_frames(
            self.fset, reference_catalog,
            fwhm=s.fwhm_px, threshold_sigma=s.threshold_sigma, plate_order=s.plate_order,
            use_online=s.use_online, hint_radec=hint_radec,
            max_sources=s.max_sources, mag_limit=s.mag_limit, aperture_photometry=True,
        )
        self.warnings = list(self.reduction.warnings)
        for fr in self.reduction.frame_reports:
            if np.isfinite(fr.rms_arcsec) and fr.rms_arcsec > 0.5:
                self.warnings.append(
                    f"image {fr.index + 1}: astrometric fit RMS {fr.rms_arcsec:.2f}\" is poor.")
            if fr.n_matched < 10:
                self.warnings.append(
                    f"image {fr.index + 1}: only {fr.n_matched} reference stars matched.")
        self.tracklets = []
        self.known_table = None
        self._snrmaps = None
        return self.info()

    def _need_reduction(self):
        if self.reduction is None:
            raise ValueError("Run Data Reduction first.")
        return self.reduction

    # --- measuring ------------------------------------------------------------------
    def measure(self, i: int, x: float, y: float, snap: bool = True) -> dict:
        """Measure the object nearest (x, y) [0-based array pixels] in frame i."""
        red = self._need_reduction()
        f = self.frame(i)
        fw = red.fwhm_px[i]
        m = measure_at(f.data, x, y, fwhm_guess=fw, snap=snap)
        out = m.to_dict()
        sol = red.solutions[i]
        ra, dec = sol.pix2sky(np.array([m.x]), np.array([m.y]))
        mag = None
        if m.flux == m.flux and m.flux > 0 and np.isfinite(sol.mag_zeropoint):
            mag = float(-2.5 * np.log10(m.flux) + sol.mag_zeropoint)
        out.update({
            "frame": i, "mjd": f.mjd, "time_utc": f.time.utc.isot,
            "ra": float(ra[0]), "dec": float(dec[0]), "mag": mag,
            "fwhm_arcsec": _finite(m.fwhm * self.pixel_scale),
            "ra_hms": _hms(float(ra[0])), "dec_dms": _dms(float(dec[0])),
        })
        return out

    def predict(self, measurements: list[dict], i: int) -> dict | None:
        """Where should an object be in frame i, given >= 2 measurements?"""
        if len(measurements) < 2:
            return None
        red = self._need_reduction()
        t = np.array([m["mjd"] for m in measurements])
        ra = np.array([m["ra"] for m in measurements])
        dec = np.array([m["dec"] for m in measurements])
        A = np.vstack([np.ones_like(t), t - t[0]]).T
        cra = np.linalg.lstsq(A, ra, rcond=None)[0]
        cdec = np.linalg.lstsq(A, dec, rcond=None)[0]
        dt = self.frame(i).mjd - t[0]
        x, y = red.solutions[i].sky2pix(cra[0] + cra[1] * dt, cdec[0] + cdec[1] * dt)
        return {"x": float(x[0]), "y": float(y[0])}

    # --- automatic moving-object detection ---------------------------------------------
    def _snr_maps(self):
        """Per-frame smoothed significance maps (for the 'is it static?' test)."""
        if getattr(self, "_snrmaps", None) is None:
            from scipy.ndimage import uniform_filter

            maps = []
            for f in self.fset:
                med, std = self._frame_stats(f.index, f.data)
                sm = uniform_filter(np.nan_to_num(f.data, nan=med).astype(np.float32), size=3)
                maps.append((sm - med) / (std / 3.0))
            self._snrmaps = maps
        return self._snrmaps

    def _is_static(self, d, sigma=3.5) -> bool:
        """True if something is also present at this sky position in other frames.

        A star too faint to be *detected* in every frame still leaves flux at the
        same place; a real mover leaves empty sky behind.
        """
        red = self.reduction
        maps = self._snr_maps()
        hits = others = 0
        for f in self.fset:
            if f.index == d.frame:
                continue
            x, y = red.solutions[f.index].sky2pix(d.ra, d.dec)
            xi, yi = int(round(float(x[0]))), int(round(float(y[0])))
            ny, nx = f.shape
            if not (1 <= xi < nx - 1 and 1 <= yi < ny - 1):
                continue
            others += 1
            if maps[f.index][yi - 1:yi + 2, xi - 1:xi + 2].max() >= sigma:
                hits += 1
        return others >= 2 and hits >= max(2, others - 1)

    def detect(self) -> list[dict]:
        """Automatic moving-object detection (+ known-object attribution)."""
        from . import detect as det

        red = self._need_reduction()
        s = self.settings
        moving = det.suppress_static(red.detections_per_frame, tol_arcsec=s.static_tol_arcsec)

        # quality gate: every surviving detection must look like a star-shaped
        # source of sensible width, and must not sit on something static.
        good = []
        for d in moving:
            if d.snr < s.threshold_sigma * 0.8:
                continue
            m = self.measure(d.frame, d.x, d.y, snap=False)
            fw = red.fwhm_px[d.frame]
            if not m["ok"] or m["fwhm"] is None:
                continue
            if not (0.55 * fw <= m["fwhm"] <= 2.2 * fw) or m["elongation"] > 2.2:
                continue
            if m["snr"] < s.threshold_sigma or m["mag"] is None:
                continue
            if m["peak"] + m["background"] >= s.saturation:
                continue
            d.x, d.y, d.ra, d.dec, d.mag = m["x"], m["y"], m["ra"], m["dec"], m["mag"]
            d.snr, d.fwhm, d.fit_rms = m["snr"], m["fwhm"], m["fit_rms"]
            if self._is_static(d):
                continue
            d.meas = m
            good.append(d)

        self.tracklets = det.link_tracklets(
            good, max_rate_arcsec_min=s.max_rate_arcsec_min,
            min_rate_arcsec_min=s.min_rate_arcsec_min, tol_arcsec=s.link_tol_arcsec,
            min_points=s.min_points, max_mag_spread=s.max_mag_spread,
        )
        self.known_objects()   # fills self.known_table (None if offline/unreachable)
        for trk in self.tracklets:
            trk.known_name = catalog.identify_tracklet(trk, self.known_table)
        return self.tracklets_json()

    def tracklets_json(self) -> list[dict]:
        out = []
        for k, trk in enumerate(self.tracklets, 1):
            meas = [dict(d.meas) for d in sorted(trk.detections, key=lambda d: d.time_mjd)]
            out.append({
                "idx": k, "known": trk.known_name, "n": trk.n,
                "rate": round(trk.rate_total, 3), "pa": round(trk.pa_deg, 1),
                "rms": round(trk.rms_arcsec, 3),
                "min_snr": _finite(min((m["snr"] or 0.0) for m in meas)),
                "measurements": meas,
            })
        out.sort(key=lambda t: (-t["n"], -(t["min_snr"] or 0)))
        for k, t in enumerate(out, 1):
            t["idx"] = k
        return out

    # --- known-object overlay --------------------------------------------------------------
    def known_objects(self) -> list[dict]:
        red = self._need_reduction()
        if self.known_table is None and self.settings.use_online:
            ra, dec = red.center_radec
            ny, nx = self.fset.frames[0].shape
            radius = max(15.0, 0.75 * self.pixel_scale * float(np.hypot(nx, ny)) / 60.0)
            code = self.settings.obs_code if self.settings.obs_code != "XXX" else "500"
            self.known_table = catalog.skybot_cone(
                ra, dec, self.fset.frames[0].mjd, radius_arcmin=radius, location=code)
        return catalog.known_positions(self.known_table, self.fset.frames, red.solutions)

    # --- Track & Stack -------------------------------------------------------------------------
    def stack(self, rate_arcsec_min: float, pa_deg: float, combine: str = "mean") -> dict:
        from .stack import shift_and_stack

        self._need_reduction()
        img, _t0 = shift_and_stack(self.fset.frames, rate_arcsec_min, pa_deg,
                                   combine=combine, subtract_static=False)
        self._stack = (img, f"{rate_arcsec_min:.2f}\"/min @ PA {pa_deg:.0f}")
        self._stats.pop(id(img), None)
        return {"label": self._stack[1], "width": int(img.shape[1]), "height": int(img.shape[0])}

    # --- report ----------------------------------------------------------------------------------
    def report(self, objects: list[dict], header: dict | None = None) -> dict:
        """objects: [{"designation": str, "measurements": [measure() dicts]}]"""
        hdr = self.settings.header()
        for k, v in (header or {}).items():
            if hasattr(hdr, k) and v is not None:
                setattr(hdr, k, v)
        robjs, problems = [], []
        for o in objects:
            obs = [ReportObs(mjd=m["mjd"], ra=m["ra"], dec=m["dec"], mag=m.get("mag"),
                             band=self.settings.band) for m in o.get("measurements", [])]
            if obs:
                robjs.append(ReportObject(o["designation"], obs))
            problems += object_warnings(o, self.pixel_scale)
        if self.settings.auto_ack and not hdr.ack:
            from datetime import datetime, timezone

            hdr.ack = "MPCReport file updated " + datetime.now(timezone.utc).strftime(
                "%Y.%m.%d %H:%M:%S")
        text = build_report(robjs, hdr, extra_digit=self.settings.extra_digit)
        problems = check_report(text) + problems
        seen = [o.designation.strip().upper() for o in robjs]
        for d in sorted({d for d in seen if seen.count(d) > 1}):
            problems.append(f"designation {d} is used for more than one object")
        if hdr.cod == "XXX":
            problems.insert(0, "observatory code is still the placeholder XXX")
        return {"text": text, "problems": problems}


def object_warnings(obj: dict, pixel_scale: float) -> list[str]:
    """Sanity checks an experienced measurer would make on one object."""
    name = obj.get("designation", "?")
    ms = sorted(obj.get("measurements", []), key=lambda m: m["mjd"])
    out = []
    if len(ms) >= 3:
        t = np.array([m["mjd"] for m in ms]) * 1440.0
        ra = np.array([m["ra"] for m in ms])
        dec = np.array([m["dec"] for m in ms])
        cosd = np.cos(np.radians(dec.mean()))
        A = np.vstack([np.ones_like(t), t - t[0]]).T
        rra = ra - A @ np.linalg.lstsq(A, ra, rcond=None)[0]
        rdec = dec - A @ np.linalg.lstsq(A, dec, rcond=None)[0]
        res = np.hypot(rra * cosd, rdec) * 3600.0
        if res.max() > 1.0:
            out.append(f"{name}: motion is not a straight line at constant speed "
                       f"(worst residual {res.max():.2f}\").")
    mags = [m["mag"] for m in ms if m.get("mag") is not None]
    if len(mags) >= 2 and max(mags) - min(mags) > 1.0:
        out.append(f"{name}: brightness varies by {max(mags) - min(mags):.1f} mag between images.")
    for m in ms:
        if (m.get("snr") or 0) < 5:
            out.append(f"{name}: image {m['frame'] + 1} has SNR {m.get('snr') or 0:.1f} (< 5).")
        if not m.get("ok", True) and m.get("note"):
            out.append(f"{name}: image {m['frame'] + 1}: {m['note']}.")
    return out


def _hms(ra_deg: float) -> str:
    from .mpcreport import format_ra

    return format_ra(ra_deg).strip()


def _dms(dec_deg: float) -> str:
    from .mpcreport import format_dec

    return format_dec(dec_deg).strip()


def motion_summary(measurements: list[dict]) -> dict | None:
    """Rate ("/min) and position angle from first to last measurement."""
    ms = sorted(measurements, key=lambda m: m["mjd"])
    if len(ms) < 2:
        return None
    a, b = ms[0], ms[-1]
    dt = (b["mjd"] - a["mjd"]) * 1440.0
    if dt <= 0:
        return None
    cosd = np.cos(np.radians((a["dec"] + b["dec"]) / 2))
    dra = (b["ra"] - a["ra"]) * cosd * 3600.0
    ddec = (b["dec"] - a["dec"]) * 3600.0
    return {"rate": float(np.hypot(dra, ddec) / dt),
            "pa": float(np.degrees(np.arctan2(dra, ddec)) % 360.0)}


# --- in-memory session registry (local single-user app) -----------------------------

_SESSIONS: dict[str, Workspace] = {}
MAX_SESSIONS = 6


def register(ws: Workspace) -> Workspace:
    while len(_SESSIONS) >= MAX_SESSIONS:
        old = next(iter(_SESSIONS))
        _SESSIONS.pop(old).close()
    _SESSIONS[ws.id] = ws
    return ws


def get(ws_id: str) -> Workspace:
    if ws_id not in _SESSIONS:
        raise KeyError("Unknown or expired workspace — load the image set again.")
    return _SESSIONS[ws_id]
