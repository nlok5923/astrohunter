"""End-to-end pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import astrometry, catalog, detect, extract
from .ades import ObsMetadata, write_ades_psv, write_obs80
from .io import FrameSet, load_frame_set


@dataclass
class FrameReport:
    index: int
    n_sources: int
    n_matched: int
    rms_arcsec: float
    mag_zeropoint: float


@dataclass
class PipelineResult:
    frame_reports: list[FrameReport]
    tracklets: list
    n_candidates: int
    n_known: int
    ades_path: str | None = None
    obs80_path: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def median_rms_arcsec(self):
        vals = [fr.rms_arcsec for fr in self.frame_reports if np.isfinite(fr.rms_arcsec)]
        return float(np.median(vals)) if vals else float("nan")


def run_pipeline(
    frame_paths,
    reference_catalog=None,
    obs_meta: ObsMetadata | None = None,
    *,
    fwhm: float = 3.5,
    threshold_sigma: float = 5.0,
    plate_order: int = 1,
    static_tol_arcsec: float = 2.0,
    link_tol_arcsec: float = 2.5,
    max_rate_arcsec_min: float = 25.0,
    min_points: int = 3,
    use_online: bool = True,
    out_ades: str | None = None,
    out_obs80: str | None = None,
) -> PipelineResult:
    """Run the full detect-and-report pipeline over a set of FITS frames.

    `reference_catalog` may be a ReferenceCatalog (offline) or None (fetch Gaia
    DR3 online). Set `use_online=False` to force fully offline operation.
    """
    obs_meta = obs_meta or ObsMetadata()
    warns: list[str] = []

    fset: FrameSet = load_frame_set(frame_paths)
    center_ra, center_dec = fset.center_radec

    # --- reference catalogue for astrometric calibration -------------------
    refcat = reference_catalog
    if refcat is None and use_online:
        radius = max(8.0, fset.pixel_scale_arcsec * max(fset.frames[0].shape) / 60.0)
        refcat = catalog.gaia_cone(center_ra, center_dec, radius_arcmin=radius)
        if refcat is None:
            warns.append("No reference catalogue (Gaia offline and none supplied).")
    if refcat is None:
        raise ValueError(
            "An astrometric reference catalogue is required. Supply one "
            "(reference_catalog=...) or enable online Gaia access."
        )

    # --- per-frame extraction + astrometric solution -----------------------
    frame_reports = []
    detections_per_frame: list[list[detect.Detection]] = []
    for frame in fset:
        srcs = extract.extract_sources(
            frame.data, fwhm=fwhm, threshold_sigma=threshold_sigma
        )
        sol, n_matched = astrometry.solve_frame(
            srcs, frame.wcs, refcat.ra, refcat.dec, cat_mag=refcat.mag,
            order=plate_order,
        )
        ra, dec = sol.pix2sky(np.asarray(srcs["x"]), np.asarray(srcs["y"]))
        mag = sol.calibrate_mag(np.asarray(srcs["mag_inst"]))

        dets = [
            detect.Detection(
                frame=frame.index, time_mjd=frame.mjd,
                x=float(srcs["x"][i]), y=float(srcs["y"][i]),
                ra=float(ra[i]), dec=float(dec[i]),
                mag=float(mag[i]) if np.isfinite(mag[i]) else float("nan"),
                flux=float(srcs["flux"][i]),
            )
            for i in range(len(srcs))
        ]
        detections_per_frame.append(dets)
        frame_reports.append(
            FrameReport(frame.index, len(srcs), n_matched, sol.rms_arcsec, sol.mag_zeropoint)
        )

    # --- moving-object detection ------------------------------------------
    moving = detect.suppress_static(detections_per_frame, tol_arcsec=static_tol_arcsec)
    tracklets = detect.link_tracklets(
        moving,
        max_rate_arcsec_min=max_rate_arcsec_min,
        tol_arcsec=link_tol_arcsec,
        min_points=min_points,
    )

    # --- known-object attribution -----------------------------------------
    skybot = None
    if use_online and tracklets:
        skybot = catalog.skybot_cone(center_ra, center_dec, fset.frames[0].mjd)
    n_known = 0
    for trk in tracklets:
        name = catalog.identify_tracklet(trk, skybot)
        trk.known_name = name
        if name:
            n_known += 1

    candidates = [t for t in tracklets if not t.known_name]
    n_candidates = len(candidates)

    # --- reporting (new candidates only) ----------------------------------
    ades_path = obs80_path = None
    if out_ades and candidates:
        ades_path = write_ades_psv(candidates, obs_meta, out_ades)
    if out_obs80 and candidates:
        obs80_path = write_obs80(candidates, obs_meta, out_obs80)

    return PipelineResult(
        frame_reports=frame_reports,
        tracklets=tracklets,
        n_candidates=n_candidates,
        n_known=n_known,
        ades_path=ades_path,
        obs80_path=obs80_path,
        warnings=warns,
    )
