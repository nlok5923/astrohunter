"""End-to-end pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import astrometry, catalog, detect, extract, platesolve
from . import vet as vetting
from .ades import ObsMetadata, write_ades_psv, write_ades_xml, write_obs80
from .io import FrameSet, load_frame_set
from .measure import aperture_flux, estimate_fwhm, local_sky


@dataclass
class FrameReport:
    index: int
    n_sources: int
    n_matched: int
    rms_arcsec: float
    mag_zeropoint: float
    n_vetoed: int = 0
    fwhm_px: float = float("nan")


@dataclass
class PipelineResult:
    frame_reports: list[FrameReport]
    tracklets: list
    n_candidates: int
    n_known: int
    ades_path: str | None = None
    ades_xml_path: str | None = None
    obs80_path: str | None = None
    html_path: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def median_rms_arcsec(self):
        vals = [fr.rms_arcsec for fr in self.frame_reports if np.isfinite(fr.rms_arcsec)]
        return float(np.median(vals)) if vals else float("nan")


@dataclass
class Reduction:
    """Everything "Data Reduction" produces: per-frame sources + plate solutions."""

    fset: FrameSet
    refcat: object
    srcs_per_frame: list
    solutions: list
    frame_reports: list[FrameReport]
    detections_per_frame: list
    fwhm_px: list[float]
    center_radec: tuple[float, float]
    warnings: list[str] = field(default_factory=list)

    @property
    def median_rms_arcsec(self):
        vals = [fr.rms_arcsec for fr in self.frame_reports if np.isfinite(fr.rms_arcsec)]
        return float(np.median(vals)) if vals else float("nan")


def aperture_mags(data, xs, ys, fwhm):
    """Instrumental aperture magnitudes (r = 1.5 FWHM) — same recipe as measure_at."""
    radius = max(2.0, 1.5 * fwhm)
    r_in = max(radius + 2.0, 2.5 * fwhm)
    r_out = r_in + max(6.0, 2.0 * fwhm)
    out = np.full(len(xs), np.nan)
    for i, (x, y) in enumerate(zip(xs, ys)):
        bg, _sig = local_sky(data, x, y, r_in, r_out)
        flux, _n = aperture_flux(data, x, y, radius, bg)
        if np.isfinite(flux) and flux > 0:
            out[i] = -2.5 * np.log10(flux)
    return out


def reduce_frames(
    fset: FrameSet,
    reference_catalog=None,
    *,
    fwhm: float | None = 3.5,
    threshold_sigma: float = 5.0,
    plate_order: int = 1,
    vet: bool = True,
    vetter: str = "rule",
    vet_model: str | None = None,
    use_online: bool = True,
    hint_radec: tuple[float, float] | None = None,
    max_sources: int | None = 2000,
    mag_limit: float = 21.0,
    aperture_photometry: bool = False,
    refine: bool = True,
) -> Reduction:
    """Astrometric + photometric "Data Reduction" of every frame.

    `fwhm=None` measures the seeing from the image itself (needed for real data,
    where the PSF width in pixels is not known in advance).
    """
    warns: list[str] = []
    have_wcs = fset.frames[0].wcs is not None
    if have_wcs:
        center_ra, center_dec = fset.center_radec
    elif hint_radec is not None:
        center_ra, center_dec = hint_radec
    else:
        raise ValueError(
            "Frames have no WCS and no pointing hint was given. Pass "
            "hint_radec=(ra, dec) so astrohunt can plate-solve them."
        )

    # --- reference catalogue for astrometric calibration -------------------
    refcat = reference_catalog
    if refcat is None and use_online:
        if have_wcs:
            ny, nx = fset.frames[0].shape
            radius = max(8.0, 0.75 * fset.pixel_scale_arcsec * float(np.hypot(nx, ny)) / 60.0)
        else:
            radius = 20.0
        refcat = catalog.gaia_cone(center_ra, center_dec, radius_arcmin=radius,
                                   mag_limit=mag_limit,
                                   epoch_year=float(fset.frames[0].time.jyear))
        if refcat is None:
            warns.append("No reference catalogue (Gaia offline and none supplied).")
    if refcat is None:
        raise ValueError(
            "An astrometric reference catalogue is required. Supply one "
            "(reference_catalog=...) or enable online Gaia access."
        )

    # --- extract sources once per frame ------------------------------------
    fwhm_px = []
    srcs_per_frame = []
    for f in fset:
        fw = fwhm
        if fw is None:
            first = extract.extract_sources(f.data, fwhm=4.0, threshold_sigma=20.0,
                                            max_sources=300)
            fw = estimate_fwhm(f.data, first["x"], first["y"], n=60, fwhm_guess=4.0)
            fw = float(np.clip(fw, 1.5, 15.0))
        fwhm_px.append(float(fw))
        srcs_per_frame.append(
            extract.extract_sources(f.data, fwhm=fw, threshold_sigma=threshold_sigma,
                                    max_sources=max_sources)
        )

    # --- real-bogus vetting: drop artifacts before they form tracklets ------
    n_vetoed = [0] * len(fset)
    if vet:
        ml_model = None
        if vetter == "ml" and vet_model:
            from .mlvet import RealBogusModel

            ml_model = RealBogusModel.load(vet_model)
            warns.append(f"using learned vetter ({ml_model.backend}) from {vet_model}.")
        cleaned = []
        for i, (f, srcs) in enumerate(zip(fset, srcs_per_frame)):
            if len(srcs) == 0:
                cleaned.append(srcs)
                continue
            if ml_model is not None:
                from .mlvet import predict_mask

                mask = predict_mask(ml_model, f.data, srcs)
            else:
                mask, _stats = vetting.vet_sources(f.data, srcs)
            n_vetoed[i] = int((~mask).sum())
            cleaned.append(srcs[mask])
        srcs_per_frame = cleaned

    # --- plate-solve any frame lacking a WCS -------------------------------
    for frame, srcs in zip(fset, srcs_per_frame):
        if frame.wcs is None:
            sr = platesolve.solve_wcs(
                srcs["x"], srcs["y"], refcat.ra, refcat.dec,
                center_ra, center_dec, ref_mag=refcat.mag,
            )
            frame.wcs = sr.wcs
            warns.append(
                f"frame {frame.index}: plate-solved "
                f"({sr.n_inliers} inliers, {sr.rms_arcsec:.2f}\" rms)."
            )

    # --- lock approximate header WCSs onto the catalogue ---------------------
    match_tol = [3.0] * len(fset)
    if refine:
        for frame, srcs in zip(fset, srcs_per_frame):
            if len(srcs) < 8:
                continue
            try:
                wcs2, _n, _rms = platesolve.refine_wcs(
                    srcs["x"], srcs["y"], frame.wcs, refcat.ra, refcat.dec, refcat.mag)
            except Exception as e:  # noqa: BLE001
                warns.append(f"image {frame.index + 1}: {e} Using the header WCS as is.")
                continue
            frame.wcs = wcs2
            from astropy.wcs.utils import proj_plane_pixel_scales

            scale = float(np.mean(proj_plane_pixel_scales(wcs2))) * 3600.0
            match_tol[frame.index] = float(np.clip(2.5 * scale, 1.0, 3.0))

    # --- per-frame astrometric solution ------------------------------------
    frame_reports = []
    solutions = []
    detections_per_frame: list[list[detect.Detection]] = []
    for frame, srcs, fw in zip(fset, srcs_per_frame, fwhm_px):
        if aperture_photometry and len(srcs):
            srcs["mag_inst"] = aperture_mags(frame.data, np.asarray(srcs["x"]),
                                             np.asarray(srcs["y"]), fw)
        sol, n_matched = astrometry.solve_frame(
            srcs, frame.wcs, refcat.ra, refcat.dec, cat_mag=refcat.mag,
            order=plate_order, match_tol_arcsec=match_tol[frame.index],
        )
        solutions.append(sol)
        ra, dec = sol.pix2sky(np.asarray(srcs["x"]), np.asarray(srcs["y"]))
        mag = sol.calibrate_mag(np.asarray(srcs["mag_inst"]))

        dets = [
            detect.Detection(
                frame=frame.index, time_mjd=frame.mjd,
                x=float(srcs["x"][i]), y=float(srcs["y"][i]),
                ra=float(ra[i]), dec=float(dec[i]),
                mag=float(mag[i]) if np.isfinite(mag[i]) else float("nan"),
                flux=float(srcs["flux"][i]),
                snr=float(srcs["snr"][i]),
            )
            for i in range(len(srcs))
        ]
        detections_per_frame.append(dets)
        frame_reports.append(
            FrameReport(frame.index, len(srcs), n_matched, sol.rms_arcsec,
                        sol.mag_zeropoint, n_vetoed[frame.index], fw)
        )

    return Reduction(fset, refcat, srcs_per_frame, solutions, frame_reports,
                     detections_per_frame, fwhm_px, (center_ra, center_dec), warns)


def find_movers(
    red: Reduction,
    *,
    static_tol_arcsec: float = 2.0,
    link_tol_arcsec: float = 2.5,
    max_rate_arcsec_min: float = 25.0,
    min_rate_arcsec_min: float = 0.05,
    min_points: int = 3,
    max_mag_spread: float | None = None,
    use_online: bool = True,
    obs_code: str = "500",
):
    """Moving-object detection + known-object attribution on a Reduction."""
    moving = detect.suppress_static(red.detections_per_frame, tol_arcsec=static_tol_arcsec)
    tracklets = detect.link_tracklets(
        moving,
        max_rate_arcsec_min=max_rate_arcsec_min,
        min_rate_arcsec_min=min_rate_arcsec_min,
        tol_arcsec=link_tol_arcsec,
        min_points=min_points,
        max_mag_spread=max_mag_spread,
    )
    known = None
    if use_online:
        ra, dec = red.center_radec
        ny, nx = red.fset.frames[0].shape
        radius = max(15.0, 0.75 * red.fset.pixel_scale_arcsec * float(np.hypot(nx, ny)) / 60.0)
        known = catalog.skybot_cone(ra, dec, red.fset.frames[0].mjd,
                                    radius_arcmin=radius, location=obs_code)
    for trk in tracklets:
        trk.known_name = catalog.identify_tracklet(trk, known)
    return tracklets, known


def run_pipeline(
    frame_paths,
    reference_catalog=None,
    obs_meta: ObsMetadata | None = None,
    *,
    fwhm: float | None = 3.5,
    threshold_sigma: float = 5.0,
    plate_order: int = 1,
    static_tol_arcsec: float = 2.0,
    link_tol_arcsec: float = 2.5,
    max_rate_arcsec_min: float = 25.0,
    min_points: int = 3,
    vet: bool = True,
    vetter: str = "rule",
    vet_model: str | None = None,
    use_online: bool = True,
    hint_radec: tuple[float, float] | None = None,
    out_ades: str | None = None,
    out_ades_xml: str | None = None,
    out_obs80: str | None = None,
    out_html: str | None = None,
) -> PipelineResult:
    """Run the full detect-and-report pipeline over a set of FITS frames.

    `reference_catalog` may be a ReferenceCatalog (offline) or None (fetch Gaia
    DR3 online). Set `use_online=False` to force fully offline operation.
    """
    obs_meta = obs_meta or ObsMetadata()
    fset: FrameSet = load_frame_set(frame_paths, allow_no_wcs=True)
    red = reduce_frames(
        fset, reference_catalog, fwhm=fwhm, threshold_sigma=threshold_sigma,
        plate_order=plate_order, vet=vet, vetter=vetter, vet_model=vet_model,
        use_online=use_online, hint_radec=hint_radec,
    )
    warns = red.warnings
    frame_reports = red.frame_reports
    tracklets, _known = find_movers(
        red, static_tol_arcsec=static_tol_arcsec, link_tol_arcsec=link_tol_arcsec,
        max_rate_arcsec_min=max_rate_arcsec_min, min_points=min_points,
        use_online=use_online,
        obs_code=obs_meta.obs_code if obs_meta.obs_code != "XXX" else "500",
    )
    n_known = sum(1 for t in tracklets if t.known_name)

    candidates = [t for t in tracklets if not t.known_name]
    n_candidates = len(candidates)

    # --- reporting (new candidates only) ----------------------------------
    ades_path = ades_xml_path = obs80_path = None
    if out_ades and candidates:
        ades_path = write_ades_psv(candidates, obs_meta, out_ades)
    if out_ades_xml and candidates:
        ades_xml_path = write_ades_xml(candidates, obs_meta, out_ades_xml)
    if out_obs80 and candidates:
        obs80_path = write_obs80(candidates, obs_meta, out_obs80)

    html_path = None
    if out_html and tracklets:
        from . import review

        html_path = review.write_html_report(fset.frames, tracklets, out_html, meta=obs_meta)

    return PipelineResult(
        frame_reports=frame_reports,
        tracklets=tracklets,
        n_candidates=n_candidates,
        n_known=n_known,
        ades_path=ades_path,
        ades_xml_path=ades_xml_path,
        obs80_path=obs80_path,
        html_path=html_path,
        warnings=warns,
    )
