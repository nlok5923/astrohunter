"""Command-line interface for astrohunt."""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__


def _print_result(result, label="RESULT"):
    print(f"\n=== {label} ===")
    print("Per-frame astrometry (LSPC):")
    rms_hdr = 'rms(")'
    print(f"  {'frame':>5}  {'sources':>7}  {'matched':>7}  {'vetoed':>6}  "
          f"{rms_hdr:>7}  {'zp':>6}")
    for fr in result.frame_reports:
        zp = f"{fr.mag_zeropoint:6.2f}" if fr.mag_zeropoint == fr.mag_zeropoint else "   n/a"
        print(f"  {fr.index:>5}  {fr.n_sources:>7}  {fr.n_matched:>7}  "
              f"{fr.n_vetoed:>6}  {fr.rms_arcsec:>7.3f}  {zp}")
    print(f"  median astrometric consistency: {result.median_rms_arcsec:.3f}\"")

    print(f"\nMoving objects: {len(result.tracklets)} tracklet(s) "
          f"({result.n_known} known, {result.n_candidates} NEW candidate(s))")
    for i, t in enumerate(result.tracklets, 1):
        tag = f"KNOWN: {t.known_name}" if t.known_name else "*** NEW CANDIDATE ***"
        mid = t.detections[len(t.detections) // 2]
        print(f"  [{i}] {tag}")
        print(f"      points={t.n}  rate={t.rate_total:.2f}\"/min  PA={t.pa_deg:.0f}deg  "
              f"fit-rms={t.rms_arcsec:.3f}\"")
        print(f"      mid RA/Dec = {mid.ra:.6f}, {mid.dec:+.6f}")
    if result.ades_path:
        print(f"\nADES PSV report : {result.ades_path}")
    if result.ades_xml_path:
        print(f"ADES XML report : {result.ades_xml_path}")
    if result.obs80_path:
        print(f"obs80 report    : {result.obs80_path}")
    if result.html_path:
        print(f"review (HTML)   : {result.html_path}")
    for w in result.warnings:
        print(f"  [warn] {w}")


def _obs_meta_from_args(args):
    from .ades import ObsMetadata

    return ObsMetadata(
        obs_code=args.obscode,
        observer=args.observer,
        measurer=args.observer,
        submitter=args.observer,
        band=args.band,
    )


def cmd_synth(args):
    from .synthetic import SyntheticConfig, generate

    cfg = SyntheticConfig(n_frames=args.frames, seed=args.seed,
                          n_cosmic_rays=args.cosmic_rays)
    paths, truth = generate(cfg, args.outdir, strip_wcs=args.no_wcs)
    print(f"Generated {len(paths)} frames in {args.outdir}:")
    for p in paths:
        print(f"  {p}")
    print(f"Truth catalogue: {truth}")


def cmd_detect(args):
    from .pipeline import run_pipeline

    refcat = None
    if args.ref_catalog:
        from .catalog import load_reference_catalog

        refcat = load_reference_catalog(args.ref_catalog)

    hint = None
    if args.hint_ra is not None and args.hint_dec is not None:
        hint = (args.hint_ra, args.hint_dec)

    result = run_pipeline(
        args.frames,
        reference_catalog=refcat,
        obs_meta=_obs_meta_from_args(args),
        use_online=not args.offline,
        plate_order=args.order,
        vet=not args.no_vet,
        hint_radec=hint,
        out_ades=args.out_ades,
        out_ades_xml=args.out_ades_xml,
        out_obs80=args.out_obs80,
        out_html=args.out_html,
    )
    _print_result(result, label="DETECTION REPORT")
    if args.out_ades:
        from .validate import validate_psv_file

        print("\n" + validate_psv_file(args.out_ades).report())


def cmd_solve(args):
    from .catalog import gaia_cone, load_reference_catalog
    from .platesolve import solve_fits

    if args.ref_catalog:
        refcat = load_reference_catalog(args.ref_catalog)
    else:
        if args.hint_ra is None or args.hint_dec is None:
            raise ValueError("online solve needs --hint-ra/--hint-dec for the Gaia query.")
        refcat = gaia_cone(args.hint_ra, args.hint_dec, radius_arcmin=args.radius)
        if refcat is None:
            raise ValueError("Gaia query failed; supply --ref-catalog for offline solve.")

    hint = None
    if args.hint_ra is not None and args.hint_dec is not None:
        hint = (args.hint_ra, args.hint_dec)

    out = args.output or args.frame
    res, out_path = solve_fits(args.frame, refcat, hint_radec=hint, out_path=out)
    print(f"Plate-solved {args.frame}: {res.n_inliers} inliers, "
          f"{res.rms_arcsec:.3f}\" rms ({len(res.matched_det)} stars).")
    print(f"WCS written to: {out_path}")


def cmd_demo(args):
    """Generate a synthetic set and run the full pipeline, fully offline."""
    from .pipeline import run_pipeline
    from .synthetic import SyntheticConfig, generate
    from .catalog import load_reference_catalog

    outdir = args.outdir
    print("1) Generating synthetic 4-image asteroid-search set (offline)...")
    cfg = SyntheticConfig(seed=args.seed)
    paths, truth_path = generate(cfg, outdir)

    print("2) Running pipeline (extract -> LSPC -> suppress -> link -> report)...")
    refcat = load_reference_catalog(truth_path)
    result = run_pipeline(
        paths,
        reference_catalog=refcat,
        obs_meta=_obs_meta_from_args(args),
        use_online=False,
        out_ades=os.path.join(outdir, "report.psv"),
        out_ades_xml=os.path.join(outdir, "report.xml"),
        out_obs80=os.path.join(outdir, "report.obs80.txt"),
        out_html=os.path.join(outdir, "review.html"),
    )
    _print_result(result, label="DEMO REPORT")

    # Validate against truth.
    import json

    with open(truth_path) as f:
        truth = json.load(f)
    truth_frames = [fr for fr in truth["asteroid"]["frames"] if fr["in_frame"]]
    if result.n_candidates >= 1 and truth_frames:
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        cand = max(result.tracklets, key=lambda t: t.n)
        errs = []
        for d in cand.detections:
            nearest = min(truth_frames, key=lambda tf: abs(tf["mjd"] - d.time_mjd))
            sep = SkyCoord(d.ra * u.deg, d.dec * u.deg).separation(
                SkyCoord(nearest["ra"] * u.deg, nearest["dec"] * u.deg)
            ).arcsec
            errs.append(sep)
        import numpy as np

        print(f"\n[validation] recovered asteroid vs truth: "
              f"mean error = {np.mean(errs):.3f}\" (max {np.max(errs):.3f}\")")
        print(f"[validation] true rate = {truth['asteroid']['rate_arcsec_min']:.2f}\"/min, "
              f"recovered = {cand.rate_total:.2f}\"/min")
        ok = np.mean(errs) < 1.0
        print(f"[validation] {'PASS' if ok else 'CHECK'}: "
              f"asteroid {'recovered to sub-arcsecond accuracy' if ok else 'recovered but check residuals'}")
    else:
        print("\n[validation] no candidate recovered — investigate parameters.")


def build_parser():
    p = argparse.ArgumentParser(
        prog="astrohunt",
        description="CPU-first, cross-platform asteroid detection (IASC-style).",
    )
    p.add_argument("--version", action="version", version=f"astrohunt {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="generate a synthetic 4-image test set")
    s.add_argument("-o", "--outdir", default="data/synthetic")
    s.add_argument("-n", "--frames", type=int, default=4)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--no-wcs", action="store_true",
                   help="omit WCS (keep only a coarse RA/DEC hint) to test plate-solving")
    s.add_argument("--cosmic-rays", type=int, default=0,
                   help="inject N cosmic-ray streaks per frame (to test vetting)")
    s.set_defaults(func=cmd_synth)

    d = sub.add_parser("detect", help="run detection on a set of FITS frames")
    d.add_argument("frames", nargs="+", help="FITS frames of the same field")
    d.add_argument("--ref-catalog", help="JSON reference catalogue (offline)")
    d.add_argument("--offline", action="store_true", help="no network (Gaia/SkyBoT)")
    d.add_argument("--order", type=int, default=1, help="plate-constant polynomial order")
    d.add_argument("--no-vet", action="store_true", help="disable real-bogus artifact vetting")
    d.add_argument("--obscode", default="XXX", help="MPC observatory code")
    d.add_argument("--observer", default="Unknown")
    d.add_argument("--band", default="G")
    d.add_argument("--hint-ra", type=float, help="pointing hint RA (deg), if frames lack WCS")
    d.add_argument("--hint-dec", type=float, help="pointing hint Dec (deg), if frames lack WCS")
    d.add_argument("--out-ades", help="write ADES PSV report to this path")
    d.add_argument("--out-ades-xml", help="write ADES XML report to this path")
    d.add_argument("--out-obs80", help="write legacy obs80 report to this path")
    d.add_argument("--out-html", help="write a visual blink-review HTML report")
    d.set_defaults(func=cmd_detect)

    sv = sub.add_parser("solve", help="plate-solve a FITS frame lacking a WCS")
    sv.add_argument("frame", help="FITS frame to solve")
    sv.add_argument("-o", "--output", help="output FITS path (default: overwrite input)")
    sv.add_argument("--ref-catalog", help="JSON reference catalogue (offline)")
    sv.add_argument("--hint-ra", type=float, help="pointing hint RA (deg)")
    sv.add_argument("--hint-dec", type=float, help="pointing hint Dec (deg)")
    sv.add_argument("--radius", type=float, default=20.0, help="Gaia query radius (arcmin)")
    sv.set_defaults(func=cmd_solve)

    m = sub.add_parser("demo", help="generate synthetic data + run pipeline (offline)")
    m.add_argument("-o", "--outdir", default="data/demo")
    m.add_argument("--seed", type=int, default=7)
    m.add_argument("--obscode", default="XXX")
    m.add_argument("--observer", default="astrohunt demo")
    m.add_argument("--band", default="G")
    m.set_defaults(func=cmd_demo)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except Exception as e:  # noqa: BLE001
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
