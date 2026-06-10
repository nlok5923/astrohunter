"""FastAPI backend exposing the astrohunt pipeline to a web UI.

Endpoints:
  GET  /api/health   -> liveness
  GET  /api/demo     -> run the offline synthetic demo, return candidates
  POST /api/detect   -> upload FITS frames (+ options), run the pipeline

Responses mirror the review.py data model so the frontend can render blink GIFs,
accept/reject candidates, and export the accepted ones to an ADES PSV — the same
flow as the standalone HTML report, but live in the browser.

Run locally with:  astrohunt serve   (uvicorn astrohunt.webapi:app)
"""

# NOTE: no `from __future__ import annotations` here — FastAPI needs real
# annotation objects (not strings) to introspect Form/File/UploadFile params.

import os
import tempfile
from typing import List, Optional

from . import review
from .ades import ADES_COLUMNS, ObsMetadata, ades_metadata_lines, tracklet_to_ades_rows
from .catalog import load_reference_catalog
from .io import load_frame_set
from .pipeline import run_pipeline
from .synthetic import SyntheticConfig, generate


def _payload(frames, result, meta: ObsMetadata):
    """Build the JSON payload (candidates + GIFs + ADES rows) from a result."""
    frame_list = sorted(frames, key=lambda f: f.index)
    candidates = []
    for i, trk in enumerate(result.tracklets, 1):
        candidates.append({
            "idx": i,
            "known": trk.known_name,
            "rate": round(trk.rate_total, 3),
            "pa": round(trk.pa_deg, 1),
            "rms": round(trk.rms_arcsec, 4),
            "n": trk.n,
            "gif": review._candidate_gif(frame_list, trk),
            "rows": tracklet_to_ades_rows(trk, meta, i),
            "detections": [
                {"frame": d.frame, "ra": round(d.ra, 6), "dec": round(d.dec, 6),
                 "mag": (None if d.mag != d.mag else round(d.mag, 2))}
                for d in sorted(trk.detections, key=lambda d: d.time_mjd)
            ],
        })
    return {
        "frames": [
            {"index": fr.index, "n_sources": fr.n_sources, "n_matched": fr.n_matched,
             "n_vetoed": fr.n_vetoed, "rms_arcsec": round(fr.rms_arcsec, 4)}
            for fr in result.frame_reports
        ],
        "median_rms_arcsec": round(result.median_rms_arcsec, 4),
        "n_candidates": result.n_candidates,
        "n_known": result.n_known,
        "ades": {"metaLines": ades_metadata_lines(meta), "columns": list(ADES_COLUMNS)},
        "candidates": candidates,
        "warnings": result.warnings,
    }


def create_app():
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="astrohunt", version="0.1.0")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    @app.get("/api/health")
    def health():
        return {"status": "ok", "service": "astrohunt"}

    @app.get("/api/demo")
    def demo(seed: int = 7, cosmic_rays: int = 0):
        with tempfile.TemporaryDirectory() as td:
            cfg = SyntheticConfig(seed=seed, n_cosmic_rays=cosmic_rays)
            paths, truth_path = generate(cfg, td)
            refcat = load_reference_catalog(truth_path)
            meta = ObsMetadata(observer="astrohunt web demo", submitter="astrohunt web demo")
            result = run_pipeline(paths, reference_catalog=refcat, use_online=False,
                                  obs_meta=meta)
            frames = load_frame_set(paths, allow_no_wcs=True)
            return _payload(frames.frames, result, meta)

    @app.post("/api/detect")
    async def detect(
        files: List[UploadFile] = File(...),
        offline: bool = Form(True),
        hint_ra: Optional[float] = Form(None),
        hint_dec: Optional[float] = Form(None),
        obscode: str = Form("XXX"),
        observer: str = Form("astrohunt web"),
        ref_catalog: Optional[UploadFile] = File(None),
    ):
        if len(files) < 2:
            raise HTTPException(400, "Upload at least 2 FITS frames of the same field.")
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for f in files:
                p = os.path.join(td, os.path.basename(f.filename or "frame.fits"))
                with open(p, "wb") as out:
                    out.write(await f.read())
                paths.append(p)

            refcat = None
            if ref_catalog is not None:
                rp = os.path.join(td, "ref.json")
                with open(rp, "wb") as out:
                    out.write(await ref_catalog.read())
                refcat = load_reference_catalog(rp)

            meta = ObsMetadata(obs_code=obscode, observer=observer,
                               measurer=observer, submitter=observer)
            hint = (hint_ra, hint_dec) if (hint_ra is not None and hint_dec is not None) else None
            try:
                result = run_pipeline(
                    paths, reference_catalog=refcat,
                    use_online=not offline, hint_radec=hint, obs_meta=meta,
                )
            except Exception as e:  # noqa: BLE001
                raise HTTPException(422, f"Pipeline error: {e}")
            frames = load_frame_set(paths, allow_no_wcs=True)
            return _payload(frames.frames, result, meta)

    return app


# Module-level app for `uvicorn astrohunt.webapi:app` / Vercel. Guarded so the
# package still imports cleanly when FastAPI isn't installed.
try:
    app = create_app()
except Exception:  # pragma: no cover
    app = None
