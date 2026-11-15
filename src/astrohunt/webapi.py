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
from . import workspace as wsmod
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
            "frames_png": review.candidate_frame_pngs(frame_list, trk),
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
        "field_gif": (review.field_blink_gif(frame_list, result.tracklets)
                      if result.tracklets else None),
        "ades": {"metaLines": ades_metadata_lines(meta), "columns": list(ADES_COLUMNS)},
        "candidates": candidates,
        "warnings": result.warnings,
    }


_EXPLAIN_SYSTEM = (
    "You are an assistant for citizen-science asteroid hunting (IASC-style). You are shown a "
    "few consecutive 'blink' frames of the same star field taken minutes apart, with a green "
    "circle marking a candidate moving object, plus its measured motion. Assess whether the "
    "marked object looks like a REAL moving solar-system object (asteroid/comet) or a BOGUS "
    "artifact (cosmic ray, hot pixel, satellite/trail, noise, or a misidentified star). "
    "Consider: does it appear as a consistent, point-like source that shifts smoothly and "
    "linearly across the frames while the background stars stay fixed? Is the rate plausible "
    "for an asteroid (main-belt objects move ~0.2-1 arcsec/min; NEOs faster)? Is the "
    "straight-line fit tight?\n\n"
    "Respond in plain text: a FIRST LINE that is exactly 'VERDICT: LIKELY REAL' or "
    "'VERDICT: LIKELY BOGUS' or 'VERDICT: UNCERTAIN', then 2-3 short sentences a curious "
    "beginner can understand explaining why. Be honest about uncertainty. End by reminding "
    "that this is an AI second opinion and a human (and the Minor Planet Center) must confirm "
    "any real discovery."
)


def create_app():
    from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
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
            # A richer, prettier field for the showcase demo (more stars, a few
            # bright ones, a clearly-visible mover).
            cfg = SyntheticConfig(
                seed=seed, n_cosmic_rays=cosmic_rays, n_stars=140,
                star_flux_range=(2.5e3, 1.3e5), asteroid_flux=1.8e4,
            )
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

    @app.post("/api/explain")
    def explain(body: dict = Body(...)):
        """AI second-opinion on a candidate (Claude vision). Degrades gracefully."""
        try:
            import anthropic
        except Exception:
            raise HTTPException(503, "AI explainer unavailable: the 'anthropic' package "
                                     "isn't installed on the server.")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise HTTPException(503, "AI explainer unavailable: set ANTHROPIC_API_KEY on "
                                     "the server to enable it.")

        images = body.get("images") or []
        if not images:
            raise HTTPException(400, "No candidate frames provided.")

        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b}}
            for b in images[:4]
        ]
        content.append({"type": "text", "text":
            "These are consecutive blink frames of the same sky patch, minutes apart. The "
            "green circle marks a candidate moving object. Measurements: on-sky rate "
            f"~{body.get('rate')} arcsec/min, position angle ~{body.get('pa')} deg, "
            f"straight-line fit RMS ~{body.get('rms')} arcsec, linked across "
            f"{body.get('n')} frames."})

        client = anthropic.Anthropic()
        try:
            msg = client.messages.create(
                model="claude-opus-4-8",
                max_tokens=2000,
                thinking={"type": "adaptive"},
                system=_EXPLAIN_SYSTEM,
                messages=[{"role": "user", "content": content}],
            )
        except anthropic.APIError as e:  # noqa: BLE001
            raise HTTPException(502, f"Claude API error: {e}")

        text = next((b.text for b in msg.content if getattr(b, "type", "") == "text"), "").strip()
        low = text.lower()
        verdict = None
        if "likely real" in low or "likely-real" in low:
            verdict = "likely-real"
        elif "likely bogus" in low or "likely-bogus" in low:
            verdict = "likely-bogus"
        elif "uncertain" in low:
            verdict = "uncertain"
        explanation = text
        if low.startswith("verdict:"):
            parts = text.split("\n", 1)
            explanation = parts[1].strip() if len(parts) > 1 else text
        return {"verdict": verdict, "explanation": explanation}

    # ------------------------------------------------------------------
    # Interactive workspace (the Astrometrica-style workflow). State lives in
    # this process, so these endpoints are for the local `astrohunt serve` app.
    # ------------------------------------------------------------------
    from fastapi.responses import HTMLResponse, Response

    def _ws(ws_id: str):
        try:
            return wsmod.get(ws_id)
        except KeyError as e:
            raise HTTPException(404, str(e.args[0]))

    def _run(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            raise HTTPException(422, str(e))

    def _settings(body: dict):
        st = dict(body.get("settings") or {})
        st.pop("preset", None)
        return wsmod.make_settings(body.get("preset") or "generic", **st)

    @app.get("/", response_class=HTMLResponse)
    def workspace_page():
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "static", "workspace.html"), encoding="utf-8") as f:
            return HTMLResponse(f.read(), headers={"Cache-Control": "no-store"})

    @app.get("/api/presets")
    def presets():
        from dataclasses import asdict

        return {name: asdict(wsmod.make_settings(name)) for name in wsmod.PRESETS}

    @app.post("/api/ws/open")
    def ws_open(body: dict = Body(...)):
        """Open an image set from a folder (or .zip) on this computer."""
        path = (body.get("path") or "").strip()
        if not path:
            raise HTTPException(400, "Give the folder (or .zip) that holds the FITS images.")
        ws = _run(wsmod.Workspace.from_folder, path, _settings(body))
        return wsmod.register(ws).info()

    @app.post("/api/ws/upload")
    async def ws_upload(files: List[UploadFile] = File(...), preset: str = Form("generic")):
        import shutil

        td = tempfile.mkdtemp(prefix="astrohunt-ws-")
        try:
            name = ""
            for f in files:
                base = os.path.basename(f.filename or "frame.fits")
                dest = os.path.join(td, base)
                with open(dest, "wb") as out:
                    while chunk := await f.read(1 << 20):
                        out.write(chunk)
                if base.lower().endswith(".zip"):
                    name = os.path.splitext(base)[0]
                    import zipfile

                    with zipfile.ZipFile(dest) as z:
                        for info in z.infolist():
                            b = os.path.basename(info.filename)
                            if info.is_dir() or b.startswith(".") or \
                                    not b.lower().endswith(wsmod.FITS_EXT):
                                continue
                            with z.open(info) as src, open(os.path.join(td, b), "wb") as dst:
                                shutil.copyfileobj(src, dst)
                    os.remove(dest)
            paths = wsmod.find_fits(td)
            if len(paths) < 2:
                raise ValueError("Upload at least 2 FITS images of the same field (or a .zip of them).")
            ws = wsmod.Workspace(paths, wsmod.make_settings(preset),
                                 name=name or "uploaded set", tmpdir=td)
        except Exception as e:  # noqa: BLE001
            shutil.rmtree(td, ignore_errors=True)
            raise HTTPException(422, str(e))
        return wsmod.register(ws).info()

    @app.get("/api/ws/{ws_id}")
    def ws_info(ws_id: str):
        return _ws(ws_id).info()

    @app.post("/api/ws/{ws_id}/settings")
    def ws_settings(ws_id: str, body: dict = Body(...)):
        ws = _ws(ws_id)
        from dataclasses import asdict

        cur = asdict(ws.settings)
        if body.get("preset"):
            cur.update(wsmod.PRESETS.get(body["preset"], {}))
        cur.update(body.get("settings") or {})
        known = wsmod.Settings.__dataclass_fields__
        ws.settings = _run(lambda: wsmod.Settings(**{k: v for k, v in cur.items() if k in known}))
        return ws.info()

    @app.post("/api/ws/{ws_id}/reduce")
    def ws_reduce(ws_id: str, body: dict = Body(default={})):
        ws = _ws(ws_id)
        hint = None
        if body.get("hint_ra") is not None and body.get("hint_dec") is not None:
            hint = (float(body["hint_ra"]), float(body["hint_dec"]))
        refcat = None
        if body.get("ref_catalog_path"):
            refcat = _run(load_reference_catalog, body["ref_catalog_path"])
        return _run(ws.reduce, refcat, hint)

    @app.get("/api/ws/{ws_id}/frame/{frame}.png")
    def ws_frame(ws_id: str, frame: str, lo: float = -2.0, hi: float = 12.0,
                 stretch: str = "linear"):
        ws = _ws(ws_id)
        key = "stack" if frame == "stack" else int(frame)
        png = _run(ws.render_png, key, lo, hi, stretch)
        return Response(png, media_type="image/png",
                        headers={"Cache-Control": "private, max-age=3600"})

    @app.post("/api/ws/{ws_id}/measure")
    def ws_measure(ws_id: str, body: dict = Body(...)):
        ws = _ws(ws_id)
        return _run(ws.measure, int(body["frame"]), float(body["x"]), float(body["y"]),
                    bool(body.get("snap", True)))

    @app.post("/api/ws/{ws_id}/predict")
    def ws_predict(ws_id: str, body: dict = Body(...)):
        ws = _ws(ws_id)
        ms = body.get("measurements") or []
        return {
            "positions": [_run(ws.predict, ms, i) for i in range(len(ws.fset))],
            "motion": wsmod.motion_summary(ms),
        }

    @app.post("/api/ws/{ws_id}/detect")
    def ws_detect(ws_id: str):
        ws = _ws(ws_id)
        return {"tracklets": _run(ws.detect), "known": _run(ws.known_objects)}

    @app.get("/api/ws/{ws_id}/known")
    def ws_known(ws_id: str):
        return {"known": _run(_ws(ws_id).known_objects)}

    @app.post("/api/ws/{ws_id}/stack")
    def ws_stack(ws_id: str, body: dict = Body(...)):
        return _run(_ws(ws_id).stack, float(body["rate"]), float(body["pa"]),
                    body.get("combine", "mean"))

    @app.post("/api/ws/{ws_id}/report")
    def ws_report(ws_id: str, body: dict = Body(...)):
        return _run(_ws(ws_id).report, body.get("objects") or [], body.get("header"))

    return app


# Module-level app for `uvicorn astrohunt.webapi:app` / Vercel. Guarded so the
# package still imports cleanly when FastAPI isn't installed.
try:
    app = create_app()
except Exception:  # pragma: no cover
    app = None
