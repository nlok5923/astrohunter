# astrohunt — Technical Documentation

How astrohunt turns a set of telescope images into MPC-ready asteroid
observations. This is the "how it works" companion to [`README.md`](../README.md)
(usage) and [`RESEARCH.md`](../RESEARCH.md) (why these choices).

---

## 1. The problem

Given **N images of the same patch of sky** taken minutes apart, find the objects
that *move* (asteroids, comets) against the *fixed* background (stars, galaxies),
measure their positions precisely enough for the **Minor Planet Center (MPC)**,
and emit a report a human can review and submit.

The MPC's real acceptance bar (see RESEARCH.md) is:
- **Sub-arcsecond *consistency*** — repeatable positions measured against the same
  reference stars (not an absolute ephemeris tolerance).
- A **proper least-squares plate-constants (LSPC)** solution — deriving positions
  "by eye" or by chart overlay is explicitly forbidden.
- **Gaia DR2/DR3** as the astrometric reference catalogue.
- Output in **ADES** (XML or PSV), validated server-side against `submit.xsd`.

astrohunt is built around hitting that bar automatically and cross-platform.

---

## 2. Pipeline overview

```
                 ┌─────────────┐
  FITS frames ──►│   ingest    │  io.py        (data, WCS, mid-exposure time)
                 └──────┬──────┘
        (no WCS?)       ▼
                 ┌─────────────┐
                 │ plate-solve │  platesolve.py (asterism/triangle matching)
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │   extract   │  extract.py    (DAOStarFinder + centroiding)
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │   vetting   │  vet.py / mlvet.py  (reject cosmic rays etc.)
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │ LSPC astrom │  astrometry.py (gnomonic + plate constants)
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │ suppress +  │  detect.py     (kill static sky, link tracklets)
                 │   link      │
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │ known-match │  catalog.py    (SkyBoT / Gaia)
                 └──────┬──────┘
                        ▼
              ADES PSV/XML + obs80 + HTML review   ades.py / validate.py / review.py
```

A separate path, **shift-and-stack** (`stack.py`), recovers movers too faint to
detect in any single frame.

---

## 3. Stage by stage

### 3.1 Ingest — `io.py`
Loads each FITS frame into a `Frame` (pixel `data`, an Astropy `WCS`, and the
mid-exposure `Time` from `DATE-OBS` + `EXPTIME/2`). Frames are sorted by time so
"frame 0" is the earliest. `allow_no_wcs=True` defers solving to the next stage.

### 3.2 Plate-solving (headerless FITS) — `platesolve.py`
Survey frames usually carry a WCS; raw frames don't. We implement the core idea
behind astrometry.net — **asterism (triangle) matching** — given a coarse
pointing hint:

1. Take the brightest ~35 detected stars and ~70 catalogue stars (projected to a
   tangent plane around the hint).
2. For every triangle, compute a **scale/rotation/reflection-invariant
   descriptor**: the two side-length ratios `(mid/long, short/long)`, with
   vertices ordered by opposite-side length so correspondences line up. Thin
   (near-collinear) triangles are dropped.
3. Match descriptors with a KD-tree. **Scale-consistency filter:** every candidate
   triangle match implies a pixel scale (`det_longest / cat_longest`); the true
   matches all share one scale, so we histogram the implied scales, keep only the
   dominant peak, and discard the rest. *(This single step is what makes it
   robust — without it, accidental descriptor collisions dominate.)*
4. Vote star↔star correspondences from the surviving matches, greedily assign
   one-to-one, and fit a TAN WCS with `astropy.wcs.utils.fit_wcs_from_points`.
5. Verify by reprojecting the full catalogue and counting inliers.

On synthetic data this recovers the WCS to ~0.03″ from a pointing hint a couple
of arcminutes off.

### 3.3 Source extraction — `extract.py`
`photutils.DAOStarFinder` over a sigma-clipped background, then **sub-pixel
centroid refinement** (`centroid_com` on small cutouts) — centroid precision
directly limits astrometric accuracy. Returns a table of `x, y, flux, peak,
mag_inst, snr`, brightest first.

### 3.4 Astrometric reduction — LSPC — `astrometry.py`
The scientific heart, and what the MPC mandates.

1. **Match** detected stars to the reference catalogue (Gaia DR3) using the WCS as
   a first guess.
2. **Project** matched catalogue stars to **standard (tangent-plane) coordinates**
   about a plate centre `(α₀, δ₀)` via the gnomonic projection:
   ```
   cos c = sin δ₀ sin δ + cos δ₀ cos δ cos(α − α₀)
   ξ = cos δ sin(α − α₀) / cos c
   η = (cos δ₀ sin δ − sin δ₀ cos δ cos(α − α₀)) / cos c
   ```
3. **Least-squares fit** the plate constants mapping pixels → standard coords
   (order 1: `ξ = a·x + b·y + c`, `η = d·x + e·y + f`; higher orders add
   quadratic/cubic terms), via `numpy.linalg.lstsq`.
4. Any source's `(x, y)` → `(ξ, η)` → inverse gnomonic → `(α, δ)`.
5. The **RMS of the fit residuals (arcsec)** is the *consistency* metric the MPC
   grades on. A photometric zeropoint is derived from matched-star magnitudes.

### 3.5 Real-bogus vetting — `vet.py`, `mlvet.py`
Removes artifacts before they create false tracklets.
- **Rule-based** (`vet.py`, default): morphological features from each cutout
  (FWHM, elongation, sharpness, npix via image moments); reject anything too
  sharp/narrow (cosmic rays, hot pixels), too elongated (trails), or far from the
  image PSF width.
- **Learned** (`mlvet.py`, `--vetter ml`): a classifier trained on cutouts
  **auto-labelled** against the catalogue (known source → real; else bogus). Two
  backends behind one interface — a from-scratch **NumPy MLP** (default, no extra
  deps) and an optional **PyTorch CNN** (`[cnn]` extra). ~98% holdout accuracy on
  synthetic data. The feature vector is the normalised cutout plus three
  morphological summaries.

### 3.6 Static-source suppression + tracklet linking — `detect.py`
- **Suppression:** a detection recurring at the same sky position across most
  frames is static (star/galaxy) → dropped. What remains is transient/moving.
- **Linking:** seed a velocity from every pair of survivors in the two earliest
  frames; predict positions in the other frames; gather inliers within a tolerance
  (a light RANSAC). Keep tracklets of ≥3 points consistent with **uniform linear
  motion**, fit `α(t), δ(t)`, and report rate (″/min), position angle, and the
  straight-line residual.

### 3.7 Known-object matching — `catalog.py`
Cone-search **SkyBoT** (IMCCE) at the field/epoch; a tracklet near a known object
is labelled (not a discovery). Gaia DR3 (via Astroquery) supplies the astrometric
reference; everything degrades gracefully to offline when there's no network.

### 3.8 Shift-and-stack / digital tracking — `stack.py`
For movers below the single-frame threshold:
1. **Subtract the static sky** (per-pixel median across frames) so bright stars
   don't smear into streaks.
2. For a trial `(rate, PA)`, shift each frame by the implied pixel offset so a
   mover at that velocity re-aligns, then **co-add**.
3. The aligned signal stays put while noise averages down → **SNR ∝ √N**. A
   `(rate, PA)` **grid search** locks onto the true velocity (the top-SNR stack).

On synthetic data a faint asteroid invisible single-frame (SNR ≈ 6) is recovered
in the 4-frame stack (SNR ≈ 12, ~√4 = 2× gain).

### 3.9 Reporting & validation — `ades.py`, `validate.py`
- **ADES** writers: PSV (pipe-separated), XML (the MPC-preferred form), and legacy
  80-column obs80. Uses ADES **version 2022**; telescope **aperture** is required.
- **Validation:** a fast offline structural check (designations, ISO-8601
  `obsTime`, coordinate ranges, enums, placeholder-obscode warnings) **plus strict
  validation against the vendored official `submit.xsd`** (via lxml) — the same
  schema the MPC enforces.

### 3.10 Review — `review.py` + `web/`
- **Static HTML report** (`review.py`): a self-contained page with a **blink GIF**
  per candidate (the mover walks across a fixed starfield), accept/reject
  (mouse + keyboard, persisted in localStorage), and export-accepted → ADES PSV.
- **Web app** (`web/` + `webapi.py`): the same review UX, live, backed by the real
  pipeline (see §5).

---

## 4. Key data formats

- **FITS** — the astronomy image standard: a header (WCS keywords `CRPIX/CRVAL/CD…`,
  `DATE-OBS`, `EXPTIME`) plus a 2-D pixel array.
- **WCS** — World Coordinate System; the pixel ↔ sky `(RA, Dec)` mapping. astrohunt
  uses/produces TAN (gnomonic) WCS.
- **ADES** — the IAU Astrometry Data Exchange Standard. PSV example:
  ```
  # version=2022
  # observatory
  ! mpcCode XXX
  ...
  trkSub |mode|stn|obsTime                |ra        |dec       |rmsRA|rmsDec|mag |band|astCat
  aH00001|CCD |XXX |2026-06-08T10:00:30.0Z|187.504791|+11.987502|0.06 |0.06  |14.8|G   |Gaia3
  ```

---

## 5. Web architecture

```
Browser (Next.js 16 / React 19 / Tailwind, web/)
   │  fetch  ${NEXT_PUBLIC_API_URL}/api/...
   ▼
FastAPI (src/astrohunt/webapi.py — `astrohunt serve`, or Vercel api/index.py)
   │  GET  /api/demo     → offline synthetic run
   │  POST /api/detect   → multipart FITS upload + options
   ▼
run_pipeline(...)  →  candidates {blink GIF (base64), ADES rows, astrometry}
```

The frontend renders candidate cards, lets you accept/reject, and builds the
accepted-only ADES PSV **client-side** for download (`buildPSV`). Response shape
mirrors `review.py` so the static report and the web app share one data model.

---

## 6. Validation & metrics

There is no external test data dependency: `synthetic.py` generates reproducible
4-image FITS sets with a TAN WCS, Gaussian-PSF stars, Poisson + read noise, an
injected linear-motion asteroid, and optional cosmic-ray streaks (and an optional
`--no-wcs` mode to exercise plate-solving). The test suite (18 tests) drives the
whole pipeline against this ground truth.

Representative demo numbers (`astrohunt demo`):
- per-frame astrometric consistency **~0.28″**
- injected asteroid recovered to **~0.077″**, rate recovered exactly.

---

## 7. Limitations & assumptions

- **Synthetic-only validation.** Everything above is measured on synthetic data.
  Real Pan-STARRS/MAST frames (messy PSFs, real bad pixels, registration error)
  are the next milestone.
- **~4 frames.** Linking and stacking assume a short arc with uniform linear
  motion; shift-and-stack gains are modest (√4 ≈ 2×) at this frame count.
- **No true difference imaging.** Static-source suppression and median-subtraction
  cover the 4-frame case; difference imaging against a deep template was skipped
  (4-frame sets lack such a template — see RESEARCH.md).
- **Not a submitter.** astrohunt produces reports for a human to review; it does
  not submit to the MPC, and `XXX` is a placeholder observatory code.

---

## 8. Module reference

| module | role |
|---|---|
| `io.py` | FITS ingest, WCS, timestamps |
| `platesolve.py` | asterism/triangle plate-solving (scale-consistency filtered) |
| `extract.py` | source extraction + sub-pixel centroids |
| `astrometry.py` | gnomonic projection + LSPC plate constants |
| `vet.py` / `mlvet.py` | rule-based / learned real-bogus vetting |
| `detect.py` | static suppression + tracklet linking |
| `catalog.py` | Gaia DR3 + SkyBoT, offline fallback |
| `stack.py` | shift-and-stack / digital tracking |
| `ades.py` | ADES PSV/XML + obs80 writers |
| `validate.py` | structural + strict `submit.xsd` validation |
| `review.py` | interactive HTML review report |
| `webapi.py` | FastAPI backend |
| `synthetic.py` | synthetic data generator |
| `cli.py` | CLI entry points |
