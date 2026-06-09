# astrohunt 🔭

A modern, **cross-platform, CPU-first** successor to Astrometrica for
citizen-science asteroid detection (IASC-style). Give it a set of telescope
images of the same field taken minutes apart; it finds the moving objects and
produces an **MPC-ready ADES report** — no GPU, runs natively on Apple Silicon.

> Design grounding: see [`RESEARCH.md`](RESEARCH.md) (deep-research synthesis,
> adversarially fact-checked). The MVP deliberately nails the *trusted,
> deterministic core* first; ML real-bogus vetting and a review UI are Phase 2.

## Why this exists
Astrometrica is the incumbent IASC tool but is **Windows-only** and relies on
**manual eyeball blinking**. `astrohunt` removes both bottlenecks: it's pure
Python on the Astropy stack (runs on macOS/arm64, Linux, Windows) and automates
the detection, leaving the human to *vet* ranked candidates rather than hunt.

## Pipeline
```
ingest FITS → [plate-solve if no WCS] → extract sources (photutils)
→ real-bogus vetting → LSPC astrometric solution → static-source suppression
→ tracklet linking → known-object match (SkyBoT)
→ ADES XML + ADES PSV + legacy obs80  +  HTML blink-review
```
- **Plate-solving** — frames without a WCS are solved automatically via
  asterism (triangle-invariant) matching against the reference catalogue, given
  a coarse pointing hint (`--hint-ra/--hint-dec`). Or solve standalone:
  `astrohunt solve raw.fits --hint-ra … --hint-dec …`.
- **LSPC astrometry** — a *proper least-squares plate-constants* fit against
  Gaia DR3 (the MPC mandates LSPC and forbids by-eye positions). The fit RMS is
  the sub-arcsecond *consistency* the MPC actually grades on.
- **Real-bogus vetting** — drops cosmic rays, hot pixels, and trails before they
  form false tracklets. Two backends: CPU-only morphological **rules** (default)
  or a **learned model** (`--vetter ml`) — a from-scratch NumPy MLP or an
  optional PyTorch CNN, trained on cutouts auto-labelled against the catalogue
  (`astrohunt train-vetter`, ~98% accuracy on synthetic data).
- **Reports** — ADES XML + PSV (the MPC-preferred formats) and legacy obs80,
  with an offline structural validator **and strict validation against the
  official IAU `submit.xsd`** (the same schema the MPC enforces).
- **Visual review** — a self-contained **interactive HTML app**: per-candidate
  blink GIF, accept/reject (mouse or keyboard, persisted), and **export-accepted
  → ADES PSV** straight from the browser. No server.
- **Shift-and-stack** (`astrohunt stack`) — digital tracking that co-adds frames
  along a velocity hypothesis to pull out movers below the single-frame
  threshold (~√N SNR gain), with a (rate, PA) grid search.

## Install (Apple Silicon / macOS)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```
All dependencies (numpy, scipy, astropy, photutils, astroquery) ship native
arm64 wheels — no Rosetta, no GPU.

## Quick start — run the offline demo
Generates a synthetic 4-image set with a known injected asteroid, runs the full
pipeline, and validates the recovery against ground truth:
```bash
astrohunt demo
```

Run on your own frames:
```bash
# online (fetches Gaia DR3 + SkyBoT automatically); writes all report formats + review
astrohunt detect frame_00.fits frame_01.fits frame_02.fits frame_03.fits \
    --obscode XXX --observer "Your Name" \
    --out-ades report.psv --out-ades-xml report.xml --out-html review.html

# raw frames without a WCS — give a coarse pointing hint to auto plate-solve
astrohunt detect *.fits --hint-ra 187.5 --hint-dec 12.0 --out-html review.html

# fully offline with a local reference catalogue
astrohunt detect *.fits --offline --ref-catalog truth.json --out-ades report.psv
```

Plate-solve a single raw frame (writes the WCS into the header):
```bash
astrohunt solve raw.fits -o solved.fits --hint-ra 187.5 --hint-dec 12.0
```

Train and use a learned real-bogus vetter:
```bash
astrohunt train-vetter -o vetter.pkl --backend mlp        # or --backend cnn (PyTorch)
astrohunt detect *.fits --vetter ml --vet-model vetter.pkl --out-html review.html
```

Hunt for a faint mover with shift-and-stack (digital tracking):
```bash
astrohunt stack *.fits --grid --rate-min 0.5 --rate-max 3 --pa-step 15 --threshold 6
```

Generate test data (optionally headerless or with artifacts):
```bash
astrohunt synth -o data/synthetic                       # clean, with WCS
astrohunt synth -o data/raw --no-wcs                     # to exercise plate-solving
astrohunt synth -o data/cr  --cosmic-rays 10             # to exercise vetting
```

## Tests
```bash
pip install -e ".[dev]"
pytest -q
```

## Status & roadmap
- **Phase 1 — deterministic core:** ingest → LSPC → tracklets → ADES, validated
  on synthetic data. ✅
- **Phase 1.5 — robustness & UX:** asterism plate-solving for headerless FITS,
  morphological real-bogus vetting, ADES XML + PSV validation, and an HTML
  blink-review UI. ✅
- **Phase 2 — smarts & rigor:** learned real-bogus vetter (NumPy MLP + optional
  PyTorch CNN, auto-labelled training), strict `submit.xsd` validation, and an
  interactive browser review app with accept/reject + ADES export. ✅
- **Phase 3 — faint movers:** shift-and-stack / digital tracking with a (rate,
  PA) grid search (~√N gain — modest on 4-frame sets, as RESEARCH.md predicted). ✅
- **Next — the real-world milestone:** validate the whole pipeline on **real
  Pan-STARRS / MAST FITS** (messy PSFs, real bad pixels, registration error), and
  resolve whether a non-Astrometrica tool can be credited inside official IASC
  campaigns.

⚠️ `astrohunt` produces reports for a human to review; it does **not** submit to
the MPC. The `XXX` observatory code is a placeholder — real submissions require
an assigned MPC code. Whether a non-Astrometrica tool can be credited *inside*
official IASC campaigns is an open question (see RESEARCH.md §Open questions).

## License
MIT
