# astrohunt — project guide

A modern, **cross-platform, CPU-first** successor to Astrometrica for
citizen-science asteroid detection (IASC-style). Give it a set of telescope
images of the same field taken minutes apart → it finds the moving objects and
produces **MPC-ready ADES reports**. Runs natively on Apple Silicon (arm64), no
GPU required.

- **Repo:** https://github.com/nlok5923/astrohunter (private, branch `main`)
- **Research grounding:** [`RESEARCH.md`](RESEARCH.md) (deep-research, adversarially fact-checked)
- **Usage:** [`README.md`](README.md) · **Web deploy:** [`web/DEPLOY.md`](web/DEPLOY.md)

## Why this exists
It started as a question — *is NASA's "find an asteroid and name it" thing real?*
Yes: the **International Astronomical Search Collaboration (IASC)**, a NASA-partnered
citizen-science program using Pan-STARRS data; discoveries flow to the Minor
Planet Center (MPC) and earn naming rights over ~3–5 years. The incumbent tool,
**Astrometrica**, is Windows-only and relies on manual eyeball "blinking" — so we
built a cross-platform, automated alternative.

## How to run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

astrohunt demo                 # offline synthetic proof-of-life (recovers an injected asteroid)
astrohunt detect *.fits --out-ades report.psv --out-html review.html
astrohunt solve raw.fits --hint-ra 187.5 --hint-dec 12.0     # plate-solve headerless FITS
astrohunt train-vetter -o vetter.pkl                          # learned real-bogus vetter
astrohunt stack *.fits --grid                                 # shift-and-stack faint movers
pytest -q                                                     # 18 tests

# Web UI:
pip install -e ".[web]" && astrohunt serve                    # backend → :8000
cd web && npm install && npm run dev                          # frontend → :3000
```

## Architecture (`src/astrohunt/`)
Pipeline: `ingest → [plate-solve if no WCS] → extract → real-bogus vetting →
LSPC astrometry → static-source suppression → tracklet linking → known-object
match → ADES XML/PSV + obs80 + HTML review`.

| module | role |
|---|---|
| `io.py` | FITS ingest, WCS, timestamps |
| `extract.py` | source extraction + sub-pixel centroids (photutils) |
| `platesolve.py` | asterism/triangle plate-solving for headerless FITS (scale-consistency filtered) |
| `astrometry.py` | **LSPC** solution — gnomonic projection + least-squares plate constants (MPC-mandated) |
| `detect.py` | static-source suppression + tracklet linking |
| `vet.py` | rule-based morphological real-bogus vetting |
| `mlvet.py` | learned vetter — NumPy MLP (default) or optional PyTorch CNN; auto-labeled cutouts |
| `catalog.py` | Gaia DR3 + SkyBoT (online) / offline reference catalog |
| `stack.py` | shift-and-stack / digital tracking (Phase 3, ~√N gain) |
| `ades.py` | ADES PSV + XML + legacy obs80 writers |
| `validate.py` | offline structural ADES validation + strict `submit.xsd` (lxml) |
| `review.py` | interactive HTML report (blink GIF, accept/reject, export) |
| `webapi.py` | FastAPI backend (`/api/demo`, `/api/detect`) |
| `synthetic.py` | offline synthetic 4-image generator (stars + mover + cosmic rays) |
| `cli.py` | CLI: demo, detect, solve, stack, train-vetter, serve, synth |
| `schemas/submit.xsd` | vendored official IAU ADES schema |

**Web UI:** `web/` (Next.js 16 + React 19 + Tailwind) → FastAPI backend.

## Status — all phases complete ✅ (validated on **synthetic** data)
- **Phase 1** — deterministic core (ingest → LSPC → tracklets → ADES)
- **Phase 1.5** — plate-solving, vetting, ADES validation, HTML review
- **Phase 2** — learned vetter (~98% acc), strict `submit.xsd`, interactive review app
- **Phase 3** — shift-and-stack / digital tracking
- **Web UI** — Next.js + FastAPI app (local verified end-to-end)

Demo metrics: astrometric consistency ~0.28″; injected asteroid recovered to ~0.077″.

## Remaining / open
1. **Real-data milestone:** run the whole pipeline on **real Pan-STARRS / MAST FITS**
   (messy PSFs, real bad pixels, registration error). Everything so far is synthetic.
2. **Vercel deploy:** scaffolded (`api/index.py`, `requirements.txt`, `vercel.json`,
   `web/DEPLOY.md`) but **not yet deploy-tested** (heavy scientific stack vs. function size).
3. **IASC integration (non-code):** confirm whether a non-Astrometrica tool can be
   credited inside official IASC campaigns — email IASC.
4. **Phase-2 follow-ups:** the rule-based/MLP vetter can graduate to a deeper CNN.

## Gotchas / conventions
- **`webapi.py` must NOT use `from __future__ import annotations`** — it breaks
  FastAPI's Form/File introspection (use `List`/`Optional` from `typing`).
- ADES output uses **version "2022"** and a telescope **aperture** is required for
  strict `submit.xsd` validation.
- Reference catalog defaults to **Gaia DR3**; the MPC standard is sub-arcsecond
  *consistency* (LSPC fit RMS), not an absolute tolerance.
- Reports are for a **human to review** — astrohunt does not submit to the MPC; the
  `XXX` observatory code is a placeholder.
- Default vetter is CPU-only; PyTorch is optional (`.[cnn]`), torch is excluded from
  the web `requirements.txt` to keep the serverless bundle lean.
- `data/` and `.venv/` are git-ignored; generated artifacts live in `data/`.

## Test layout (`tests/`, 18 passing)
`test_pipeline.py` (end-to-end + ADES), `test_features.py` (plate-solve, validation,
vetting, review, strict XSD), `test_ml.py` (learned vetter + CNN-if-torch),
`test_stack.py` (faint-mover recovery + grid search).
