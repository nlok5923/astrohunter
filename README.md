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
ingest FITS → extract sources (photutils) → LSPC astrometric solution
→ static-source suppression → tracklet linking → known-object match (SkyBoT)
→ ADES (PSV) + legacy obs80 report
```
The astrometric step is a **proper least-squares plate-constants (LSPC)** fit
against a Gaia DR3 reference (the MPC mandates LSPC and forbids by-eye
positions). The fit RMS is the sub-arcsecond *consistency* the MPC actually
grades on.

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
# online (fetches Gaia DR3 + SkyBoT automatically)
astrohunt detect frame_00.fits frame_01.fits frame_02.fits frame_03.fits \
    --obscode XXX --observer "Your Name" --out-ades report.psv

# fully offline with a local reference catalogue
astrohunt detect *.fits --offline --ref-catalog truth.json --out-ades report.psv
```

Generate test data only:
```bash
astrohunt synth -o data/synthetic
```

## Tests
```bash
pip install -e ".[dev]"
pytest -q
```

## Status & roadmap
- **Phase 1 (this MVP):** deterministic core — ingest → LSPC → tracklets →
  ADES, validated on synthetic data. ✅
- **Phase 2:** CNN real-bogus vetting, ranked-candidate review UI, plate-solving
  for headerless FITS (astrometry.net), submit.xsd validation.
- **Phase 3 (optional):** shift-and-stack / synthetic tracking for faint movers
  (modest gains on 4-frame sets — see RESEARCH.md).

⚠️ `astrohunt` produces reports for a human to review; it does **not** submit to
the MPC. The `XXX` observatory code is a placeholder — real submissions require
an assigned MPC code. Whether a non-Astrometrica tool can be credited *inside*
official IASC campaigns is an open question (see RESEARCH.md §Open questions).

## License
MIT
