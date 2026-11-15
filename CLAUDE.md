# astrohunt — project guide

A modern, **cross-platform, CPU-first** successor to Astrometrica for
citizen-science asteroid detection (IASC-style). Give it a set of telescope
images of the same field taken minutes apart → it finds the moving objects and
produces the **MPC 80-column report IASC asks for** (and ADES). Runs natively on Apple Silicon (arm64), no
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
pytest -q                                                     # 25 tests
astrohunt serve --open                                        # IASC workspace (browser)
astrohunt iasc path/to/set --prefix ABC                       # headless IASC first pass

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
| `measure.py` | click-to-measure: PSF fit -> centroid, FWHM, SNR, aperture flux |
| `mpcreport.py` | MPC 80-column report (Astrometrica `MPCReport.txt` format) + checks |
| `workspace.py` | interactive session: settings presets (PS1/PS2), reduce, detect, measure, known overlay, stack, report |
| `static/workspace.html` | single-file browser workspace served by FastAPI at `/` (no Node needed) |
| `webapi.py` | FastAPI backend (`/api/demo`, `/api/detect`, `/api/ws/*` workspace) |
| `synthetic.py` | offline synthetic 4-image generator (stars + mover + cosmic rays) |
| `cli.py` | CLI: demo, detect, iasc, solve, stack, train-vetter, serve, synth |
| `schemas/submit.xsd` | vendored official IAU ADES schema |

**Web UI:** `web/` (Next.js 16 + React 19 + Tailwind) → FastAPI backend.

## Status — all phases complete ✅ (validated on **synthetic** data)
- **Phase 1** — deterministic core (ingest → LSPC → tracklets → ADES)
- **Phase 1.5** — plate-solving, vetting, ADES validation, HTML review
- **Phase 2** — learned vetter (~98% acc), strict `submit.xsd`, interactive review app
- **Phase 3** — shift-and-stack / digital tracking
- **Web UI** — Next.js + FastAPI app (local verified end-to-end)

Demo metrics: astrometric consistency ~0.28″; injected asteroid recovered to ~0.077″.

- **IASC workspace (2026-11)** — Astrometrica-style workflow (`astrohunt serve --open`,
  `astrohunt iasc <set>`), validated on IASC's 5 public PS1 practice sets: ~0.05″ fit
  RMS, all catalogued asteroids brighter than ~V 21.3 found, no false candidates.

## Remaining / open
1. **Real-data milestone:** done for IASC PS1 practice sets (see above). Not yet tried on
   a live campaign set, PS2 frames, or non-Pan-STARRS data.
2. **Vercel deploy:** scaffolded (`api/index.py`, `requirements.txt`, `vercel.json`,
   `web/DEPLOY.md`) but **not yet deploy-tested** (heavy scientific stack vs. function size).
3. **IASC integration (non-code):** confirm whether a non-Astrometrica tool can be
   credited inside official IASC campaigns — email IASC.
4. **Phase-2 follow-ups:** the rule-based/MLP vetter can graduate to a deeper CNN.

## IASC / Astrometrica compatibility facts (web research, 2026-11-14)
Verified against IASC's own downloads (installer, Quick Start Guide, practice sets
at `iasc.cosmosearch.org/Home/Astrometrica`) unless marked otherwise.

**What IASC ships:** Astrometrica **4.10.6.453** (32-bit x86, installer updated
2026-04) with `PS1.cfg` / `PS2.cfg`. This build writes the **80-column
`MPCReport.txt` only** (no ADES — upstream added ADES in 4.13) and its newest
catalog is **Gaia DR1** (IASC FAQ: "Select GAIA DR1").

**IASC workflow:** load 4 FITS → Data Reduction (if it fails: "Automatic Reference
Star Match") → Known Object Overlay → Blink → click object on each image →
Object Designation lookup → Accept → File > View MPC Report → paste into the
team page → Reset Files. One report per image set, submitted only on the IASC site.

**PS1.cfg key values:** `MPCCode=F51` (PS2: `F52`), `FocalLength=8070.0`,
`PixelWide/High=0.0100` (→ 0.2556"/px, unbinned), `PA=323.0` (PS2: `273.8`),
`ApertureRadius=5`, `ApertureGap=1`, `SigmaLimit=4.0`, `MinFWHM=1.20`,
`FitRMS=0.25`, `SearchRadius=0.75`, `Saturation=25000`, `Catalog=10` (Gaia DR1,
index inferred), `Color=G`, `FitOrder=3`, `MinMag=25.0`, `MaxMag=13.0`,
`PosResidual=0.40`, `MagResidual=0.70`, `NoMatch=100`, `MatchRadius=2.00`,
`ExtraDigit=1`, `TimeMode=0`, `ExpTimeMode=1`, `FlipHoriz=0`, `FlipVert=0`,
site `Longitude=150.4170 Latitude=20.7070 Height=3066.0`.

**IASC PS1 FITS (practice set `ps1-20190828_1_XY25_p10`):** plain uncompressed
single-HDU `.fits`, `BITPIX=16`, `BZERO=32768`, **2423 × 2434**, 4 frames ~18.7 min
apart, `EXPTIME=45.`. Time = **`DATE-OBS` (UTC start of exposure)**; no `MJD-OBS`.
`SHUTOPEN` is **TAI (+37 s)** — do not use it as UTC. Approximate old-style WCS is
present (`CTYPE RA---TAN/DEC--TAN`, `CRVAL`, `CRPIX`, `CDELT≈7.12E-05`, `CROTA`;
no CD matrix, no distortion), plus `OBJCTRA`/`OBJCTDEC`. File names look like
`o8723g0213o.1515750.ch.2211509.XY25.p10.fits`. Images have gray chip-gap grid areas.

**Report header (verbatim from IASC screenshots, 2020):**
```
COD F51
OBS J. Bulger, T. Lowe, A. Schultz, M. Willman
MEA T. Vorobjov, PS1 Science Consortium
TEL 1.8-m f/4.4 Ritchey-Chretien + CCD
ACK MPCReport file updated 2020.02.15 10:58:07
NET PPMXL
     NEW0001  C2019 10 24.58871204 11 48.498+14 24 22.67         21.0 R      F51
----- end -----
```
- OBS/MEA/TEL text matches the 2026 `PS1.cfg`. No CON/AC2/COM lines. IASC: do not
  edit the MEA line or the measurement lines.
- **Unverified:** current NET/band. All verbatim examples found (2018–2021) show
  `NET PPMXL` + band `R`; the 2026 cfg implies `NET Gaia DR1` + band `G`. No
  evidence for band `w`.
- Empty set: keep the header, replace measurements with the line
  `"No moving objects detected"`, then `----- end -----`.

**80-column line** (MPC `iau/info/OpticalObs.html`): 1–5 packed number · 6–12
designation · 13 discovery `*` · 14 note 1 · 15 note 2 (`C` = CCD) · 16–32 date
`YYYY MM DD.dddddd` · 33–44 RA `HH MM SS.ddd` · 45–56 Dec `sDD MM SS.dd` · 57–65
blank · 66–71 mag (F5.2) + band · 72–77 blank · 78–80 obs code. No TABs. IASC
lines use 6-decimal day, 3-decimal RA sec, 2-decimal Dec sec, mag to 0.1.

**Rules:** new objects named **3 letters + 4 digits** (`ABC0001`), unique per
campaign, alphanumeric only (7 chars = MPC absolute max). Known object = top
lookup entry with dRA/dDec < ~0.2; known objects need not be reported. True
signature = straight line, constant speed, roughly constant magnitude (reject if
it fluctuates by > 1 mag); never measure saturated objects or guess positions;
**3 measurements are enough**. "SNR > 5" comes from a third-party guide, not IASC.
No IASC "fit RMS < 0.2" rule was found.

**Still open:** IASC publishes no statement on reports made with other software
(FAQ only says Astrometrica is Windows-only and the IASC build is "tailored to
IASC"). Submission is a pasted text box. Emailing IASC (item 3 above) is still needed.

**Astrometrica on Apple Silicon:** one documented success —
`github.com/adityaratnam09/astrometrica-apple-silicon` (M1, macOS 15.7.4,
Sikarugir 1.0.1 engine WS12WineSikarugir + Rosetta 2, full IASC workflow). Needs
`mkdir %localappdata%\Astrometrica`, manual `MPCORB.DAT`, VizieR mirror set to CDS.
Whisky and Porting Kit failed there; CrossOver untested.

## IASC practice run — `ps1-NewPractice_3` (2026-11-15)
- Set lives at `../XY54_p10/` (4 frames, 2025-09-16 10:02–10:50 UTC, chip XY54);
  report saved as `../MPCReport_ps1-NewPractice_3.txt` (not yet submitted).
- `astrohunt iasc ../XY54_p10 --include-known`: fit RMS 0.06–0.08", one mover =
  known Mars-crosser **2018 LT** (`K18L00T`), 0.71"/min, G≈20.0, measured on images
  1, 2, 4. Image 3 skipped — the object sits on a chip-gap edge. No new candidates.
- SkyBoT's geocentric prediction was ~7" from the measured position (parallax);
  the match rests on rate/PA agreement.
- The report says `NET Gaia DR3` / band `G`; IASC's Astrometrica would say Gaia DR1.
- **Catalogue outage:** the ESA Gaia archive and all TAP VizieR mirrors in
  `catalog.VIZIER_SERVERS` timed out that day. `vizier.cfa.harvard.edu/viz-bin/asu-tsv`
  (`-source=I/355/gaiadr3`) worked and was used via a monkeypatch of
  `catalog._gaia_esa`; the field is now cached in `~/.cache/astrohunt`. Adding this
  mirror to `catalog.py` as a fallback is still to do.

## Astrometrica on this Mac (set up 2026-11-15)
Decision: **submit IASC reports from real Astrometrica; use astrohunt as the finder**
(we do not disguise astrohunt output as Astrometrica's).
- Lives in `~/Applications/Astrometrica-Wine/`: `engine/` (Sikarugir engine
  `WS12WineSikarugir10.0_8`, Wine 10.0, x86_64 via Rosetta 2), `template/` (Sikarugir
  Template 1.0.27 — only its `Contents/Frameworks` dylibs are used), `prefix/` (Wine
  prefix), `env.sh`, `pdl.sh` (parallel ranged downloader; GitHub release downloads
  are very slow on this network).
- **Launch:** double-click `~/Applications/Astrometrica-Wine/Astrometrica.command`.
  Files: `~/Applications/Astrometrica-Wine/Astrometrica-files` → `C:\Astrometrica`.
- Installed IASC build 4.10.6.453 with `msiexec /i ... /qn`; verified it opens
  (trial: "100 days left for evaluation" — no IASC licence key entered yet).
- Done by hand: `AppData/Local/Astrometrica` folder created; `MPCORB.DAT`
  (2026-11-15) placed in `Catalogs/`; practice frames copied to `Data/NewPractice_3/`.
- The .NET "IASC Astrometrica Launcher" is not used (it only refreshes MPCORB);
  refresh `MPCORB.DAT` manually. Sikarugir Creator itself was not installed (the brew
  cask download failed) — the tap `sikarugir-app/sikarugir` is tapped and trusted.
- `VizierServer` in `PS1.cfg`/`PS2.cfg` changed to `vizier.cds.unistra.fr` (shipped
  values `webviz.u-strasbg.fr` is dead); originals kept as `*.cfg.iasc-original`.
- **Blocker seen 2026-11-15:** CDS VizieR answered "Page unavailable" (404) and the
  reachable mirrors (`vizier.cfa.harvard.edu` https-only, `vizier.idia.ac.za`) hold
  PPMXL/UCAC4/USNO-B1/CMC15 but **no Gaia DR1** rows — so Astrometrica's Data
  Reduction with Gaia DR1 cannot run until CDS is back. Not yet tested end-to-end here.

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
- **PS1 headers:** the header WCS is only approximate (several arcsec, slight rotation) and
  the header has private `PCA1X..` keywords that break `astropy.wcs.WCS(header)` — `io.py`
  builds the WCS from whitelisted cards and `platesolve.refine_wcs` locks it onto Gaia.
  Use `DATE-OBS` (UTC); `MJD-OBS`/`SHUTOPEN` are TAI (+37 s).
- Workspace sessions live in server memory (`workspace._SESSIONS`) — local app only, not
  for the stateless Vercel deploy. Catalogues are cached in `~/.cache/astrohunt`.
- SkyBoT positions for old epochs can be several arcsec off; the overlay is a guide and
  identification also requires the motion to agree.
- `test_vetting_keeps_real_sources` fails with current photutils (80 vs >= 85 stars
  kept); it predates the workspace work.

## Test layout (`tests/`, 18 passing)
`test_pipeline.py` (end-to-end + ADES), `test_features.py` (plate-solve, validation,
vetting, review, strict XSD), `test_ml.py` (learned vetter + CNN-if-torch),
`test_stack.py` (faint-mover recovery + grid search).
