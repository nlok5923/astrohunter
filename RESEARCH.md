# Research: A Modern Successor to Astrometrica (Citizen-Science Asteroid Detection)

> Deep-research synthesis, 2026-06-08. 23 sources, 104 claims extracted, 25 adversarially
> verified (23 confirmed / 2 refuted). Confidence levels and votes noted per finding.
> Goal: ground the design of a cross-platform, semi-automated, ADES-native tool for
> IASC-style asteroid hunting (4 FITS frames of a field, minutes apart → report new movers
> to the Minor Planet Center).

## TL;DR
A credible successor to Astrometrica is **technically feasible**, and the building blocks
already exist as open-source components. The proven pipeline is: plate-solve each frame
against **Gaia DR3** with **astrometry.net** → extract sources (**SExtractor/photutils**) →
suppress static sources → **link detections into tracklets** across the 4-image cadence →
fit positions with a proper **least-squares plate-constants (LSPC)** solution → match
known objects (**SkyBoT/MPC**) → emit **MPC-compliant astrometry (ADES)**. The real,
durable differentiators vs. Astrometrica are **cross-platform/web delivery** and
**automated detection** — NOT chasing magnitude depth, which is limited on 4-frame sets.

---

## 1. The proven detection pipeline (high confidence)
Standard professional NEO-survey workflow (Pan-STARRS, Catalina, ZTF), and exactly the
IASC use case: re-image a field ~4× within ~1 hour, link a mover's detections into a
**tracklet**, then attribute to a known object or report as a new candidate.
*(Source: Vereš/Payne et al. 2023, arXiv:2309.16407 — MPC-authored. Vote 3-0.)*

Pipeline stages a new tool must implement:
1. **Plate-solving / astrometric calibration** — astrometry.net for blind WCS; refine
   against Gaia DR3.
2. **Source extraction** — SExtractor or photutils/SEP.
3. **Static-source suppression** — cross-match across frames; remove fixed stars/galaxies.
4. **Tracklet detection/linking** — link sources moving linearly across the 4 frames.
5. **Astrometric reduction** — a *proper LSPC solution* (mandatory, see §3).
6. **Known-object matching** — SkyBoT / MPC ephemeris → "known" vs "new candidate".
7. **Report generation** — ADES (and/or legacy obs80).

## 2. State-of-the-art detection beyond eyeball blinking
- **Shift-and-stack / synthetic tracking ("digital tracking")**: co-add frames offset
  along an object's expected motion vector. **SNR scales as √N**; stacking hundreds of
  exposures pushes limits **a few magnitudes fainter**. Heinze et al. 2015 — "ten times
  fainter than conventional searches" (~2.5 mag).
  *(Sources: arXiv:2503.08854, 1309.3248, 1508.01599, 2210.01219. Vote 3-0 on √N / few-mag.)*
  ⚠️ **Caveat:** √N holds only in the background/read-noise-limited regime; gains are
  tempered by registration error and trajectory-search compute. **With only 4 IASC frames,
  the realistic depth gain is modest** — this is an enhancement, not the headline win.
- **Difference imaging** and **CNN real-bogus classifiers** (ZTF/DeepStreaks, Pan-STARRS,
  ATLAS, Rubin/LSST Solar System Processing) are the pro standard for artifact rejection —
  BUT difference imaging needs a **deep template/reference image** that a 4-frame
  citizen set does not have. (See Open Questions — this area is under-evidenced.)

## 3. MPC accuracy & submission standards (high confidence)
- **The standard is sub-arcsecond CONSISTENCY** (repeatability using the same comparison
  stars), **NOT a fixed absolute ephemeris tolerance.** A claim that the MPC requires
  positions within 2″ of predicted ephemeris was **REFUTED (vote 1-2)**.
- **A proper LSPC solution is effectively mandatory.** MPC: *"You must not attempt to
  derive positions by overlaying charts… or by estimating positions by eye"* and *"Be
  suspicious of any package that does not do a proper LSPC solution!"*
- **Reference catalog: default to Gaia DR3** (~1 mas quoted; ~0.4–0.5 mas measured at
  G=20, arXiv:2405.02017). UCAC4/UCAC5 acceptable. **B1950.0 catalogs and DSS FITS-header
  WCS MUST NOT be used.** Query via VizieR/astroquery.
- **Two formats:** legacy **obs80** (80-column MPC1992) and **ADES** (IAU standard, 2015),
  which has **two representations — XML and PSV** (pipe-separated values) and carries
  per-detection uncertainties. ADES is **dominant but not strictly mandatory**; preference
  order **XML > PSV > obs80**. obs80 may be emailed; **ADES is cURL/web submission only**.
- **The hard acceptance gate:** every submission is validated server-side against the W3C
  schema **`submit.xsd`** (PSV is converted to XML first). The MPC does conversion +
  validation themselves. **Target `submit.xsd` conformance in development to guarantee
  acceptance.** (Schema: github.com/IAU-ADES/ADES-Master.)

## 4. Existing tooling & the gap (high confidence)
- **ASTAP** — free, open-source (MPL 2.0), genuinely cross-platform (Win/Linux/macOS incl.
  **Apple Silicon**/RPi/Android), with a **built-in blink comparator** and plate-solving.
  ✅ Proves cross-platform eyeball-blink works everywhere. ❌ **Emits only obs80, not ADES.**
- **A-Track** — closest open-source architectural precedent: Python 3 + astropy, SExtractor,
  astrometry.net (Gaia index files), **MILD** tracklet detection, SkyBoT matching, MPC
  report output. Peer-reviewed (Atay et al. 2016). GPL v3. ❌ **Dormant since 2019**, legacy
  PyFITS, 80-column only → use as **design/license reference, not a base**.
- **Astropy ecosystem** (astropy, photutils, astroquery) + **astrometry.net** + **SkyBoT** =
  the realistic open-source backbone.
- **The gap:** *no* existing tool combines **cross-platform + semi-automated detection +
  education-friendly + ADES-native**. That is precisely the opening.

## 5. ML training-set bootstrap (medium confidence)
Use **MPC catalog / SkyBoT to auto-label known asteroids** in archival images → free
labeled data + a validation-against-known-objects strategy. The attribution mechanism
(ephemeris → pixel position → label) is **confirmed operational** in A-Track and the 2020
photometry pipeline. *Medium* because no surviving source explicitly described using those
labels to **train a CNN real-bogus classifier** — that step is a reasonable extrapolation.

---

## Refuted / unverified (do NOT rely on)
- ❌ "MPC requires positions within 2″ of predicted ephemeris; observatory code only after
  an accepted submission." (Vote 1-2.) Real standard = **sub-arcsecond consistency**.
- ❌ Specific Astrometrica internals (Gaussian PSF vs Gaia DR2, 4th-order polynomial plate
  constants) — **unverified** (vote 1-2). Don't assume Astrometrica's exact math.

## Open questions to resolve before/while building
1. **CNN real-bogus**: concrete published performance + compute for ZTF/DeepStreaks,
   Pan-STARRS, ATLAS, Rubin/LSST SSP — and **how they degrade on 4-image sets with no deep
   template** for difference imaging. (Under-evidenced; needs primary follow-up.)
2. **IASC integration**: Can a non-Astrometrica tool be **formally accepted in official IASC
   campaigns and credited toward MPC numbering/naming**, or only used **independently on
   public Pan-STARRS/MAST archive data**? IASC's data-delivery format/cadence and any
   tool-accreditation requirement were not established. **Resolve early (email IASC).**
3. **Tycho Tracker** exact synthetic-tracking/GPU capability, licensing, cross-platform —
   not characterized; do a head-to-head.
4. **Delivered astrometric residuals** (not catalog precision) from an automated
   astrometry.net + LSPC pipeline on real 4-frame Pan-STARRS sets — is it reliably within
   MPC sub-arcsecond consistency across varied fields and faint magnitudes?

## Caveats
- Time-sensitive: Astrometrica/IASC pages "updated April 2026" — re-verify before building
  IASC integration.
- ADES dominant but not mandatory; the **hard gate is `submit.xsd` conformance**, not ADES per se.
- A-Track is a precedent, not a maintainable base (dormant, legacy deps).

---

## Implied architecture & MVP (design takeaways)
**Phase 1 (MVP) — nail the trusted, deterministic core on public/sample FITS, cross-platform Python:**
`ingest 4 FITS → astrometry.net plate-solve → photutils source extraction → LSPC reduction
→ static-source suppression → tracklet linking → SkyBoT known-object match → ADES output
validated locally against submit.xsd`. Get *correct, MPC-acceptable output* first.

**Phase 2 — automation & UX:** CNN real-bogus vetting, ranked-candidate review UI
(smart blink), cross-platform/web packaging, education mode.

**Phase 3 — depth (optional):** shift-and-stack/synthetic tracking (expect modest gains on
4-frame sets), difference imaging if a reference template can be sourced.

**Key reframe from the research:** the durable win is **automation + cross-platform + ADES**,
not magnitude depth. The "advanced models" angle is **computer vision (CNN vetting)**, not
LLMs, and even that sits in Phase 2 behind a correct deterministic core.
