"""Validation of generated ADES PSV reports.

The MPC validates submissions server-side against the W3C schema submit.xsd
(it converts PSV->XML first). We can't replicate that gate offline, but we can
catch the common acceptance blockers before a human ever submits: missing
designations, malformed timestamps, out-of-range coordinates, bad enums, and
placeholder observatory codes. This is a fast, dependency-free pre-flight check.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from astropy.time import Time

# Vendored official IAU ADES submission schema.
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schemas", "submit.xsd")

# Subset of ADES controlled vocabularies relevant to optical CCD astrometry.
_VALID_MODES = {"CCD", "CMO", "VID", "PHO", "ENC", "PMT", "MIC", "MER", "TDI", "OCC", "UNK"}
_RECOMMENDED_ASTCAT = {"Gaia2", "Gaia3", "Gaia1", "Gaia3E"}
_DESIG_COLUMNS = ("permID", "provID", "trkSub")
_REQUIRED_COLUMNS = ("mode", "stn", "obsTime", "ra", "dec")
_OBSTIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$"
)


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_rows: int = 0

    def report(self) -> str:
        lines = [f"ADES validation: {'PASS' if self.ok else 'FAIL'} "
                 f"({self.n_rows} observation row(s))"]
        for e in self.errors:
            lines.append(f"  [error] {e}")
        for w in self.warnings:
            lines.append(f"  [warn]  {w}")
        return "\n".join(lines)


def _parse_psv(text: str):
    """Return (metadata_lines, columns, rows) from an ADES PSV document."""
    meta, columns, rows = [], None, []
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        if line.startswith("#") or line.startswith("!"):
            meta.append(line)
            continue
        if columns is None:
            columns = [c.strip() for c in line.split("|")]
            continue
        values = [v.strip() for v in line.split("|")]
        rows.append(dict(zip(columns, values)))
    return meta, columns, rows


def validate_psv_text(text: str) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    meta, columns, rows = _parse_psv(text)

    if not any(m.replace(" ", "").startswith("#version=") for m in meta):
        errors.append("missing '# version=...' header line.")
    if columns is None:
        return ValidationResult(False, ["no PSV column-header line found."], warnings, 0)

    if not any(c in columns for c in _DESIG_COLUMNS):
        errors.append(
            f"no designation column ({'/'.join(_DESIG_COLUMNS)}) present."
        )
    for col in _REQUIRED_COLUMNS:
        if col not in columns:
            errors.append(f"required column '{col}' missing.")

    if not rows:
        errors.append("no observation data rows.")

    for i, row in enumerate(rows, 1):
        # designation present (at least one of the three, non-empty)
        if not any(row.get(c, "").strip() for c in _DESIG_COLUMNS):
            errors.append(f"row {i}: no designation (permID/provID/trkSub all empty).")

        ot = row.get("obsTime", "")
        if not _OBSTIME_RE.match(ot):
            errors.append(f"row {i}: obsTime '{ot}' is not ISO-8601 UTC with 'Z'.")
        else:
            try:
                Time(ot.replace("Z", ""), scale="utc")
            except Exception:
                errors.append(f"row {i}: obsTime '{ot}' not parseable.")

        for key, lo, hi in (("ra", 0.0, 360.0), ("dec", -90.0, 90.0)):
            val = row.get(key, "")
            try:
                f = float(val)
                if not (lo <= f <= hi):
                    errors.append(f"row {i}: {key}={f} out of range [{lo}, {hi}].")
            except ValueError:
                errors.append(f"row {i}: {key}='{val}' is not numeric.")

        mode = row.get("mode", "")
        if mode and mode not in _VALID_MODES:
            errors.append(f"row {i}: mode '{mode}' not a valid ADES mode.")

        stn = row.get("stn", "")
        if not stn:
            errors.append(f"row {i}: empty observatory code (stn).")
        elif stn == "XXX":
            warnings.append(
                f"row {i}: stn 'XXX' is a placeholder — a real assigned MPC "
                f"observatory code is required for actual submission."
            )

        for key in ("rmsRA", "rmsDec", "mag"):
            v = row.get(key, "")
            if v:
                try:
                    fv = float(v)
                    if key.startswith("rms") and fv < 0:
                        errors.append(f"row {i}: {key}={fv} must be non-negative.")
                except ValueError:
                    errors.append(f"row {i}: {key}='{v}' is not numeric.")

        ac = row.get("astCat", "")
        if ac and ac not in _RECOMMENDED_ASTCAT:
            warnings.append(
                f"row {i}: astCat '{ac}' is not a Gaia catalogue; the MPC "
                f"recommends Gaia DR2/DR3."
            )

    return ValidationResult(len(errors) == 0, errors, warnings, len(rows))


def validate_psv_file(path: str) -> ValidationResult:
    with open(path) as f:
        return validate_psv_text(f.read())


def validate_xsd(xml_text: str, xsd_path: str = SCHEMA_PATH) -> ValidationResult:
    """Strict validation of ADES XML against the official submit.xsd (needs lxml).

    This is the same schema the MPC validates against server-side, so passing it
    is a strong guarantee the XML is structurally acceptable.
    """
    try:
        from lxml import etree
    except Exception:
        return ValidationResult(
            False, [], ["lxml not installed; skipped strict submit.xsd validation."], 0
        )
    if not os.path.exists(xsd_path):
        return ValidationResult(False, [f"schema not found at {xsd_path}."], [], 0)
    try:
        schema = etree.XMLSchema(etree.parse(xsd_path))
        doc = etree.fromstring(xml_text.encode())
        if schema.validate(doc):
            return ValidationResult(True, [], [], 0)
        errs = [f"line {e.line}: {e.message}" for e in schema.error_log]
        return ValidationResult(False, errs, [], 0)
    except Exception as e:  # noqa: BLE001
        return ValidationResult(False, [f"XSD validation error: {e}"], [], 0)


def validate_xsd_file(xml_path: str, xsd_path: str = SCHEMA_PATH) -> ValidationResult:
    with open(xml_path) as f:
        return validate_xsd(f.read(), xsd_path)
