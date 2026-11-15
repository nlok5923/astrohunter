"""MPC 80-column report — the text Astrometrica writes to MPCReport.txt.

This is the report IASC campaigns ask for (header lines + one 80-column line
per measurement + "----- end -----"). Column layout follows the MPC's "Format
For Optical Astrometric Observations Of Minor Planets":

    cols  1- 5  packed permanent number (blank if none)
    cols  6-12  packed provisional or temporary designation
    col  13     discovery asterisk
    col  14     note 1
    col  15     note 2 ('C' = CCD)
    cols 16-32  date of observation  "YYYY MM DD.ddddd "
    cols 33-44  RA  (J2000)          "HH MM SS.ss "
    cols 45-56  Dec (J2000)          "sDD MM SS.s "
    cols 57-65  blank
    cols 66-70  magnitude            "MM.M "
    col  71     band
    cols 72-77  blank
    cols 78-80  observatory code
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from astropy.time import Time

END_LINE = "----- end -----"

_B62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


@dataclass
class ReportHeader:
    cod: str = "XXX"                 # observatory code
    con: str = ""                    # contact (name, address [email])
    obs: str = ""                    # observers
    mea: str = ""                    # measurers
    tel: str = ""                    # telescope description
    ack: str = ""                    # acknowledgement subject line
    ac2: str = ""                    # acknowledgement e-mail
    net: str = ""                    # astrometric reference catalogue
    com: list[str] = field(default_factory=list)   # free comments

    def lines(self) -> list[str]:
        out = [f"COD {self.cod}"]
        for key in ("con", "obs", "mea", "tel", "ack", "ac2", "net"):
            val = getattr(self, key)
            if val:
                out.append(f"{key.upper()} {val}")
        out += [f"COM {c}" for c in self.com if c]
        return out


@dataclass
class ReportObs:
    mjd: float
    ra: float                        # deg, J2000
    dec: float                       # deg, J2000
    mag: float | None = None
    band: str = ""
    note1: str = " "
    note2: str = "C"
    discovery: bool = False


@dataclass
class ReportObject:
    designation: str                 # temporary ("ABC0001"), provisional ("2020 AB1") or number ("12345")
    observations: list[ReportObs]


# --- designation packing ----------------------------------------------------

def pack_number(n: int) -> str:
    """Pack a permanent minor-planet number into 5 columns."""
    if n < 100000:
        return f"{n:05d}"
    if n < 620000:
        return _B62[n // 10000] + f"{n % 10000:04d}"
    m = n - 620000
    s = ""
    for _ in range(4):
        s = _B62[m % 62] + s
        m //= 62
    return "~" + s


_PROV_RE = re.compile(r"^(\d{4})\s+([A-HJ-Y])([A-HJ-Z])(\d*)$")
_SURVEY_RE = re.compile(r"^(\d{4})\s+(P-L|T-[123])$")


def pack_provisional(desig: str) -> str | None:
    """'2020 AB1' -> 'K20A01B'; returns None if `desig` isn't a provisional one."""
    d = desig.strip()
    m = _SURVEY_RE.match(d)
    if m:
        return {"P-L": "PLS", "T-1": "T1S", "T-2": "T2S", "T-3": "T3S"}[m.group(2)] + m.group(1)
    m = _PROV_RE.match(d)
    if not m:
        return None
    year, h1, h2, cyc = m.group(1), m.group(2), m.group(3), int(m.group(4) or 0)
    century = {"18": "I", "19": "J", "20": "K", "21": "L"}.get(year[:2])
    if century is None or cyc > 619:
        return None
    cyc_s = f"{cyc:02d}" if cyc < 100 else _B62[cyc // 10] + str(cyc % 10)
    return f"{century}{year[2:]}{h1}{cyc_s}{h2}"


def designation_columns(desig: str) -> str:
    """Return the 12 characters for columns 1-12 of an observation line."""
    d = desig.strip()
    # "(12345)", "12345", "12345 Name" -> numbered object
    m = re.match(r"^\(?(\d{1,7})\)?(\s+\D.*)?$", d)
    if m and not _PROV_RE.match(d) and not _SURVEY_RE.match(d):
        return pack_number(int(m.group(1))) + " " * 7
    packed = pack_provisional(d)
    if packed:
        return " " * 5 + packed
    # already-packed provisional or a temporary designation: cols 6-12
    if len(d) > 7 or not d or not re.match(r"^[A-Za-z0-9_\-]+$", d):
        raise ValueError(
            f"designation {desig!r} must be 1-7 letters/digits (e.g. ABC0001), "
            f"a provisional designation (2020 AB1) or a number."
        )
    return " " * 5 + f"{d:<7}"


# --- field formatting --------------------------------------------------------

def format_date(mjd: float, extra_digit: bool = False) -> str:
    """'YYYY MM DD.ddddd ' (or 6 decimals with extra_digit) — 17 chars, UTC."""
    nd = 6 if extra_digit else 5
    scale = 10 ** nd
    day0 = float(mjd // 1)
    total = round((mjd - day0) * scale)
    if total >= scale:          # rounding carried into the next calendar day
        day0 += 1.0
        total -= scale
    t = Time(day0, format="mjd", scale="utc").ymdhms
    out = f"{t.year:04d} {t.month:02d} {t.day:02d}.{total:0{nd}d}"
    return f"{out:<17}"


def format_ra(ra_deg: float, extra_digit: bool = False) -> str:
    """'HH MM SS.ss ' (or 'HH MM SS.sss') — 12 chars."""
    nd = 3 if extra_digit else 2
    scale = 10 ** nd
    v = round((ra_deg % 360.0) / 15.0 * 3600.0 * scale) % (24 * 3600 * scale)
    h, rem = divmod(v, 3600 * scale)
    m, rem = divmod(rem, 60 * scale)
    out = f"{h:02d} {m:02d} {rem // scale:02d}.{rem % scale:0{nd}d}"
    return f"{out:<12}"


def format_dec(dec_deg: float, extra_digit: bool = False) -> str:
    """'sDD MM SS.s ' (or 'sDD MM SS.ss') — 12 chars."""
    nd = 2 if extra_digit else 1
    scale = 10 ** nd
    sign = "-" if dec_deg < 0 else "+"
    v = round(abs(dec_deg) * 3600.0 * scale)
    d, rem = divmod(v, 3600 * scale)
    m, rem = divmod(rem, 60 * scale)
    out = f"{sign}{d:02d} {m:02d} {rem // scale:02d}.{rem % scale:0{nd}d}"
    return f"{out:<12}"


def format_obs_line(desig: str, obs: ReportObs, code: str, extra_digit: bool = False) -> str:
    mag = "     "
    band = " "
    if obs.mag is not None and obs.mag == obs.mag:
        mag = f"{obs.mag:4.1f} "
        band = (obs.band or " ")[:1]
    line = (
        designation_columns(desig)
        + ("*" if obs.discovery else " ")
        + (obs.note1 or " ")[:1]
        + (obs.note2 or " ")[:1]
        + format_date(obs.mjd, extra_digit)
        + format_ra(obs.ra, extra_digit)
        + format_dec(obs.dec, extra_digit)
        + " " * 9
        + mag
        + band
        + " " * 6
        + f"{code:>3.3}"
    )
    if len(line) != 80:
        raise ValueError(f"internal error: observation line is {len(line)} chars")
    return line


NO_OBJECTS_LINE = "No moving objects detected"


def build_report(objects: list[ReportObject], header: ReportHeader,
                 extra_digit: bool = False) -> str:
    """Full MPC report text: header, observation lines, end marker.

    With no objects, the IASC "empty set" form is written instead
    (header + "No moving objects detected").
    """
    lines = header.lines()
    n = 0
    for obj in objects:
        for obs in sorted(obj.observations, key=lambda o: o.mjd):
            lines.append(format_obs_line(obj.designation, obs, header.cod, extra_digit))
            n += 1
    if n == 0:
        lines.append(NO_OBJECTS_LINE)
    lines.append(END_LINE)
    return "\n".join(lines) + "\n"


# --- checking a report before a human submits it -----------------------------

def check_report(text: str, min_obs: int = 3) -> list[str]:
    """Return a list of human-readable problems (empty list = looks fine)."""
    problems: list[str] = []
    lines = text.rstrip("\n").split("\n")
    if not lines or not lines[0].startswith("COD "):
        problems.append("first line must be 'COD <observatory code>'")
    if lines and lines[-1].strip() != END_LINE:
        problems.append(f"last line must be '{END_LINE}'")
    code = lines[0][4:7] if lines and lines[0].startswith("COD ") else None
    per_obj: dict[str, list[str]] = {}
    for n, ln in enumerate(lines, 1):
        if re.match(r"^[A-Z0-9]{3} ", ln) and len(ln) != 80 or ln.strip() == END_LINE:
            continue
        if len(ln) != 80:
            if re.match(r"^[A-Z0-9]{3} ", ln) or ln == NO_OBJECTS_LINE:
                continue
            problems.append(f"line {n}: observation line is {len(ln)} chars, must be 80")
            continue
        key = ln[:12].strip()
        per_obj.setdefault(key, []).append(ln)
        if code and ln[77:80] != code:
            problems.append(f"line {n}: observatory code {ln[77:80]!r} != header {code!r}")
        if not re.match(r"^\d{4} \d{2} \d{2}\.(\d{5} |\d{6})$", ln[15:32]):
            problems.append(f"line {n}: bad date field {ln[15:32]!r}")
        if not re.match(r"^\d{2} \d{2} \d{2}\.(\d{2} |\d{3})$", ln[32:44]):
            problems.append(f"line {n}: bad RA field {ln[32:44]!r}")
        if not re.match(r"^[+-]\d{2} \d{2} \d{2}\.(\d |\d{2})$", ln[44:56]):
            problems.append(f"line {n}: bad Dec field {ln[44:56]!r}")
    if not per_obj and NO_OBJECTS_LINE not in lines:
        problems.append("report contains no observations")
    for key, obs in per_obj.items():
        if len(obs) < min_obs:
            problems.append(f"{key}: only {len(obs)} observation(s); need at least {min_obs}")
        dates = [o[15:32] for o in obs]
        if len(set(dates)) != len(dates):
            problems.append(f"{key}: two observations share the same time")
    return problems
